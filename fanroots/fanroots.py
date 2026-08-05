# =============================================================================
#    Copyright (C) 2026  Nate MacFadden and contributors
#    Originally developed in the Liam McAllister Group at Cornell University.
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================

from __future__ import annotations

import copy
import joblib
import numpy as np
from datetime import datetime
import time

# plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.validator_cache import ValidatorCache

SymbolValidator = ValidatorCache.get_validator("scatter.marker", "symbol")
plotly_symbols = SymbolValidator.values
plotly_symbols = [i for i in plotly_symbols if isinstance(i,int) and i<100]

# warnings/logs/debugging
import sys
import traceback
import warnings
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="A worker stopped while some jobs were given to the executor.*"
)

# local imports
from fanroots.step_proposal import newton, gauss_newton, lma, gradient_descent
from fanroots.step_size import naive, backtracking_line_search, shrink, ternary
from fanroots.step_taking import flop, jump
from numpy.typing import ArrayLike
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type checkers only; the cytools types are resolved at runtime
    from cytools.vector_config import VectorConfiguration, Fan

# misc helpers
# ------------
class ResNormError(Exception):
    """
    Raised when residual norm computation fails (overflow).
    """
    pass

def always_true(*args, **kwargs):
    return True

def fanroots_from_state(state):
    obj = FanRoots.__new__(FanRoots)
    obj.__dict__.update(state)
    return obj

# main method
# -----------
class FanRoots:
    def __init__(self,
        # required
        vc: "VectorConfiguration",
        fct: Callable,
        jac: Callable,

        # halting
        tolerance:               float = 1e-4,
        min_step_size:           float = 1e-8,
        growth_demand_timescale: float = float('inf'),
        user_halting_fct:     Callable = None,

        # initial parameters
        heights0: ArrayLike = None,
        other0:   ArrayLike = None,
        triang:      "Fan" = None,
        kappa:    ArrayLike = None,

        # step proposal/taking
        step_proposal        = "newton",
        step_size_optimizer  = "shrink",
        step_taking_method   = "flop",
        step_taking_schedule = None,
        learning_rate: float = None,

        # momentum (semi-questionable)
        use_momentum:      bool = True,
        min_momentum:     float = 1e-6,
        max_momentum:     float = 1,
        momentum_penalty: float = 0.5,
        momentum_reward:  float = 1.5,

        # plotting/diagnostics
        plotting: bool       = False,
        plot_condition_num: bool = False,
        concerning_angle: float = np.pi/2,
        history_level: int = 0,
        
        # verbosity/error handling
        reckless_mode: bool = False,
        verbosity: int       = 0):
        """
        Root-find f(h, x=None) where h lives in the secondary fan.

        General purpose optimization method for finding the root of a
        function f(h, x=None) where h are heights living in the
        secondary fan and x are optional other arguments.

        Many functions will naturally operate on Kahler parameters.
        These can be found by mapping h -> t=GLSM@h.

        This can be used to:
            - solve inverse-volume problems (find Kahler parameters
              for which divisor volumes take a user-requested value)
            - AdS precursors (D_T W = 0)
            - and more

        The key difficulty and reason why this class is necessary is
        that the functional form of f can vary cone-by-cone.

        Parameters
        ----------
        vc : VectorConfiguration
            The vector configuration.
        fct : Callable
            The function to find root of. First argument is class
            instance. Second argument is height vector. If extra
            parameters are being carried around, they are concatenated
            to the height vector.
        jac : Callable
            The Jacobian of ``fct``. First argument is class instance.
            Second argument is height vector. If extra parameters are
            being carried around, they are concatenated to the height
            vector.
        tolerance : float, optional
            The tolerance to use. Accept a solution if
            |fct(h)|_2 < tolerance. Defaults to 1e-4.
        min_step_size : float, optional
            Minimum allowed step size. Defaults to 1e-8.
        growth_demand_timescale : float, optional
            Halt if residual doesn't decrease by 50% after this
            number of steps. Defaults to inf.
        user_halting_fct : Callable, optional
            A function taking a single argument (the FanRoots
            instance) that halts the optimization if it returns True.
        heights0 : ArrayLike of shape (N_vecs,), optional
            Starting value of h to use.
        other0 : ArrayLike of shape (N_other,), optional
            Starting value of x to use. If not provided, no
            x-dependence is assumed.
        triang : Fan, optional
            The initial triangulation, corresponding to
            vc.subdivide(heights). Can be computed on the fly, but
            that's less efficient for batches starting from the same
            triangulation.
        kappa : ArrayLike of shape (h11, h11, h11), optional
            The initial intersection numbers, corresponding to
            triang.intersection_numbers(in_basis=True,
            pushed_down=True, as_np_array=True). Can be computed on
            the fly, but that's less efficient for batches.
        step_proposal : str | Callable, optional
            The step proposal method. Set as a string: "newton" (for
            Newton's method), "grad" (for gradient descent), or "lma"
            (for Levenberg-Marquardt algorithm). N.B.: lma subsumes
            Gauss-Newton (lmbda=0) and gradient descent (lmbda->inf).
            Defaults to "newton".
        step_size_optimizer : str | Callable, optional
            An optimizer for setting the step size after a proposal by
            ``step_proposal``. Think backtracking line search.
            Defaults to "shrink".
        step_taking_method : str | object, optional
            The method for taking steps. Primarily ``bigstepper`` and
            ``flopper``. Primarily has efficiency implications but can
            also lead to somewhat different trajectories. Defaults to
            "flop".
        step_taking_schedule : list, optional
            A schedule setting the step taking method. Requires a list
            of tuples. Use method tuple[1] if check tuple[0] passes.
        learning_rate : float, optional
            Scale down the proposed step by this factor, before doing
            any step size optimization. Normally paired with trivial
            step size optimization (always accept proposed step).
        use_momentum : bool, optional
            Whether to use momentum. Defaults to True.
        min_momentum : float, optional
            Minimum allowed momentum. Defaults to 1e-6.
        max_momentum : float, optional
            Maximum allowed momentum. Defaults to 1.
        momentum_penalty : float, optional
            Factor by which to reduce momentum on bad steps.
            Defaults to 0.5.
        momentum_reward : float, optional
            Factor by which to increase momentum on good steps.
            Defaults to 1.5.
        history_level : int, optional
            The level at which we record history. Higher means more
            recording. Defaults to 0.
        plotting : bool, optional
            Whether to plot diagnostics. Defaults to False.
        plot_condition_num : bool, optional
            Whether to include condition number in plots.
            Defaults to False.
        concerning_angle : float, optional
            Angle threshold (radians) for recording large-angle steps.
            Defaults to pi/2.
        reckless_mode : bool, optional
            If True, do not halt on undefined behavior like 1/0.
            Defaults to False.
        verbosity : int, optional
            The verbosity level. Defaults to 0.
        """
        # parse string-specifications of step proposal and taking methods
        if step_proposal == "newton":
            self.step_proposal = newton.propose_newton
            if learning_rate is None:
                learning_rate = 1e-1
        elif (step_proposal == "grad") or (step_proposal == "gradient"):
            self.step_proposal = gradient_descent.propose_gradient_descent
            if learning_rate is None:
                learning_rate = 1e-3
        elif step_proposal == "lma":
            self.step_proposal = lma.propose_lma
            if learning_rate is None:
                learning_rate = 1e-1
        elif step_proposal == "gauss_newton":
            self.step_proposal = gauss_newton.propose_gauss_newton
            if learning_rate is None:
                learning_rate = 1e-1
        else:
            self.step_proposal = step_proposal
            if learning_rate is None:
                learning_rate = 1e-1

        # this is ignored if step_taking_schedule is set...
        if (step_taking_method == "flop") or (step_taking_method == "flopper"):
            step_taking_method = flop.FlopStep(max_num_flips=1)
        elif (step_taking_method=="jump") or (step_taking_method=="bigstepper"):
            step_taking_method = jump.JumpStep()
        # else: a step-taking object was passed in directly; use it as-is

        # step size optimizers
        if step_size_optimizer == "naive":
            self.step_size_optimizer = naive.naive_scaling
        elif step_size_optimizer == "bls":
            self.step_size_optimizer = backtracking_line_search
        elif step_size_optimizer == "shrink":
            self.step_size_optimizer = shrink
        elif step_size_optimizer == "ternary":
            self.step_size_optimizer = ternary
        else:
            self.step_size_optimizer = step_size_optimizer

        # the brain
        # ---------
        # square the tolerance since we compute the (|res|_2)**2
        self.tolerance     = tolerance**2
        self.min_step_size = min_step_size
        self.growth_demand_timescale = growth_demand_timescale
        if self.growth_demand_timescale is None:
            self.growth_demand_timescale = float('inf')

        # user halting
        self._user_halting_fct = user_halting_fct

        # the vector configuration defining the fan
        self.vc            = vc
        self.num_vecs      = len(vc.vectors())
        self.glsm          = self.vc.gale().T
        self.h11           = self.glsm.shape[0]

        # primary methods
        self._fct          = fct
        self._jac          = jac

        # step sizes
        self.learning_rate = learning_rate
        self.use_momentum  = use_momentum
        self.momentum      = 1
        self.min_momentum  = min_momentum
        self.max_momentum  = max_momentum
        self.momentum_penalty = momentum_penalty
        self.momentum_reward = momentum_reward

        # different methods for taking steps have different performance
        if step_taking_schedule is None:
            step_taking_schedule = [[always_true, step_taking_method]]
    
        self.step_taking_schedule = step_taking_schedule
        self.step_taking_method   = None
        self.step_taking_method_i = None

        # whether we only have heights as arguments, or if there are other
        # variables too
        self.only_heights         = (other0 is None)

        # the state
        # ---------
        if heights0 is None:
            heights0 = vc.subdivide().heights()

        # the variables being optimized
        self.heights = heights0
        self.other   = other0

        # the current phase/intersection numbers
        self.set_triang(triang)
        self.set_kappa(kappa)

        # initialize cached local information
        self.clear_local_cache(clear_momentum=True, clear_finished_state=True)

        # initialize diagnostic information
        self.clear_diagnostics()

        # history
        # -------
        self.history_level = history_level
        self.concerning_angle = concerning_angle

        # initialize the other history values
        self.clear_history()

        # plotting
        # --------
        self.plot_condition_num = plot_condition_num
        self.plotting   = plotting
        self.plot_names = [
            "\\sum_i res_i^2",
            "Step Size",
            "delta Heading",
            "Momentum",
            "t_1 vs t_0"
            ]
        if self.plot_condition_num:
            self.plot_names.append("Condition #")
        self.fig        = None
        self.fig_lines  = None # access to figure lines for easy updates

        # misc/kwargs
        # -----------
        if reckless_mode and (verbosity >= 0):
            print(
                "By setting `reckless_mode=True`, you are explicitly requesting"
            )
            print(
                "that the optimizer does NOT halt upon undefined behavior"
                " like `1/0`."
            )
            print()
            print(
                "In this case, there will be less certification that anything"
                " works..."
            )
            print(
                "If this is intended, decrease the verbosity level to `-1`"
                " to silence this warning"
            )
        self.reckless_mode = reckless_mode
        self.verbosity = verbosity

    # generic methods
    # ---------------
    def __next__(self):
        """
        Take a step and then return the status
        """
        if self.finished:
            raise StopIteration()

        return self.step()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result

        for k, v in self.__dict__.items():
            if k in ("fig", "fig_lines"):
                setattr(result, k, None)
            else:
                setattr(result, k, copy.deepcopy(v, memo))

        return result

    def copy(self):
        return copy.deepcopy(self)

    # history/init
    # ------------
    def set_triang(self, val=None):
        if val is None:
            self.triang = self.vc.subdivide(self.heights)
        else:
            self.triang = val

    def set_kappa(self, val=None):
        self._kappa_nz   = None
        self._kappa_vals = None

        if val is None:
            # `triang.kappa` is an alias for `intersection_numbers` in cytools,
            # so we can't reuse that name for our cached array. Use a private one.
            if not hasattr(self.triang, '_fanroots_kappa'):
                self.triang._fanroots_kappa = self.triang.intersection_numbers(
                    in_basis=True,
                    pushed_down=True,
                    as_np_array=True
                )
            self.kappa = self.triang._fanroots_kappa
        else:
            self.kappa = val
            self.triang._fanroots_kappa = val

    def clear_local_cache(self,
        clear_momentum=False,
        clear_finished_state=False
    ):
        self._fct_val  = None
        self._jac_val  = None
        self._condition_number = None
        self._res_norm = None
        self._grad     = None

        if clear_momentum:
            self.momentum = 1

        if clear_finished_state:
            self.finished_reason = "N/A"
            self.finished = False
            self.success  = None

    def clear_diagnostics(self):
        self.num_steps      = 0
        self.num_flips      = None
        self.num_fct_calls  = 0
        self._num_jac_calls = 0

        self._fct_time    = []
        self._jac_time    = []
        self._step_proposal_time = []
        self._step_taking_time   = []
        self._other_time  = []

        self._res_norm    = None # residual
        self.heading      = None # unit vector along the most recent step
        self.prev_heading = None # unit vector along previous step
        self.delta_heading= None # angle (radians) between prev heading and curr
        self.anc          = None # misc ancillary data
        
        self.last_proposal_size = None
        self.last_step_size     = None
        self.step_overestimation= None
        self.last_step_success  = True

    def clear_history(self):
        self.history = []
        self.history_largeangle = []
        self.history_triang  = []
        self.history_kappa   = []

        self.history_res_norm        = []
        self.history_proposal        = []
        self.history_successful_step = []
        
        self.history_anc    = []

    # properties
    # ----------
    @property
    def kahler(self):
        return self.glsm@self.heights

    @property
    def tau(self):
        t = self.kahler
        return 0.5 * (self.kappa@t)@t

    # status/diagnostics
    # ==================
    def get_state(self, deepcopy=False):
        if deepcopy:
            return copy.deepcopy(self.__dict__)
        return self.__dict__.copy()

    def load_state(self, state):
        self.__dict__.update(state)

    def load_heights_other(self, heights, other):
        self.clear_local_cache(clear_momentum=True, clear_finished_state=True)
        self.heights = heights
        self.other   = other
        self.set_triang()
        self.set_kappa()

    def get_status(self):
        """
        Return the current status
        """
        status = dict()

        # variables
        status['heights']   = self.heights.tolist()
        if not self.only_heights:
            status['other'] = self.other.tolist()

        # tolerance/completion
        status['finished']  = self.finished
        status['finished_reason'] = self.finished_reason
        status['res_norm']  = self.res_norm()
        status['tol_sq (compare to res_norm)'] = self.tolerance
        status['learning_rate'] = self.learning_rate
        
        # last step status
        status['last_step_size']    = self.last_step_size
        status['last_step_success'] = self.last_step_success

        status['heading']   = (
            self.heading.tolist() if self.heading is not None else None
        )
        status['anc']       = self.anc
        
        # misc
        status['step_taking_method_i'] = self.step_taking_method_i
        
        # history
        status['num_steps'] = self.num_steps
        status['num_flips'] = self.num_flips
        status['num_fct_calls'] = self.num_fct_calls
        status['num_jac_calls'] = self._num_jac_calls
        
        return status
    status = get_status

    # misc
    # ----
    def condition_number(self):
        """
        Basically, how well/ill condition the jacobian is for solving jac@x=b

        "If the condition number is not significantly larger than one, the
        matrix is well-conditioned, which means that its inverse can be
        computed with good accuracy. If the condition number is very large,
        then the matrix is said to be ill-conditioned. Practically, such a
        matrix is almost singular, and the computation of its inverse, or
        solution of a linear system of equations is prone to large numerical
        errors." - https://en.wikipedia.org/wiki/Condition_number
        """
        if self._condition_number is None:
            # computed lazily on first request and cached; reset by
            # clear_local_cache()
            J = self.jac()
            if len(J)==2:
                # above is equivalent to checking if self.only_heights=False
                J_h, J_other = J

                J = np.block([
                    [J_h.real, J_other.real],
                    [J_h.imag, J_other.imag]
                ])

            u, s, vh = np.linalg.svd(J)

            self._condition_number = s[0]/s[-1]
    
        return self._condition_number

    # timing
    # ------
    def timing_fct(self, N=100):
        """
        Call self.fct #N times, recording/returning the average time per call.

        If this is small compared to the cost of computing kappa, then certain
        step taking methods (e.g., flopper) are more attractive. If this is
        large, then other methods (e.g., bigstepper) are better.
        """
        fct_calls = self.num_fct_calls
        tic = time.time()
        for _ in range(N):
            self._fct(self, self.x())
        toc = time.time()
        self.num_fct_calls = fct_calls

        return (toc-tic)/N

    def timing_jac(self, N=100):
        """
        Call self.jac N times, recording/returning the average time per call.

        If this is small compared to the cost of computing kappa, then certain
        step taking methods (e.g., flopper) are more attractive. If this is
        large, then other methods (e.g., bigstepper) are better.
        """
        jac_calls = self._num_jac_calls
        tic = time.time()
        for _ in range(N):
            self._jac(self, self.x())
        toc = time.time()
        self._num_jac_calls = jac_calls

        return (toc-tic)/N

    # core methods
    # ============
    def kappa_nz(self):
        if self._kappa_nz is None:
            self._kappa_nz = np.nonzero(self.kappa)
        return self._kappa_nz

    def kappa_vals(self):
        if self._kappa_vals is None:
            self._kappa_vals = self.kappa[self.kappa_nz()]

        return self._kappa_vals

    def x(self):
        """
        Wrap up all parameters (heights and, optionally, other parameters) into
        a long vector of shape (n,), where n = len(heights) + len(other).
        """
        if self.only_heights:
            return self.heights
        else:
            return np.concatenate((self.heights, self.other))

    def fct(self, x=None, **kwargs):
        """
        Evaluate the function at the current location.

        If a location is provided, evaluate the function there instead.
        """
        tic = time.time()
        
        if x is not None:
            # override with manually input location
            self.num_fct_calls += 1
            out = self._fct(self, x, **kwargs)
        elif len(kwargs):
            # extra arguments passed... don't use caching
            self.num_fct_calls += 1
            out = self._fct(self, self.x(), **kwargs)
        else:
            # just evaluate locally (and cache it)
            if self._fct_val is None:
                self.num_fct_calls += 1
                self._fct_val = self._fct(self, self.x())
            out = self._fct_val

        toc = time.time()
        self._fct_time.append(toc-tic)
        if isinstance(out, tuple):
            out = tuple([np.array(f_i, copy=True, order='C') for f_i in out])
        else:
            out = np.array(out, copy=True, order='C')
        return out

    def jac(self, x=None, **kwargs):
        """
        Evaluate the Jacobian at the current location
        """
        tic = time.time()
        
        if x is not None:
            # override with manually input location
            self._num_jac_calls += 1
            out = self._jac(self, x, **kwargs)
        elif len(kwargs):
            # extra arguments passed... don't use caching
            self._num_jac_calls += 1
            out = self._jac(self, self.x(), **kwargs)
        else:
            # just evaluate locally (and cache it)
            if self._jac_val is None:
                self._num_jac_calls += 1
                self._jac_val = self._jac(self, self.x())
            out = self._jac_val

        toc = time.time()
        self._jac_time.append(toc-tic)
        if isinstance(out, tuple):
            out = tuple([np.array(J_i, copy=True, order='C') for J_i in out])
        else:
            out = np.array(out, copy=True, order='C')
        return out

    def res_norm(self, x=None, use_actual_kappa=False):
        """
        For checking convergence, we monitor the sum of squared residuals.
        For root finding, the residual is just the function itself.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", RuntimeWarning)

            # vvvv FUNCTION CALL vvvv
            # override with manually input location
            if use_actual_kappa:
                # store the real values
                h     = self.heights
                other = self.other
                tri   = self.triang
                kappa = self.kappa

                # fake with new values
                self.heights = x[:self.num_vecs]
                self.other   = x[self.num_vecs:]
                self.set_triang()
                self.set_kappa()

            f = self.fct(x)
            out = np.sum(np.square(f.real) + np.square(f.imag))

            if use_actual_kappa:
                # reset
                self.heights = h
                self.other   = other
                self.triang  = tri
                self.kappa   = kappa
            # ^^^^ FUNCTION CALL ^^^^

            for warn in w:
                if issubclass(warn.category, RuntimeWarning):
                    # set flags indicating that we've halted and why
                    self.finished_reason = "RuntimeWarning in res_norm"
                    self.finished = True
                    self.success  = False
                    self.last_step_success = False

                    # construct a specialized warning to raise
                    msg =   "RuntimeWarning caught in res_norm!\n"
                    msg += f"f.real: {f.real}\n"
                    msg += f"f.imag: {f.imag}\n"
                    msg += f"norm parts: {np.square(f.real), np.square(f.imag)}"

                    if self.reckless_mode:
                        print(w)
                    else:
                        raise ResNormError(msg)

        return out

    def grad(self):
        """
        Compute the gradient of the objective (sum of squared residuals).

        TBH, here, one assumes you use *half* the sum of squared residuals.
        This is just a scaling that gets absorbed into the learning rate...

        This is easy:
            grad_i = d_i 0.5 * \\sum_j F_j(x)^2.
                   = \\sum_j F(x)_j (d_i F(x)_j)
                   = jac.T @ F(x)
        """
        if self._grad is None:
            F = self.fct()

            if self.only_heights:
                J = self.jac()
            else:
                J = np.hstack(self.jac())

            if np.iscomplexobj(F) or np.iscomplexobj(J):
                J = np.vstack((J.real, J.imag))
                F = np.concatenate((F.real, F.imag))

            try:
                self._grad = J.T @ F
            except FloatingPointError:
                self._grad = None
                self.finished_reason = "FloatingPointError in grad()"
                self.finished = True
                self.success  = False

        return self._grad

    # steps
    # =====
    def step(self, num=1):
        """
        User request step(s).
        
        Basically just a wrapper of _step that enables cleaner plotting.
        """
        # update the plots
        if self.plotting:
            if self.fig is None:
                self._create_figures()
                self._add_traces()
            if not self.fig['displayed']:
                self._display_figures()

        # actually step
        out = self._step(num=num)

        # update plot variables
        if self.plotting and (self.fig is not None):
            self.fig['displayed'] = False
        return out

    def _step(self, num=1, return_full_state=False, paranoid=False):
        """
        Actual step computation.
        """
        # allow batching of steps
        if num == float('inf'):
            while not self.finished:
                out = self._step(return_full_state=False)

            # optionally return the full state for the final (redundant) call
            out = self._step(return_full_state=return_full_state)
            return out
        elif num > 1:
            # return only status for the earlier calls
            for _ in range(num-1):
                out = self._step(return_full_state=False)

            # optionally return the full state for the final call
            out = self._step(return_full_state=return_full_state)
            return out
        
        # single step
        # -----------
        if self.finished:
            # all done :)
            if return_full_state:
                out = self.get_state()
            else:
                out = self.get_status()
            if self.verbosity >= 1:
                print(f"Finished with status/state:\n{out}")
            return out

        try:
            # not done - compute next step
            if self.verbosity >= 2:
                print("Deciding upon a step to propose...")
            tic = time.time()
            step = self.compute_next_step()
            if self.only_heights:
                step_t = step
            else:
                step_t     = step[:self.num_vecs]
                step_other = step[self.num_vecs:]

            # shrink the step if it's significantly larger than previous one
            new_raw_step_size = np.linalg.norm(step_t)
            r = 1.5
            if (self.last_step_size is not None) and (
                new_raw_step_size > r*self.last_step_size
            ):
                rat = r*self.last_step_size/new_raw_step_size
                step_t *= rat
                if not self.only_heights:
                    step_other *= rat

            toc = time.time()
            self._step_proposal_time.append(toc-tic)

            # attempt the step
            # (update step taking method if necessary)
            self.update_step_taking_method()

            if self.verbosity == 2:
                print("Attempting the step...")
            elif self.verbosity > 2:
                print(f"Attempting the step from x={self.x()} to x+{step}...")
            
            tic = time.time()
            success, h, triang, anc = self.step_taking_method(self, step_t)
            if not self.only_heights:
                other = self.other + step_other*anc['step_scaling']
            else:
                other = None

            # check the step angle to ensure it wasn't too large
            prev_heading = self.heading
            heading      = h-self.heights
            heading_norm = np.linalg.norm(heading)
            if heading_norm > 0:
                heading /= heading_norm

            if prev_heading is not None:
                delta_heading = np.dot(heading, prev_heading)
                # avoid numerical issues
                delta_heading = np.arccos(
                    np.clip(delta_heading, -1.0, 1.0)
                )
            else:
                delta_heading = None

            # update info
            self.num_steps  += 1
            if 'num_flips' in anc:
                if self.num_flips is None:
                    self.num_flips = anc['num_flips']
                else:
                    self.num_flips += anc['num_flips']
            else:
                if self.num_flips is not None:
                    if self.verbosity >= 1:
                        print('performing a non-flip step after flip steps...')
                        print('spoils the num_flips tracker...')
                    self.num_flips = None
            toc = time.time()

            self._step_taking_time.append(toc-tic)
            if self.verbosity >= 2:
                if success:
                    print("Successful!")
                else:
                    print("Not successful :(")

            # clear the cache
            self.clear_local_cache(
                clear_momentum=False, clear_finished_state=False
            )

            # diagnostics for the step
            # ------------------------
            # compute the step size, heading
            self.last_proposal_size = np.linalg.norm(step_t)
            self.last_step_size     = np.linalg.norm(h-self.heights)
            if self.last_step_size == 0:
                self.step_overestimation = float('inf')
            else:
                self.step_overestimation = (
                    self.last_proposal_size/self.last_step_size
                )
            self.prev_heading = prev_heading
            self.heading      = heading
            self.delta_heading= delta_heading

            # momentum
            if self.use_momentum:
                if self.step_overestimation > 2:
                    self.momentum = max(
                        self.min_momentum,
                        self.momentum_penalty*self.momentum)
                elif self.step_overestimation < 1.01:
                    self.momentum = min(
                        self.max_momentum,
                        self.momentum_reward*self.momentum)

            # record the history
            if self.history_level >= 1:
                # record the current parameters
                self.history.append(self.x())

            if (self.delta_heading is not None and
                self.delta_heading >= self.concerning_angle):
                # record steps with large changes in angle
                self.history_largeangle.append(self.x())

            if self.history_level >= 2:
                # also record the step proposals and whether or not they were
                # successful
                self.history_proposal.append(step)
                self.history_successful_step.append(success)
            
            # update
            self.last_step_success = success
            self.heights = h
            self.other   = other
            self.set_triang(triang)
            self.set_kappa(getattr(triang, '_fanroots_kappa', None))
            self.anc     = anc

            if self.history_level >= 2:
                # also record the triangulation, intersection numbers, and
                # ancillary data
                self.history_triang.append(triang)
                self.history_kappa.append(self.kappa)
                self.history_anc.append(anc)
            if paranoid:
                assert triang.is_fine()
                dists = triang.secondary_cone_hyperplanes(via_circuits=True)@h
                if not min(dists)>0:
                    # we violated a hyperplane constraint
                    problematic = np.where(dists<1e-8)[0]

                    # multiple hyperplane constraints :( quit
                    if len(problematic) > 1:
                        raise ValueError(
                            "secondary cone didn't contain heights..."
                            f" violation={min(dists)};"
                            f" {len(problematic)} constraints violated"
                        )
                    # a SINGLE hyperplane constraint - push off wall
                    else:
                        n = triang.secondary_cone_hyperplanes(
                            via_circuits=True
                        )[problematic[0]]
                        self.heights = self.heights + 1e-8*n/np.linalg.norm(n)

            # update residual norm in history
            self.history_res_norm.append(self.res_norm())

            # if there was a warning in res_norm, that artificially ends
            # the run... only override finished if no such early halting
            if not self.finished:
                if (
                    len(self.history_res_norm) >= self.growth_demand_timescale
                    and (
                        0.50 * self.history_res_norm[
                            -self.growth_demand_timescale
                        ] <= self.history_res_norm[-1]
                    )
                ):
                    # did not see growth over the demanded timescale
                    self.finished_reason = "didn't meet demanded timescale"
                    self.finished = True
                    self.success = False
                else:
                    # compute the norm of the residuals, check if we're done
                    if self.res_norm()<self.tolerance:
                        self.finished_reason = "converged"
                        self.finished = True
                        self.success  = True
                    elif not self.last_step_success:
                        self.finished_reason = "stalled: step below min_step_size"
                        self.finished = True
                        self.success  = False

                if (not self.finished and
                    self._user_halting_fct is not None and
                    self._user_halting_fct(self) == True):
                    # not naturally done, but halted by user
                    self.finished = True
        except ResNormError:
            pass

        # plot status
        if self.plotting:
            self._update_plots()

        # all done - return status
        if return_full_state:
            return self.get_state()
        else:
            return self.get_status()

    def optimize(self):
        """
        Take steps until completed (res_norm<tolerance or failed)
        """
        return self.step(num=float('inf'))

    def compute_next_step(self):
        """
        Compute the next step and, optionally, optimize the size
        (assumed scaled by alpha \\in (0,1])
        """
        naive_step  = self.step_proposal(self)
        scaled_step = self.momentum * self.learning_rate * naive_step

        self.alpha  = self.step_size_optimizer(self, scaled_step)
        self.proposed_step   = self.alpha * scaled_step
        return self.proposed_step

    def update_step_taking_method(self):
        for i, (criteria, method) in enumerate(self.step_taking_schedule):
            if criteria(self):
                self.step_taking_method = method
                self.step_taking_method_i = i
                return

    # plotting
    # ========
    def _display_figures(self):
        """
        Display the current figure.
        """
        if self.fig is None:
            self._create_figures()

        # display is the IPython/Jupyter builtin; the plot is a plotly
        # FigureWidget that only renders inline in a notebook. warn rather
        # than crash the optimize loop if we are not in such an environment
        try:
            from IPython.display import display
            display(self.fig['fig'])
        except ImportError:
            warnings.warn(
                "plotting=True needs a Jupyter/IPython environment to show "
                "the figure; continuing without display"
            )
        self.fig['displayed'] = True

    def _create_figures(self, add_traces=True):
        """
        Create figures to plot to.
        """
        if self.fig is not None:
            return

        raw_fig = go.FigureWidget(make_subplots(rows=1, 
                                                cols=len(self.plot_names),
                                                subplot_titles=self.plot_names))
        
        # set size
        raw_fig.update_layout(
            showlegend=False,
            width=200*len(self.plot_names),
            height=400
        )

        self.fig = {'fig': raw_fig, 'displayed':False}
        self.fig['fig'].update_layout(showlegend=False)

        if add_traces:
            self._add_traces()

    def _add_traces(self, num=None):
        # add traces (empty initially)
        self.fig_lines = []

        if num is not None:
            num_str = str(num)
        else:
            num_str = ""

        for i, name in enumerate(self.plot_names):
            self.fig_lines.append([])

            if i == 4:
                # plot #4 is a scatter plot... we set x and y
                line = self.fig['fig'].add_trace(
                    go.Scattergl(x=[], y=[], text=[], mode='lines+markers',
                        marker=dict(
                            size=6,
                            symbol=(
                                num if num is None
                                else (num%len(plotly_symbols))
                            ),
                        ),
                        line=dict(width=1),
                        name = num_str
                    ),
                    row=1, col=1+i  # Plotly indexing starts at 1
                )['data'][-1]  # Store reference to the last added trace
            else:
                line = self.fig['fig'].add_trace(
                    go.Scattergl(
                        y=[], mode='lines',
                        line=dict(color='black'),
                        name=num_str
                    ),
                    row=1, col=1+i  # Plotly indexing starts at 1
                )['data'][-1]  # Store reference to the last added trace

            
            self.fig_lines[-1].append(line)
            
            if i==1:
                # extra trace for step size... the proposed step size...
                line = self.fig['fig'].add_trace(
                    go.Scattergl(
                        y=[], mode='lines',
                        line=dict(color='black', dash='dash'),
                        name=num_str + ' Proposed'
                    ),
                    row=1, col=1+i  # Plotly indexing starts at 1
                )['data'][-1]  # Store reference to the last added trace
                self.fig_lines[-1].append(line)
        
        self.fig['fig'].update_yaxes(type="log", row=1, col=1)
        self.fig['fig'].update_yaxes(type="log", row=1, col=2)

    def _update_plots(self):
        """
        Update the figures
        """
        if self.fig is None:
            self._create_figures()
        if not self.fig['displayed']:
            self._display_figures()
        
        N = len(self.fig_lines[0][0].y)
        with self.fig['fig'].batch_update():
            # Update line plots
            self.fig_lines[0][0].y += (self.res_norm(),)
            self.fig_lines[1][0].y += (self.last_step_size,)
            self.fig_lines[1][1].y += (self.last_proposal_size,)
            
            if self.delta_heading is not None:
                self.fig_lines[2][0].y += (self.delta_heading,)
            
            self.fig_lines[3][0].y += (self.momentum,)

            # Update scatter plot
            line = self.fig_lines[4][0]
            kahler = self.kahler
            line.x = list(line.x) + [kahler[0]]
            line.y = list(line.y) + [kahler[1]]
            line.text = list(line.text) + [N]
            n = len(line.x)

            line.marker.opacity = np.linspace(0.05, 1.0, n)
            if self.finished:
                if not self.success:
                    line.marker.opacity = 0
                else:
                    opacity = np.zeros(n)
                    opacity[-1] = 1
                    line.marker.opacity = opacity

            if self.plot_condition_num:
                self.fig_lines[5][0].y += (self.condition_number(),)

    # misc
    # ----
    def swarm(self, N, scale, max_N_misses=100, plotting=None,
              seed=None, verbosity=0):
        if seed is None:
            seed = int(datetime.now().timestamp())
        rng = np.random.default_rng(seed=seed)

        # output_object
        _swarm = []

        num_misses = 0
        swarmling = self.copy()
        while len(_swarm) < N:
            if verbosity > 0:
                print(len(_swarm), N)

            # generate the new heights
            new_heights = self.heights + rng.normal(
                scale=scale, size=len(self.heights)
            )

            # update the swarmling's variables according to new heights
            swarmling.heights = new_heights
            try:
                # try to make a new fine triangulation
                swarmling.triang = swarmling.vc.subdivide(swarmling.heights)
                assert swarmling.triang.is_fine()
            except Exception:
                # failed to build a fine triangulation... retry
                # (Exception, not bare except, so KeyboardInterrupt/SystemExit propagate)
                num_misses += 1
                if num_misses > max_N_misses:
                    raise Exception(
                        f"Tried more than {max_N_misses} to make swarmlings"
                        f"... scale={scale} likely too high"
                    )
                continue
            swarmling.set_kappa()

            # clear caches
            swarmling.clear_local_cache(
                clear_momentum=True, clear_finished_state=True
            )
            swarmling.clear_diagnostics()
            swarmling.clear_history()

            # save the add the swarmling to the swarm
            _swarm.append(swarmling)

            # prep the next swarmling
            swarmling = self.copy()

        if plotting is None:
            plotting = self.plotting

        _swarm = BatchOptimizer(
            optimizers = _swarm,
            plotting = plotting,
            verbosity = self.verbosity
        )

        return _swarm

class BatchOptimizer():
    def __init__(self,
        optimizers,
        plotting = False,
        verbosity = 0,):

        self.batch     = optimizers
        self.plotting  = plotting
        self.verbosity = verbosity

        self.finished  = [False] * len(self.batch)

        # plotting considerations
        # -----------------------
        # ensure all optimizers are on the same page
        for optimizer in self.batch:
            optimizer.plotting = plotting

        # make a single, uniform figure
        if self.plotting:
            print("Plotting only works with serial mode...")

            # clear old figures
            for optimizer in self.batch:
                optimizer.fig = None

            # create new figures
            optimizers[0]._create_figures(add_traces=False)
            for i, optimizer in enumerate(optimizers):
                time.sleep(0.08)
                optimizer.fig = optimizers[0].fig
                optimizer._add_traces(num=i)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.batch[index]
        elif isinstance(index, slice):
            return self.batch[index]
        else:
             raise TypeError("Invalid index")

    def get_status(self):
        return [optimizer.get_status() for optimizer in self.batch]

    def optimize(self, serial=False, backend='loky'):
        self.step(num=float('inf'), serial=serial, backend=backend)

    def step(self, num=1, serial=False, backend='loky'):
        outputs = []

        # force serial mode if there is a single optimizer
        # (the parallel mode has some annoying overhead)
        if serial == False:
            if len(self.batch) == 1:
                serial =True

        # run in series
        if serial:
            if self.plotting:
                self.batch[0]._display_figures()
            for i, optimizer in enumerate(self.batch):
                if self.verbosity >= 1:
                    print(f"Running optimizer #{i}...", end="")
                    t0 = time.time()
                optimizer._step(num=num)
                if self.verbosity >= 1:
                    t1 = time.time()
                    print(f" finished in {t1 - t0}s!")

        # run in parallel
        if not serial:
            if self.plotting:
                raise ValueError("Plotting is not supported in parallel mode.")

            # workers report success/exception via joblib's return tuples below;
            # if OOM-killed workers (which return nothing) become a problem, a
            # per-task heartbeat file written here and checked after Parallel()
            # would let us pinpoint which task died
            def run_and_capture(i):
                opt = self.batch[i]
                try:
                    state = opt._step(num=num, return_full_state=True)
                    return (i, state, None)
                except Exception as e:
                    # get extra details for the exception
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb_list = traceback.extract_tb(exc_traceback)
                    last_frame = tb_list[-1]

                    # attach details to exception object
                    e.exc_file = last_frame.filename
                    e.exc_func = last_frame.name
                    e.exc_lineno = last_frame.lineno
                    e.exc_line = last_frame.line

                    e.exc_type_name = exc_type.__name__
                    e.exc_value = exc_value

                    return (i, None, e)

            results = joblib.Parallel(
                n_jobs=-1, backend=backend, timeout=None, verbose=0
            )(
                joblib.delayed(run_and_capture)(i)
                for i, opt in enumerate(self.batch)
            )

            # ensure we have the expected number of outputs
            assert len(results) == len(self.batch)

            # read outputs
            # ------------
            for i, state, err in results:
                if err is None:
                    self.batch[i].load_state(state)
                    self.finished[i] = True
                else:
                    print(f"Task {i} failed:")
                    print(f"  File: {getattr(err, 'exc_file', 'N/A')}")
                    print(f"  Function: {getattr(err, 'exc_func', 'N/A')}")
                    print(f"  Line number: {getattr(err, 'exc_lineno', 'N/A')}")
                    print(f"  Line of code: {getattr(err, 'exc_line', 'N/A')}")
                    print(
                        f"  Exception type: "
                        f"{getattr(err, 'exc_type_name', type(err).__name__)}"
                    )
                    print(f"  Exception value: {err}")

        return
