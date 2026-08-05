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
#
# -----------------------------------------------------------------------------
# Description: Attempt an optimization step t->t+step in a fan, for which the
#              objective function depends on the intersection numbers, kappa.
#
#              Update the objective function/kappa by recomputing from scratch
#              kappa at t+step.
# -----------------------------------------------------------------------------


class JumpStep:
    """
    Step method that advances through the fan by jumping directly.

    Attempt to take a step t->t+r*step for r=1.

    Restrict t to live in the (pushed down) secondary subfan of fine,
    regular triangulations (i.e., valid Kahler parameters). Failure
    modes are:
        1) t is outside the secondary subfan (doesn't define a
           subdivision)
        2) t defines a non-triangulation subdivision
        3) t defines a non-fine triangulation

    If a step fails for any of the above reasons, try
        r->0.5*r
        t->t+r*step
    Quit if r<min_step_size.

    Notes
    -----
    When called, the instance accepts:
        optimizer : FanRoots
            The FanRoots instance with current state (heights, triang,
            min_step_size, verbosity, etc.).
        step : ndarray of shape (N_vecs,)
            The requested step h->h+step.

    And returns:
        success : bool
            Whether some r>0 was found such that t->t+r*step is valid.
        h : ndarray of shape (N_vecs,)
            The heights after the step.
        triang : Fan
            The triangulation at the new location. Kappa accessible
            via triang.kappa if precomputed.
        anc : dict
            Ancillary data.
    """

    def __init__(self):
        return

    def __call__(self, optimizer, step):
        kappa  = None
        triang = None
        h = optimizer.heights + step

        # keep on try to step t->t+r*step
        # for decreasing r, until the step is accepted (or r<min_step_size)
        half_r             = 0.5
        half_min_step_size = 0.5*optimizer.min_step_size

        while True:
            # try the step,
            # see if the step leaves us somewhere valid
            success   = True
            fail_mode = None
            try:
                triang = optimizer.vc.subdivide(heights=h)
                # if we aren't an F(R)T, need to walk back
                if not triang.is_fine():
                    triang    = None
                    success   = False
                elif not triang.is_triangulation():
                    triang    = None
                    success   = False

            except ValueError:
                # heights don't even define a regular fan
                triang    = None
                success   = False

            # break if the step was OK
            if success:
                break

            # step was NOT OK - step back by 0.5*r*step
            h -= half_r*step
            half_r /= 2
            if half_r<half_min_step_size:
                fail_mode = f"step scaling={2*half_r} is too small"
                if optimizer.verbosity >= 1:
                    print(f"Failed because {fail_mode}")
                success   = False
                break

        # all set :)
        anc = {
            'step_scaling': 2*half_r,
            'failure_mode': fail_mode, # None indicates success
            }
        
        if triang is None:
            h      = optimizer.heights
            triang = optimizer.triang

        return success, h, triang, anc
