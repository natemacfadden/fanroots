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
# Description: Propose an optimization step h->h+step in a fan using the
#              Gauss-Newton algorithm.
# -----------------------------------------------------------------------------

import numpy as np
import scipy as sp


def propose_gauss_newton(optimizer):
    """
    Propose a step h->h+step using the Gauss-Newton algorithm.

    See https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm.

    Solve F(h, x) = 0 using the Gauss-Newton algorithm, where h are
    the heights (point in the fan) and x are some optional other
    parameters.

    In case F is complex, we split the real/imaginary components,
    effectively solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]

    Agrees with Levenberg-Marquardt algorithm (LMA) when that
    algorithm has lambda=0.

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots optimizer containing the current state.

    Returns
    -------
    step : ndarray of shape (n,)
        The step in parameters, where n = len(heights) + len(other).
        If there are more parameters than just heights, the step in
        those parameters is concatenated.
    """
    # fetch the value of the function of interest F (and its Jacobian, J)
    F = optimizer.fct()
    J_h = optimizer.jac()

    # if there are other variables, split by the Jacobian
    if not optimizer.only_heights:
        J_h, J_other = J_h
    else:
        J_other = np.zeros(shape=(J_h.shape[0],0))

    # split by real, imaginary components
    if np.any(np.iscomplex(F)) or np.any(np.iscomplex(J_h))\
        or np.any(np.iscomplex(J_other)):
        F = np.concatenate((F.real, F.imag))
        J = np.block([
                [J_h.real, J_other.real],
                [J_h.imag, J_other.imag]
            ])
    else:
        J = np.hstack([J_h, J_other])

    # Solve J @ step = -F via QR-pivoted lstsq, not (J.T@J)@step = -J.T@F.
    # The normal-equations form squares cond(J) and loses signal at cond>=1e6.
    step, *_ = sp.linalg.lstsq(J, -F, lapack_driver='gelsy')

    return step
