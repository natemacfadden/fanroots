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
# Description: Propose an optimization step h->h+step in a fan using gradient
#              descent.
# -----------------------------------------------------------------------------


def propose_gradient_descent(optimizer):
    """
    Propose a step h->h+step using gradient descent.

    Solve F(h, x) = 0 using gradient descent, where h are the heights
    (point in the fan) and x are some optional other parameters.

    In case F is complex, we split the real/imaginary components,
    effectively solving
        F'(h, x) = [Re(F(h,x)); Im(F(h,x))] = 0.
    This requires modifying
        J'(h, x) = [Re(J(h,x)); Im(J(h,x))]

    This really solves the least squares problem
        argmin S = argmin \\sum_i F_i(x)^2.
    It does so via stepping
        step = - lr \\grad S = - lr jac.T @ F(x)

    Parameters
    ----------
    optimizer : FanRoots
        The FanRoots instance containing the current state.

    Returns
    -------
    step : ndarray of shape (n,)
        The proposed step, where n = len(heights) + len(other).
        Contains the step in heights (and optionally other parameters,
        concatenated).
    """
    # compute the step
    step = -optimizer.grad()
    return step