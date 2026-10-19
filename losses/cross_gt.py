"""GT-pose cross_unnorm monitoring loss.

The objective is defined once in the solver module (clockwise convention);
this module re-exports it under the losses package for the training
scripts' monitoring term. It is evaluated at the GT pose
(C_gt, psi_gt = -yaw_gt) and never enters the gradient path.
"""
from solver.cross_unnorm import cross_unnorm_loss

__all__ = ["cross_unnorm_loss"]
