"""A collection of utils to help with success detection."""

import numpy as np

_QVEL_TOL = 1e-3


def any_props_moving(props, physics, qvel_tol=_QVEL_TOL):
    for prop in props:
        vel = prop.get_velocity(physics)
        max_qvel = np.max(np.abs(vel[0]))
        if max_qvel >= qvel_tol:
            return True
    return False