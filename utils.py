import numpy as np

def create_inflow(fluid, position, radius, velocity):
    inflow_velocity = np.zeros_like(fluid.velocity)
    inflow_smoke = np.zeros(fluid.shape)

    mask = np.linalg.norm(fluid.indices - np.array(position)[:, None, None], axis=0) <= radius
    inflow_velocity[1, mask] = -velocity
    inflow_smoke[mask] = 1

    return inflow_velocity, inflow_smoke