import numpy as np

def constant_velocity_model(state, dt):

    new_state = state.copy()
    vx = state[3]
    vy = state[4]
    vz = state[5]

    new_state[0] += vx*dt
    new_state[1] += vy*dt
    new_state[2] += vz*dt
    return new_state
