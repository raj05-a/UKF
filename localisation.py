

import numpy as np
import matplotlib.pyplot as plt
from ukf import UKF
from process_model import constant_velocity_model

def run_localization(filename):
    num_states = 6
    Q=np.eye(num_states)*0.1

    init_state = np.zeros(num_states)
    init_covar = np.eye(num_states)
    ukf = UKF(num_states, Q, init_state, init_covar, 0.1, 0, 2, constant_velocity_model)

    last_time = 0.0
    results=  []

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if not parts: continue

            timestamp = float(parts[0])
            sensor_type = parts[1].strip()

            dt = timestamp - last_time
            last_time = timestamp

            if dt > 0:
                ukf.predict(dt)

            if sensor_type == 'DEPTH':
                z_meas = float(parts[2])
                ukf.update(states=[2], data=[z_meas], r_matrix=np.eye(1) * 0.5)

            elif sensor_type == 'GPS':
                x_meas = float(parts[2])
                y_meas = float(parts[3])
                ukf.update(states=[0, 1], data=[x_meas, y_meas], r_matrix=np.eye(2) * 2.0)

            current_state = ukf.get_state()
            results.append([timestamp, current_state[0], current_state[1], current_state[2]])

        return np.array(results)

    data = run_localization('sensor_data.txt')

    plt.plot(data[:, 0], data[:, 3], label='Filtered Z Altitude')
    plt.xlabel('Time')
    plt.ylabel('Depth (m)')
    plt.legend()
    plt.show()