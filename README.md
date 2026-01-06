# Unscented Kalman Filter (UKF)

A robust implementation of the Unscented Kalman Filter for non-linear state estimation. This class uses the Unscented Transform (UT) to pick sigma points and propagate them through non-linear functions, offering superior performance over the Extended Kalman Filter (EKF) for highly non-linear systems.

## Features

* **Sigma Point Generation:** Uses Van der Merwe scaled sigma points.
* **Customizable Dynamics:** Supports arbitrary non-linear state transition functions ($f(x)$) and measurement functions ($h(x)$).
* **Noise Handling:** Easy configuration for process noise ($Q$) and measurement noise ($R$).
* **Modular Design:** Designed as a standalone class for easy integration into existing projects.

## Mathematical Context

The UKF addresses the approximation issues of the EKF by neglecting the linearization step (Jacobian matrices). Instead, it uses a deterministic sampling approach.

The filter operates in two main steps:

1.  **Prediction:**
    Project the state ahead using the non-linear state transition function:
    $$\hat{x}_{k|k-1} = f(\hat{x}_{k-1|k-1}, u_k)$$
    $$P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q_k$$

2.  **Update (Correction):**
    Correct the predicted state using the measurement $z_k$:
    $$K_k = P_{xy} P_{yy}^{-1}$$
    $$\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k(z_k - h(\hat{x}_{k|k-1}))$$

## Installation

Ensure you have [NumPy](https://numpy.org/) installed, as this class relies heavily on matrix operations.

```bash
pip install numpy
