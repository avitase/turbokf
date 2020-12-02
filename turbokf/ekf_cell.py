import typing

import torch
import torch.nn as nn


class EKFCell(nn.Module):
    def __init__(self,
                 batch_size: int,
                 state_size: int,
                 measurement_size: int):
        """
        Args:
            batch_size: number of batches
            state_size: dimension of state
            measurement_size: dimension of measurements
        """
        super(EKFCell, self).__init__()
        self.batch_size = batch_size
        self.state_size = state_size
        self.measurement_size = measurement_size

        b, m, n = batch_size, state_size, measurement_size
        self.process_noise = nn.Parameter(torch.zeros(b, ((m + 1) * m) // 2),
                                          requires_grad=True)
        self.measurement_noise = nn.Parameter(torch.zeros(b, ((n + 1) * n) // 2),
                                              requires_grad=True)

    def motion_model_jacobian(self, state: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of the motion model

        Args:
            state: state vector (batched)
            ctrl: control-input

        Returns:
            Jacobian of the motion model
        """
        return torch.empty(self.batch_size, self.state_size, self.state_size)

    def motion_model(self, state: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Motion model

        Args:
            state: state vector (batched)
            ctrl: control-input

        Returns:
            Advanced states
        """
        return torch.matmul(self.motion_model_jacobian(state, ctrl), state)

    def measurement_model_jacobian(self, state: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Jacobian of the measurement model

        Args:
            state: state vector (batched)
            ctrl: control-input

        Returns:
            Jacobian of the measurement model
        """
        return torch.empty(self.batch_size, self.measurement_size, self.state_size)

    def measurement_model(self, state: torch.Tensor, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Measurement model

        Args:
            state: state vector (batched)
            ctrl: control-input

        Returns:
            Predicted measurements
        """
        return torch.matmul(self.measurement_model_jacobian(state, ctrl), state.unsqueeze(2)) \
            .squeeze(2)

    def innovation(self, measurement: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """
        Error of prediction w.r.t. measurement

        The default measurement pre-fit residual (aka innovation) should be the difference between
        actual measurement and prediction. Mind our sign convention: the measurement is the minuend
        and the prediction is the subtrahend.
        Overload this function if you want to reflect a special metric, e.g., between angles.

        Args:
            measurement: measurements (batched)
            prediction: predictions (batched)

        Returns:
            Batched error of prediction w.r.t. measurement
        """
        return measurement - prediction

    def tril_square(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Square of triangular matrices

        The square of the (triangular) matrix L is defined as L @ L.T. Elements [a, b, c, d, e, f]
        are passed linearized and are transformed into square matrices
         [[a, 0, 0, ...],
          [b, c, 0, ...],
          [d, e, f, ...]].

        Args:
            x: batched n * (n + 1) / 2 elements of the batched (n, n) triangular matrices L
            n: size of expanded matrix

        Returns:
            Batched product L @ L.T
        """
        idx = torch.tril_indices(n, n)
        y = torch.zeros((self.batch_size, n, n), dtype=x.dtype)
        y[:, idx[0], idx[1]] = x

        return torch.matmul(y, y.transpose(1, 2))

    def process_noise_cov(self, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Getter for the process noise covariance matrix

        Args:
            ctrl: control-input

        Returns:
            Process noise covariance matrix
        """
        return self.tril_square(self.process_noise, self.state_size)

    def measurement_noise_cov(self, ctrl: torch.Tensor) -> torch.Tensor:
        """
        Getter for the measurement noise covariance matrix

        Args:
            ctrl: control-input

        Returns:
            Measurement noise covariance matrix
        """
        return self.tril_square(self.measurement_noise, self.measurement_size)

    def forward(self,
                measurement: torch.Tensor,
                state: torch.Tensor,
                state_cov: torch.Tensor,
                ctrl: torch.Tensor = torch.tensor(0.)) \
            -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        EKF predict and update step

        State and state covariance are propagated according to a motion and measurement model, new
        measurement values are predicted, compared with actual measurements, and eventually used to
        update the state and the state covariance.

        Args:
            measurement: batch of new measurements as (b, m) tensor
            state: batch of current state as (b, n) tensor
            state_cov: batch of current state covariances as (b, n, n) tensor
            ctrl: control-input for motion model

        Returns:
            Predicted measurements, updated states, and updated state covariances
        """
        process_noise_cov = self.process_noise_cov(ctrl)
        measurement_noise_cov = self.measurement_noise_cov(ctrl)

        # predicted state
        new_state = self.motion_model(state, ctrl)

        # predict state covariance estimate
        motion_model_jacobian = self.motion_model_jacobian(state, ctrl)
        new_state_cov = torch.matmul(
            torch.matmul(motion_model_jacobian, state_cov),
            motion_model_jacobian.transpose(1, 2)
        ) + process_noise_cov

        # innovation
        prediction = self.measurement_model(new_state, ctrl)
        res = self.innovation(measurement, prediction)

        # innovation covariance
        measurement_model_jacobian = self.measurement_model_jacobian(new_state, ctrl)
        res_cov = torch.matmul(
            torch.matmul(measurement_model_jacobian, new_state_cov),
            measurement_model_jacobian.transpose(1, 2)
        ) + measurement_noise_cov
        kgain = torch.matmul(
            torch.matmul(new_state_cov, measurement_model_jacobian.transpose(1, 2)),
            res_cov.inverse()
        )

        return (
            prediction,
            new_state + torch.matmul(kgain, res.unsqueeze(2)).squeeze(2),
            torch.matmul(
                torch.eye(self.state_size).repeat(self.batch_size, 1, 1) -
                torch.matmul(kgain, measurement_model_jacobian),
                new_state_cov
            ),
        )
