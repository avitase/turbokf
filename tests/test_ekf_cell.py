import math
import warnings

import torch

from turbokf import ekf_cell


class MyEKFCell(ekf_cell.EKFCell):
    def __init__(self):
        super().__init__(batch_size=2, state_size=2, measurement_size=1)
        self.u: float = -2.
        self.s: float = 20.
        self.d: float = 40.

    def motion_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        jac = torch.eye(2).unsqueeze(0).repeat(self.batch_size, 1, 1)
        jac[:, 0, 1] = t
        return jac

    def motion_model(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        new_state = torch.matmul(self.motion_model_jacobian(state, t), state.unsqueeze(2)) \
            .squeeze(2)
        new_state[:, 1] += t * self.u
        return new_state

    def measurement_model_jacobian(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        jac = torch.empty(self.batch_size, self.measurement_size, self.state_size)
        jac[0, 0] = torch.tensor([1., 0.])
        jac[1, 0] = torch.tensor([self.s / ((self.d - state[1, 0].item()) ** 2 + self.s ** 2), 0.])
        return jac

    def measurement_model(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pred = torch.empty(self.batch_size, self.measurement_size)
        pred[0, 0] = state[0, 0]
        pred[1, 0] = torch.arctan(self.s / (self.d - state[1, 0]))
        return pred


def compile(module, *, debug=False):
    if debug:
        warnings.warn('Skipping JIT compilation!')
        return module
    else:
        return torch.jit.script(module)


def test_ekf_cell():
    with torch.no_grad():
        cell = MyEKFCell()
        cell.process_noise.data = torch.sqrt(torch.tensor([
            [.1, 0., .1],
            [.1, 0., .1],
        ]))
        cell.measurement_noise.data = torch.sqrt(torch.tensor([
            [.05, ],
            [.01, ],
        ]))
        cell = compile(cell, debug=False)

        measurement = torch.tensor([
            [2.2, ],
            [math.pi / 6., ],
        ])
        state_init = torch.tensor([
            [0., 5.],
            [0., 5.],
        ])
        state_cov_init = torch.tensor([
            [[.01, 0.], [0., 1.]],
            [[.01, 0.], [0., 1.]],
        ])
        t = torch.tensor([.5, .5])

        pred, state, state_cov = cell.forward(measurement, state_init, state_cov_init, t)
        assert torch.allclose(pred, torch.tensor([
            [2.5, ],
            [.48996, ]
        ]))
        assert torch.allclose(state, torch.tensor([
            [2.2366, 3.6341],
            [2.5134, 4.0185]
        ]), atol=1e-4)
        assert torch.allclose(state_cov, torch.tensor([
            [[.0439, .061], [.061, .4902]],
            [[.3584, .4978], [.4978, 1.0969]],
        ]), atol=1e-4)
