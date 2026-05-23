"""Abstract base class for all benchmark problems.

A ``BenchProblem`` is a self-contained optimization task with a fixed
parameter init, a forward function producing a scalar loss, and a
documented convergence criterion. The benchmark harness in
``bench/run_bench.py`` instantiates the problem with a seed, asks for its
initial parameters, then iteratively calls ``loss_and_grad`` and the
selected optimizer's ``step()``.

Subclasses must define ``name``, ``max_steps``, ``converged_tol`` as
class-level attributes, and must implement ``init_params``. They may
implement either:

  (a) ``forward(params) -> torch.Tensor`` — the default
      ``loss_and_grad`` will use ``torch.autograd.grad`` to compute
      gradients automatically, OR
  (b) ``loss_and_grad(params) -> tuple[float, list[torch.Tensor]]``
      directly, for problems whose gradients are computed by hand
      (e.g. analytic gradients on a quadratic for speed).

Implementing both is allowed; the explicit ``loss_and_grad`` takes
precedence over ``forward``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import torch


class BenchProblem(ABC):
    """Abstract benchmark problem contract.

    Class attributes (must be set by subclasses):
        name: unique short identifier, e.g. ``"p1_off_axis_quad"``
        max_steps: integer step budget for this problem
        converged_tol: loss threshold for the default ``converged`` check
    """

    name: str = ""
    max_steps: int = 0
    converged_tol: float = 0.0

    def __init__(self, seed: int, device: str = "cpu") -> None:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be int, got {type(seed).__name__}")
        self.seed = seed
        self.device = torch.device(device)
        # Subclasses are expected to use ``self._generator`` for any
        # randomness so reproducibility holds. Generator stays on CPU;
        # subclasses move sampled tensors to ``self.device``.
        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

    @abstractmethod
    def init_params(self) -> List[torch.Tensor]:
        """Return the list of parameter tensors under optimization.

        Each returned tensor must have ``requires_grad=True`` and be a
        leaf tensor (so that ``torch.autograd.grad`` can populate its
        gradient). The list order is canonical and must match the order
        of gradients returned by ``loss_and_grad``.
        """
        raise NotImplementedError

    def forward(self, params: List[torch.Tensor]) -> torch.Tensor:
        """Return scalar loss tensor for the given parameter list.

        Default implementation raises ``NotImplementedError`` — subclasses
        either implement this (and inherit the default ``loss_and_grad``)
        or implement ``loss_and_grad`` directly.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement either forward() "
            "or loss_and_grad()"
        )

    def loss_and_grad(
        self, params: List[torch.Tensor]
    ) -> Tuple[float, List[torch.Tensor]]:
        """Compute scalar loss and gradients via autograd by default.

        Subclasses may override this directly for analytic gradients.
        The default uses ``torch.autograd.grad`` against ``forward``.
        Returns ``(loss_value, gradient_list)`` where the gradient list
        is aligned to ``params``.
        """
        loss = self.forward(params)
        if not isinstance(loss, torch.Tensor):
            raise TypeError(
                f"{type(self).__name__}.forward must return a Tensor, "
                f"got {type(loss).__name__}"
            )
        if loss.dim() != 0:
            raise ValueError(
                f"{type(self).__name__}.forward must return a scalar; "
                f"got shape {tuple(loss.shape)}"
            )
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )
        return float(loss.detach().item()), list(grads)

    def converged(self, current_loss: float, step: int) -> bool:
        """Default convergence check: loss below ``converged_tol``.

        Subclasses can override for problems without a clear zero (e.g.
        a saddle problem where the target is "escape", not "minimize").
        """
        if not isinstance(current_loss, float):
            current_loss = float(current_loss)
        return current_loss < self.converged_tol
