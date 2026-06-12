# pylint: disable=too-many-arguments,too-many-locals,too-many-instance-attributes,too-many-positional-arguments
"""Learning rate scheduler module.

This module implements the custom learning rate schedule proposed in the original
"Attention is All You Need" paper. It ensures that the learning rate increases linearly
during a warmup phase and then decays proportionally to the inverse square root of the step.
"""

from torch.optim.optimizer import Optimizer


class NoamScheduler:
    """Implements the Noam learning rate scheduler.

    The learning rate is updated at every step according to the formula:
        `lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))`

    Attributes:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        d_model (int): The dimensionality of the model's embeddings.
        warmup_steps (int): The number of steps over which the learning rate increases linearly.
        current_step (int): The current training step counter.
    """

    def __init__(self, optimizer: Optimizer, d_model: int, warmup_steps: int) -> None:
        """Initializes the NoamScheduler.

        Args:
            optimizer (Optimizer): The PyTorch optimizer (typically Adam) attached to the model.
            d_model (int): The hidden dimension size of the Transformer.
            warmup_steps (int): Number of warmup steps before the inverse-square-root decay begins.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def step(self) -> None:
        """Updates the learning rate and increments the step counter.

        This method should be called once per batch update in the training loop.
        It updates the learning rate for all parameter groups in the attached optimizer.
        """
        self.current_step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        
        self.optimizer.step()

    def _get_lr(self) -> float:
        """Calculates the learning rate for the current step.

        Returns:
            float: The computed learning rate based on the Noam formula.
        """
        # The first term acts as a scaling factor based on model size.
        # The second term calculates either the linear warmup or the inverse square root decay.
        return (self.d_model**-0.5) * min(
            self.current_step**-0.5, self.current_step * (self.warmup_steps**-1.5)
        )

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized parameters.

        This is a convenience method that simply calls `zero_grad()` on the underlying optimizer.
        """
        self.optimizer.zero_grad()
