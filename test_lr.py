"""Code to test the learning rate functions in nets.py"""

# pylint: disable=import-error, invalid-name

from torch.optim import SGD
import torch
from shrinkbench.models import resnet20
from nets import get_lr_schedule


def test_lr(
    num_epochs: int = 10,
    initial_rate: float = 1,
    threshold: float = 1e-10,
    method: str=None,
    verbose: bool = True,
) -> bool:
    """Checks if the actual and expected learning rate is the same for a torch optimizer."""

    f = get_lr_schedule(initial_rate, num_epochs, method=method)
    model = resnet20()
    optim = SGD(
        model.parameters(), **{"momentum": 0.9, "nesterov": True, "lr": initial_rate}
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, f)

    for i in range(num_epochs):
        lr_theoretical = initial_rate * f(i)
        lr_empirical = scheduler.get_last_lr()[0]
        if verbose:
            print(
                f"Epoch {i}, theoretical_lr {lr_theoretical}, empirical_lr: {lr_empirical}"
            )
        if abs(lr_empirical - lr_theoretical) > threshold:
            print(
                f"Error on Epoch {i}, theoretical_lr {lr_theoretical}, empirical_lr: {lr_empirical}"
            )
            return False
        optim.step()
        scheduler.step()

    return True


if __name__ == "__main__":
    print(test_lr(initial_rate=0.1))
    print(test_lr(300, 1e-4, method="fixed"))
