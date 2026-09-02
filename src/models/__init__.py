"""Neural models for spectrum allocation, prediction, and detection."""

from .supervised_learning import SupervisedPolicyNetwork, train_supervised

__all__ = ["SupervisedPolicyNetwork", "train_supervised"]