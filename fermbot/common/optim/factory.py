from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from fermbot.common.policies.pretrained import PreTrainedPolicy
from fermbot.configs.train import TrainPipelineConfig


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler