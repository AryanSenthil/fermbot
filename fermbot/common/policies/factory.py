import logging

from torch import nn 

from fermbot.common.datasets.fermbot_dataset import LeRobotDatasetMetadata
from fermbot.common.datasets.utils import dataset_to_policy_features
from fermbot.common.envs.configs import EnvConfig
from fermbot.common.envs.utils import env_to_policy_features
from fermbot.common.policies.act.configuration_act import ACTConfig
from fermbot.common.policies.pretrained import PreTrainedPolicy

from fermbot.configs.policies import PreTrainedConfig
from fermbot.configs.types import FeatureType 


def get_policy_class(name: str) -> PreTrainedPolicy:
    """
    Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""

    if name == "act":
        from fermbot.common.policies.act.modelling_act import ACTPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")
    

def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "act":
        return ACTConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")
    
def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if cfg.pretrained_path:
        # Load a pretrained policy and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    else:
        # Make a fresh policy.
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy