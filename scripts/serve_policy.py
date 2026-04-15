import dataclasses
import enum
import logging
import socket
from typing import Literal

from openpi.policies import policy as _policy
from openpi.serving import websocket_policy_server
import tyro

import lap.policies.policy_config_adapter as _policy_config
from lap.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    # LAP-3B, generating actions via action expert
    LAP = "lap"
    # LAP-3B, generating language actions via autogressive sampling
    LAP_AR = "lap_ar"
    # LAP-3B fine-tuned on LIBERO
    LAP_LIBERO = "lap_libero"
    # Open-sourced baseline model from Physical Intelligence
    PI05_DROID = "pi05_droid"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str
    type: Literal["flow", "ar"] = "flow"


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LAP
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Training config name (e.g. "lap_rby1_eef"). Overrides --env if provided.
    policy_config: str | None = None
    # Checkpoint directory. Required when --policy-config is set.
    policy_dir: str | None = None
    # Policy type: "flow" or "ar". Only relevant with --policy-config.
    policy_type: Literal["flow", "ar"] = "flow"
    # ------ RTC (Real-Time Chunking) ------
    # Enable RTC guidance between consecutive action chunks.
    use_rtc: bool = False
    # Maximum guidance weight applied at the start of each denoising trajectory.
    rtc_max_guidance_weight: float = 1.0
    # Number of actions executed per chunk; determines the unexecuted tail for blending.
    rtc_execution_horizon: int = 8
    # Guidance weight schedule across the chunk prefix: "linear" | "exp" | "ones" | "zeros".
    rtc_schedule: str = "linear"


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LAP: Checkpoint(config="lap", dir="checkpoints/lap", type="flow"),
    EnvMode.LAP_AR: Checkpoint(config="lap", dir="checkpoints/lap", type="ar"),
    EnvMode.LAP_LIBERO: Checkpoint(config="lap_libero", dir="checkpoints/lap_libero", type="flow"),
    EnvMode.PI05_DROID: Checkpoint(config="pi05_droid", dir="gs://openpi-assets/checkpoints/pi05_droid", type="flow"),
}


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if args.policy_config is not None:
        if args.policy_dir is None:
            raise ValueError("--policy-dir is required when --policy-config is provided.")
        checkpoint = Checkpoint(config=args.policy_config, dir=args.policy_dir, type=args.policy_type)
    else:
        checkpoint = DEFAULT_CHECKPOINT.get(args.env)

    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {args.env}")

    config = _config.get_config(checkpoint.config)
    # Always disable stop_action_to_vlm_grad for inference — this flag is only
    # meaningful during training.
    config = dataclasses.replace(config, model=dataclasses.replace(config.model, stop_action_to_vlm_grad=False))

    if checkpoint.type == "ar":
        return _policy_config.create_trained_policy_ar(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    if checkpoint.type == "flow":
        return _policy_config.create_trained_policy(
            config, checkpoint.dir, default_prompt=args.default_prompt
        )
    raise NotImplementedError


def main(args: Args) -> None:
    policy = create_policy(args)
    # Apply RTC configuration post-construction.
    if args.use_rtc:
        rtc_config = {
            "enabled": True,
            "max_guidance_weight": args.rtc_max_guidance_weight,
            "execution_horizon": args.rtc_execution_horizon,
            "schedule": args.rtc_schedule,
        }
        policy._rtc_config = rtc_config
        policy._execute_chunk_size = args.rtc_execution_horizon
        policy._metadata["rtc_enabled"] = True
        policy._metadata["rtc_execution_horizon"] = args.rtc_execution_horizon
        logging.info(
            "RTC enabled (max_guidance_weight=%.2f, execution_horizon=%d, schedule=%s)",
            args.rtc_max_guidance_weight,
            args.rtc_execution_horizon,
            args.rtc_schedule,
        )

    policy_metadata = policy.metadata
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
