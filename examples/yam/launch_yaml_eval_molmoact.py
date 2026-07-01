"""MolmoAct eval launcher.

Runs N rollouts, prompting for an instruction each time. Saves all three
cameras frame-by-frame (PNG) plus the joint trajectory (``episode.h5``) per
rollout, classifies rollouts via cv2 keypress (y/n/q) or a post-timeout
stdin prompt, and converts the session's labeled rollouts to a LeRobot v3.0
dataset on the way out.

CLI::

    python examples/yam/launch_yaml_eval_molmoact.py \
        --left_config_path examples/yam/configs/yam_left.yaml \
        --right_config_path examples/yam/configs/yam_right.yaml \
        -n 10
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import numpy as np
# import torch
import tyro
from omegaconf import OmegaConf

from camera_client import CameraClient
from gello_min.realsense_camera import RealSenseCamera, get_device_ids
from gello_min.env import RobotEnv
from eval_utils import (
    EvalRolloutSaver,
    LiveCameraView,
    RolloutOutcome,
    convert_session_to_lerobot,
    move_rollout,
    prompt_instruction,
    resolve_label,
)
from gello_min.robot import BimanualRobot
from gello_min.launch_utils import instantiate_from_dict, move_to_start_position
from gello_min.logging_utils import log_collect_demos
from molmoact_client import MolmoAct, MolmoActLocal

from action_lipo.lipo import ActionLiPo
from concurrent.futures import ThreadPoolExecutor, Future


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# atexit parking
# ---------------------------------------------------------------------------

_env: Optional[RobotEnv] = None
_bimanual: bool = False
_left_cfg: Optional[Dict[str, Any]] = None
_right_cfg: Optional[Dict[str, Any]] = None
_cleanup_done: bool = False


def _park_robot() -> None:
    """atexit hook: park arm back at start_joints regardless of exit path."""
    global _cleanup_done
    if _cleanup_done or _env is None:
        return
    _cleanup_done = True
    print("Parking robot at start position...")
    try:
        if _bimanual:
            move_to_start_position(_env, True, _left_cfg, _right_cfg)
        else:
            move_to_start_position(_env, False, _left_cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Parking failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    num_rollouts: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 1
    """How many rollouts to run in this session."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _build_env(
    args: Args,
) -> Tuple[RobotEnv, Dict[str, Any], Optional[Dict[str, Any]], bool]:
    """Build cameras + robot(s) + RobotEnv from the launch configs.

    Camera source is decided by the ``eval.camera_server.enabled`` flag in the
    left config:

    * ``true``  -> connect to the long-lived camera server over ZMQ. RealSense
      devices are owned by that server; this process never opens them.
    * ``false`` -> open ``RealSenseCamera`` objects in-process (legacy path).
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)
    bimanual = args.right_config_path is not None
    right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if bimanual else None
    )

    cam_server_cfg = ((left_cfg.get("eval") or {}).get("camera_server") or {})
    use_server = bool(cam_server_cfg.get("enabled", False))

    camera_dict = None
    camera_client = None
    if use_server:
        endpoint = str(cam_server_cfg.get("endpoint", "tcp://127.0.0.1:5555"))
        timeout_ms = int(cam_server_cfg.get("request_timeout_ms", 500))
        max_age = cam_server_cfg.get("max_frame_age_sec", 0.5)
        max_age = float(max_age) if max_age is not None else None
        print(f"[eval] Using camera server at {endpoint} (timeout={timeout_ms} ms)")
        camera_client = CameraClient(
            endpoint=endpoint,
            request_timeout_ms=timeout_ms,
            max_frame_age_sec=max_age,
        )
        if not camera_client.ping():
            raise RuntimeError(
                f"Camera server at {endpoint} did not respond to ping. "
                "Start it with scripts/start_camera_server.sh."
            )
    else:
        ids = get_device_ids()
        print(f"Found {len(ids)} camera devices: {ids}")
        camera_cfg = left_cfg["sensors"]["cameras"]
        camera_dict = {
            "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
            "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
            "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
        }

    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )
    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )
        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    env = RobotEnv(
        robot,
        control_rate_hz=left_cfg.get("hz", 30),
        camera_dict=camera_dict,
        camera_client=camera_client,
    )
    return env, left_cfg, right_cfg, bimanual


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------

JOINT_DIMS = np.array([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]) # skip gripper
MAX_DELTA_RAD = 0.22
MAX_DELTA_GRIPPER = 0.30
MAX_START_INDEX = 12
MIN_EXEC_STEPS = 15
MAX_BOUNDARY_DELTA = 0.12
QUERY_STEP = 15
EXECUTE_STEPS = 25
NUM_STEPS = 8
STEP_DT = 1 / 30
RESIDUAL_BLEND_STEPS = 4
CHUNK_SMOOTH_SIGMA = 1.5
BLEND_STEPS = 8
MAX_FRAME_AGE_MS = 500.0

class MotionLimiter:
    def __init__(
        self,
        *,
        enabled: bool,
        max_velocity: float,
        max_acceleration: float,
        include_grippers: bool,
    ) -> None:
        self.enabled = enabled
        self.max_velocity = max(float(max_velocity), 0.0)
        self.max_acceleration = max(float(max_acceleration), 0.0)
        self.include_grippers = include_grippers
        self.last_action: np.ndarray | None = None
        self.velocity = np.zeros(14, dtype=np.float32)
        self.acceleration = np.zeros(14, dtype=np.float32)

    def reset(self) -> None:
        self.last_action = None
        self.velocity.fill(0.0)
        self.acceleration.fill(0.0)

    def limit(self, target: np.ndarray, *, state14: np.ndarray) -> np.ndarray:
        target = np.asarray(target, dtype=np.float32)
        state = np.asarray(state14, dtype=np.float32)
        if not self.enabled or target.shape != (14,) or state.shape != (14,):
            return target

        base = state if self.last_action is None else self.last_action
        dims = np.arange(14) if self.include_grippers else JOINT_DIMS
        desired_velocity = target - base
        velocity = desired_velocity.copy()
        acceleration = velocity - self.velocity

        if self.max_acceleration > 0.0:
            acceleration[dims] = np.clip(
                acceleration[dims],
                -self.max_acceleration,
                self.max_acceleration,
            )

        velocity[dims] = self.velocity[dims] + acceleration[dims]
        if self.max_velocity > 0.0:
            velocity[dims] = np.clip(velocity[dims], -self.max_velocity, self.max_velocity)

        out = target.copy()
        out[dims] = base[dims] + velocity[dims]
        self.acceleration = velocity - self.velocity
        self.velocity = velocity
        self.last_action = out.copy()
        return out

def _clamp_delta(
    target: np.ndarray,
    current: np.ndarray,
    max_joint: float,
    max_gripper: float,
) -> tuple[np.ndarray, int]:
    delta = target - current
    limits = np.full_like(delta, max_joint)
    limits[-1] = max_gripper
    n = int(np.sum(np.abs(delta) > limits + 1e-9))
    return current + np.clip(delta, -limits, limits), n

def _chunk_switch_usable(
    new_chunk: np.ndarray,
    splice: int,
    last_action: np.ndarray,
) -> tuple[bool, str]:
    remaining = len(new_chunk) - splice
    if remaining < MIN_EXEC_STEPS:
        return False, f"too_short:{remaining}"
    if splice > MAX_START_INDEX:
        return False, f"splice_high:{splice}"
    boundary = float(np.max(np.abs(
        new_chunk[splice][JOINT_DIMS] - last_action[JOINT_DIMS]
    )))
    if boundary > MAX_BOUNDARY_DELTA:
        return False, f"boundary_high:{boundary:.3f}"
    return True, "ok"

def _smooth_chunk(actions: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0 or actions.shape[0] < 3:
        return actions
    radius = min(int(3.0 * sigma) + 1, actions.shape[0] // 2)
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    out = actions.copy()
    for d in JOINT_DIMS:
        padded = np.pad(actions[:, d].astype(np.float64), radius, mode="edge")
        out[:, d] = np.convolve(padded, kernel, mode="valid").astype(np.float32)
    return out

def run_one_rollout(
    env: RobotEnv,
    policy: MolmoAct,
    instruction: str,
    saver: EvalRolloutSaver,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
) -> RolloutOutcome:

    def fetch_chunk(obs):
        input_dict = policy.prepare_input(obs, instruction)
        input_dict["num_steps"] = NUM_STEPS
        return np.asarray(policy.inference(input_dict)["actions"])

    print("initing vars")

    executor = ThreadPoolExecutor(max_workers=1)
    next_chunk: Optional[Future] = None
    action_chunk: Optional[np.ndarray] = None
    last_action: Optional[np.ndarray] = None
    apply_step: Optional[int] = None
    chunk_index = 0

    print("initing motion limiter")

    motion_limiter = MotionLimiter(
        enabled=True,
        max_velocity=0.10,
        max_acceleration=0.035,
        include_grippers=False,
    )

    print("starting loop")
    for step in range(max_steps):
        t0 = time.perf_counter()

        # fire inference
        if next_chunk is None and (action_chunk is None or chunk_index >= QUERY_STEP):
            obs_snapshot = env.get_obs()
            next_chunk = executor.submit(fetch_chunk, obs_snapshot)
            apply_step = chunk_index

        # handle chunk arrival
        if next_chunk is not None and next_chunk.done():
            new_chunk = next_chunk.result()
            splice = chunk_index - apply_step

            if action_chunk is None:
                # always accept first chunk
                action_chunk = new_chunk
                chunk_index = 0
                next_chunk = None
            else:
                usable, reason = _chunk_switch_usable(new_chunk, splice, last_action)
                if usable:
                    # lipo -> residual blend -> guassian smooth
                    motion_limiter.reset()

                    horizon = len(new_chunk) - splice
                    blending_horizon = min(splice + BLEND_STEPS, horizon)

                    lipo = ActionLiPo(
                    solver="OSQP",
                    chunk_size=30,
                    blending_horizon=blending_horizon,
                    action_dim=14,
                    len_time_delay=splice,
                    dt=STEP_DT, # 30 hz
                    epsilon_blending=0.02,
                    epsilon_path=0.003,
                    )
                
                    solved, _ = lipo.solve(
                        new_chunk.astype(np.float64),
                        action_chunk.astype(np.float64),
                        len_past_actions=splice + BLEND_STEPS,
                    )

                    blended = np.asarray(solved if solved is not None else new_chunk, dtype=np.float32)

                    # if last_action is not None:
                    #     residual = last_action[JOINT_DIMS] - blended[splice, JOINT_DIMS]
                    #     blend_steps = min(RESIDUAL_BLEND_STEPS, len(blended) - splice)
                    #     for b in range(blend_steps):
                    #         t = b / blend_steps
                    #         weight = 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep
                    #         blended[splice + b, JOINT_DIMS] += weight * residual
                    
                    blended = _smooth_chunk(blended, sigma=CHUNK_SMOOTH_SIGMA)

                    action_chunk = blended
                    chunk_index = splice
                    next_chunk = None

                elif chunk_index >= 30:
                    # exhausted, discard and re-request
                    print(f"[discard] {reason}, re-requesting")
                    next_chunk = None
                    obs_snapshot = env.get_obs()
                    next_chunk = executor.submit(fetch_chunk, obs_snapshot)
                    apply_step = chunk_index
                else:
                    # defer
                    print(f"[defer] {reason}")

        # execute
        # print(chunk_index)
        if action_chunk is not None and chunk_index < len(action_chunk):
            action = np.asarray(action_chunk[chunk_index])
            state14 = env.get_robot_state()["joint_positions"]
            action = motion_limiter.limit(action, state14=state14)
            act_l, _ = _clamp_delta(action[:7], state14[:7], MAX_DELTA_RAD, MAX_DELTA_GRIPPER)
            act_r, _ = _clamp_delta(action[7:], state14[7:], MAX_DELTA_RAD, MAX_DELTA_GRIPPER)
            action14 = np.concatenate([act_l, act_r])
            last_action = action14.copy()
            print(f"Sending action at index {chunk_index}")
            # print(action14)
            # env.step_command_only(action14)
            chunk_index += 1

        # rate limit
        dt = time.perf_counter() - t0
        if dt < STEP_DT:
            time.sleep(STEP_DT - dt)
        
    return RolloutOutcome(end_reason="timeout", last_step=max_steps)

# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------

def run_session(
    env: RobotEnv,
    policy: MolmoAct,
    left_cfg: Dict[str, Any],
    right_cfg: Optional[Dict[str, Any]],
    bimanual: bool,
    num_rollouts: int,
) -> None:
    """Drive ``num_rollouts`` rollouts; convert the labeled set to LeRobot at the end.

    Catches ``KeyboardInterrupt`` so an in-progress rollout still gets flushed
    (as incomplete, with ``err.md``) and any rollouts already labeled in this
    session are still converted.
    """
    print("started session")

    storage = left_cfg["storage"]
    base_save_dir = Path(storage["base_dir"]) / "data" / storage["task_directory"]
    max_steps = int(left_cfg.get("max_steps", 1000))
    last_prompt = storage.get("language_instruction") or ""

    eval_cfg = left_cfg.get("eval") or {}
    cam_srv_cfg = eval_cfg.get("camera_server") or {}
    pub_endpoint = cam_srv_cfg.get("pub_endpoint") if cam_srv_cfg.get("enabled") else None
    live_view = LiveCameraView(
        enabled=bool(eval_cfg.get("live_view_enabled", True)),
        pub_endpoint=pub_endpoint,
        recv_timeout_ms=int(cam_srv_cfg.get("recv_timeout_ms", 100)),
    )

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labeled_rollouts: List[Path] = []
    saver: Optional[EvalRolloutSaver] = None
    outcome: Optional[RolloutOutcome] = None

    try:
        for rollout_idx in range(num_rollouts):
            move_to_start_position(env, bimanual, left_cfg, right_cfg)
            instruction = prompt_instruction(rollout_idx, num_rollouts, last_prompt)
            last_prompt = instruction

            rollout_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rollout_dir = base_save_dir / "eval" / rollout_timestamp
            saver = EvalRolloutSaver(
                rollout_dir=rollout_dir,
                instruction=instruction,
                max_workers=int(storage.get("saver_max_workers", 2)),
                png_compress_level=int(storage.get("png_compress_level", 1)),
            )

            print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
            print(f"  instruction: {instruction}")
            print(f"  rollout_dir: {rollout_dir}")

            outcome = run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction=instruction,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                live_view=live_view,
            )

            saver.flush()
            label = resolve_label(outcome)
            if label is not None:
                new_path = move_rollout(rollout_dir, label, base_save_dir)
                labeled_rollouts.append(new_path)
                print(f"  -> labeled '{label}': {new_path}")
            else:
                print(f"  -> kept in eval/: {rollout_dir}")

            saver = None
            outcome = None
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl-C received — saving incomplete rollout, then converting...")
        if saver is not None:
            try:
                saver.flush()
                saver.write_err(
                    reason="KeyboardInterrupt",
                    step=outcome.last_step if outcome else saver.num_steps,
                )
                print(f"  -> incomplete rollout saved: {saver.rollout_dir}")
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.exception("Failed to flush incomplete rollout: %s", exc)
    finally:
        live_view.close()
        _convert_if_any(labeled_rollouts, base_save_dir, session_timestamp, left_cfg)


def _convert_if_any(
    labeled_rollouts: List[Path],
    base_save_dir: Path,
    session_timestamp: str,
    left_cfg: Dict[str, Any],
) -> None:
    """Best-effort LeRobot conversion of this session's labeled rollouts."""
    if not labeled_rollouts:
        print("\n[session] No labeled rollouts this session — nothing to convert.")
        return

    lerobot_cfg = left_cfg.get("lerobot", {}) or {}
    output_dir = base_save_dir / "eval_lerobot_v30" / session_timestamp
    print(
        f"\n[session] Converting {len(labeled_rollouts)} labeled rollouts "
        f"to LeRobot v3.0 at {output_dir} ..."
    )
    try:
        convert_session_to_lerobot(
            session_rollout_dirs=labeled_rollouts,
            output_dir=output_dir,
            fps=int(lerobot_cfg.get("fps", left_cfg.get("hz", 30))),
            robot_type=str(lerobot_cfg.get("robot_type", "molmoact_dual_arm")),
            repo_id=str(lerobot_cfg.get("hf_repo_id", "local/eval_session")),
            action_mode=str(lerobot_cfg.get("action_mode", "next_joint_fields")),
            vcodec=str(lerobot_cfg.get("vcodec", "libsvtav1")),
            sanitize_online_viz_meta=bool(lerobot_cfg.get("sanitize_online_viz_meta", True)),
            image_writer_processes=int(lerobot_cfg.get("image_writer_processes", 0)),
            image_writer_threads=int(lerobot_cfg.get("image_writer_threads", 0)),
            parallel_encoding=bool(lerobot_cfg.get("parallel_encoding", True)),
        )
    except Exception as exc:  # noqa: BLE001 — keep raw rollouts even if conversion fails
        logger.exception("LeRobot conversion failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    atexit.register(_park_robot)

    args = tyro.cli(Args)
    if args.num_rollouts < 1:
        raise SystemExit("--num_rollouts must be >= 1")

    env, left_cfg, right_cfg, bimanual = _build_env(args)

    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg

    if bimanual:
        move_to_start_position(env, True, left_cfg, right_cfg)
    else:
        move_to_start_position(env, False, left_cfg)

    print(f"Launching robot: {env.robot().__class__.__name__}")
    print(f"Control loop: {left_cfg.get('hz', 30)} Hz")
    print(
        f"Rollouts this session: {args.num_rollouts}, "
        f"max_steps: {left_cfg.get('max_steps', 1000)}"
    )

    eval_cfg = left_cfg.get("eval") or {}
    mode = eval_cfg.get("mode", "server")
    if mode == "local":
        policy = MolmoActLocal(**(eval_cfg.get("local") or {}))
    elif mode == "server":
        policy = MolmoAct(server=eval_cfg.get("molmoact_server"))
    else:
        raise SystemExit(f"eval.mode must be 'server' or 'local', got {mode!r}")
    
    print("run session")
    run_session(
        env=env,
        policy=policy,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        bimanual=bimanual,
        num_rollouts=args.num_rollouts,
    )


if __name__ == "__main__":
    main()
