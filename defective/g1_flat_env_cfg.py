"""The same task, carrying one instance of every defect the static rules see.

Each is a mistake taken from how these files actually go wrong, not an invented
one. None of them is a syntax error and none would fail to import — that is the
point. Every defect here produces a run that starts, trains, reports a number,
and is quietly worthless.

Deliberately absent, and each absence is its own finding:
  - no `TerminationsCfg` — episodes never end
  - no `EventCfg`        — no domain randomisation at all
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

import mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    num_envs = 4096
    env_spacing = 2.5

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
    )

    # DEFECT: not scene-rooted. Isaac Lab resolves sensor paths against the
    # scene root, so this attaches to nothing and returns empty frames for the
    # entire run — with no error, because an unattached sensor is legal.
    head_camera = CameraCfg(
        prim_path="Robot/head_link/camera",
        width=640,
        height=480,
    )


@configclass
class RewardsCfg:
    track_lin_vel = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.0)

    # DEFECT: zeroed while debugging and never restored. It is reported as part
    # of the reward and contributes nothing to it.
    track_ang_vel = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.0)

    # DEFECT: four orders of magnitude below the largest term. Present in the
    # code, absent from the gradient.
    joint_torques = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.00005)

    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )


@configclass
class ObservationsCfg:
    base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel)
    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel)
    actions = ObservationTermCfg(func=mdp.last_action)


@configclass
class G1FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg()
    rewards = RewardsCfg()
    observations = ObservationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
