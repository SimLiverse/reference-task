"""G1 flat-terrain locomotion — the shape a healthy task has.

Deliberately close to a real Isaac Lab manager-based task rather than a stub:
the repo gate reads these files with `ast`, so a fixture that is not really
Python would prove nothing about the rules.

The robot USD is *not* in this repo. It lives in the asset registry and the
blueprint names it — `assets.embodiment: assets/unitree/g1_29dof.urdf@v3` —
which is what lets the platform pin the exact bytes a run trained on.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.utils import configclass

import mdp


@configclass
class SceneCfg(InteractiveSceneCfg):
    """The world. The robot is attached by the launcher from the pinned asset."""

    num_envs = 4096
    env_spacing = 2.5

    # Scene-rooted, so Isaac Lab resolves them against each cloned environment.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        debug_vis=False,
    )


@configclass
class RewardsCfg:
    """Weights within two orders of magnitude of each other.

    Not an aesthetic preference: a term three orders below the largest cannot
    influence the gradient, so it is in the code and absent from the training.
    """

    track_lin_vel = RewardTermCfg(func=mdp.track_lin_vel_xy_exp, weight=1.0)
    track_ang_vel = RewardTermCfg(func=mdp.track_ang_vel_z_exp, weight=0.5)
    lin_vel_z = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    joint_torques = RewardTermCfg(func=mdp.joint_torques_l2, weight=-0.05)
    action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01)
    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )


@configclass
class TerminationsCfg:
    """An episode has to be able to end.

    Without a time-out the value function never bootstraps and the reward
    plateaus for reasons nothing in the logs explains.
    """

    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso_link")},
    )


@configclass
class EventCfg:
    """Domain randomisation — where sim2real lives or dies.

    Ranges are physical: mass and friction are non-negative, and friction stays
    inside what a real surface does.
    """

    physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.1),
        },
    )
    add_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"mass_range": (-2.0, 2.0), "operation": "add"},
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
    )


@configclass
class ObservationsCfg:
    base_lin_vel = ObservationTermCfg(func=mdp.base_lin_vel, noise_std=0.1)
    base_ang_vel = ObservationTermCfg(func=mdp.base_ang_vel, noise_std=0.2)
    joint_pos = ObservationTermCfg(func=mdp.joint_pos_rel, noise_std=0.01)
    joint_vel = ObservationTermCfg(func=mdp.joint_vel_rel, noise_std=1.5)
    actions = ObservationTermCfg(func=mdp.last_action)


@configclass
class G1FlatEnvCfg(ManagerBasedRLEnvCfg):
    scene = SceneCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    observations = ObservationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
