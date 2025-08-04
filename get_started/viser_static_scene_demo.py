"""This script is used to test the static scene."""

from __future__ import annotations

from typing import Literal

try:
    import isaacgym  # noqa: F401
except ImportError:
    pass


import rootutils
import torch
import tyro
from loguru import logger as log
from rich.logging import RichHandler

rootutils.setup_root(__file__, pythonpath=True)
log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.cfg.sensors import PinholeCameraCfg
from metasim.constants import PhysicStateType, SimType
from metasim.utils import configclass
from metasim.utils.setup_util import get_sim_env_class


@configclass
class Args:
    """Arguments for the static scene."""

    robot: str = "franka"

    ## Handlers
    sim: Literal["isaacgym", "isaaclab", "genesis", "pybullet", "sapien2", "sapien3", "mujoco", "mjx"] = "isaaclab"

    ## Others
    num_envs: int = 1
    headless: bool = False

    def __post_init__(self):
        """Post-initialization configuration."""
        log.info(f"Args: {self}")


args = tyro.cli(Args)

# initialize scenario
scenario = ScenarioCfg(
    robots=[args.robot],
    try_add_table=False,
    sim=args.sim,
    headless=args.headless,
    num_envs=args.num_envs,
)

# add cameras
scenario.cameras = [PinholeCameraCfg(width=1024, height=1024, pos=(1.5, -1.5, 1.5), look_at=(0.0, 0.0, 0.0))]

# add objects
scenario.objects = [
    PrimitiveCubeCfg(
        name="cube",
        size=(0.1, 0.1, 0.1),
        color=[1.0, 0.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveSphereCfg(
        name="sphere",
        radius=0.1,
        color=[0.0, 0.0, 1.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    PrimitiveCylinderCfg(
        name="cylinder",
        radius=0.1,
        height=0.2,
        color=[0.0, 1.0, 0.0],
        physics=PhysicStateType.RIGIDBODY,
    ),
    RigidObjCfg(
        name="bbq_sauce",
        scale=(2, 2, 2),
        physics=PhysicStateType.RIGIDBODY,
        usd_path="get_started/example_assets/bbq_sauce/usd/bbq_sauce.usd",
        urdf_path="get_started/example_assets/bbq_sauce/urdf/bbq_sauce.urdf",
        mjcf_path="get_started/example_assets/bbq_sauce/mjcf/bbq_sauce.xml",
    ),
]


log.info(f"Using simulator: {args.sim}")
env_class = get_sim_env_class(SimType(args.sim))
env = env_class(scenario)

init_states = [
    {
        "objects": {
            "cube": {
                "pos": torch.tensor([0.3, -0.2, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "sphere": {
                "pos": torch.tensor([0.4, -0.6, 0.05]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "cylinder": {
                "pos": torch.tensor([0.5, -0.8, 0.15]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
            "bbq_sauce": {
                "pos": torch.tensor([0.7, -0.3, 0.14]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            },
        },
        "robots": {
            "franka": {
                "pos": torch.tensor([0.0, 0.0, 0.0]),
                "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                "dof_pos": {
                    "panda_joint1": 0.0,
                    "panda_joint2": -0.785398,
                    "panda_joint3": 0.0,
                    "panda_joint4": -2.356194,
                    "panda_joint5": 0.0,
                    "panda_joint6": 1.570796,
                    "panda_joint7": 0.785398,
                    "panda_finger_joint1": 0.04,
                    "panda_finger_joint2": 0.04,
                },
            },
        },
    }
]
obs, extras = env.reset(states=init_states)


# ========================================================================
# viser visuallization
# ========================================================================
from get_started.viser_util import ViserVisualizer


def extract_states_from_init(init_states, key):
    """
    key: "objects" or "robots"
    Return: dict[name] = {"pos": ..., "rot": ..., "dof_pos": ...}
    """
    result = {}
    if init_states and len(init_states) > 0:
        state = init_states[0]
        if key in state:
            for name, item in state[key].items():
                state_dict = {}
                if "pos" in item and item["pos"] is not None:
                    state_dict["pos"] = (
                        item["pos"].cpu().numpy().tolist() if hasattr(item["pos"], "cpu") else list(item["pos"])
                    )
                if "rot" in item and item["rot"] is not None:
                    state_dict["rot"] = (
                        item["rot"].cpu().numpy().tolist() if hasattr(item["rot"], "cpu") else list(item["rot"])
                    )
                if "dof_pos" in item and item["dof_pos"] is not None:
                    state_dict["dof_pos"] = item["dof_pos"]
                result[name] = state_dict
    return result


# initialize the viser server
visualizer = ViserVisualizer(port=8080)
visualizer.add_grid()
visualizer.add_frame("/world_frame")

# extract states from objects and robots
default_object_states = extract_states_from_init(init_states, "objects")
default_robot_states = extract_states_from_init(init_states, "robots")

# visualize all objects and robots
visualizer.visualize_scenario_items(scenario.objects, default_object_states)
visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

log.info("Viser has been initialized, visit http://localhost:8080 to view the scene!")

# scene info string
scene_info = ["The static scene includes:"]
for obj in scenario.objects:
    scene_info.append(f"  • {obj.name} ({type(obj).__name__})")
for robot in scenario.robots:
    scene_info.append(f"  • {robot.name} ({type(robot).__name__})")

# print for debugging
log.info("\n".join(scene_info))

# Enable camera controls
# The camera can be controlled via the GUI sliders
visualizer.enable_camera_controls(
    initial_position=[1.5, -1.5, 1.5],
    render_width=1024,
    render_height=1024,
    look_at_position=[0, 0, 0],
    initial_fov=71.28,
)

# if you want to enable trajectory playback, uncomment the following line
# visualizer.enable_trajectory_playback()

# if you want to enable joint control, uncomment the following line
# visualizer.enable_joint_control()

while True:
    pass
