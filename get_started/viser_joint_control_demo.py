"""Demo script for robot joint control in viser."""

from __future__ import annotations

import rootutils

rootutils.setup_root(__file__, pythonpath=True)

import torch
from loguru import logger as log
from rich.logging import RichHandler

from get_started.viser_util import ViserVisualizer
from metasim.cfg.objects import PrimitiveCubeCfg, PrimitiveCylinderCfg, PrimitiveSphereCfg, RigidObjCfg
from metasim.cfg.scenario import ScenarioCfg
from metasim.constants import PhysicStateType

log.configure(handlers=[{"sink": RichHandler(), "format": "{message}"}])


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


def main():
    """Demo robot joint control functionality."""

    # Create a simple scenario with a robot
    scenario = ScenarioCfg(
        robots=["franka", "h1"],  # You can try other robots like "ur5e", "kinova_gen3", etc.
        try_add_table=True,  # Add table for context
        sim="isaaclab",  # or your preferred simulator
        headless=True,  # Run headless since we're only using viser
        num_envs=1,
    )

    # Add some objects for visualization context
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

    # Initialize visualizer
    visualizer = ViserVisualizer(port=8080)
    visualizer.add_grid()
    visualizer.add_frame("/world_frame")

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
                "h1": {
                    "pos": torch.tensor([0.0, 0.0, 1.0]),
                    "rot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
                    "dof_pos": {
                        "left_hip_yaw": 0.0,
                        "left_hip_roll": 0.0,
                        "left_hip_pitch": -0.4,
                        "left_knee": 0.8,
                        "left_ankle": -0.4,
                        "right_hip_yaw": 0.0,
                        "right_hip_roll": 0.0,
                        "right_hip_pitch": -0.4,
                        "right_knee": 0.8,
                        "right_ankle": -0.4,
                        "torso": 0.0,
                        "left_shoulder_pitch": 0.0,
                        "left_shoulder_roll": 0.0,
                        "left_shoulder_yaw": 0.0,
                        "left_elbow": 0.0,
                        "right_shoulder_pitch": 0.0,
                        "right_shoulder_roll": 0.0,
                        "right_shoulder_yaw": 0.0,
                        "right_elbow": 0.0,
                    },
                },
            },
        }
    ]

    default_object_states = extract_states_from_init(init_states, "objects")
    default_robot_states = extract_states_from_init(init_states, "robots")

    visualizer.visualize_scenario_items(scenario.objects, default_object_states)
    visualizer.visualize_scenario_items(scenario.robots, default_robot_states)

    # Enable camera controls
    visualizer.enable_camera_controls(
        initial_position=[3, 3, 1.0],
        render_width=512,
        render_height=512,
        look_at_position=[0, 0, 0.5],  # Look at robot level
        initial_fov=50.0,
    )

    # Enable joint control
    visualizer.enable_joint_control()

    log.info("Viser server started at http://localhost:8080")
    log.info("Robot Joint Control Demo Usage:")
    log.info("1. Open 'Joint Control' panel in the GUI")
    log.info("2. Select robot from 'Control Robot' dropdown")
    log.info("3. Click 'Setup Joint Control' to create joint sliders")
    log.info("4. Use individual joint sliders to control robot pose")
    log.info("5. Click 'Reset Joints' to return to initial position")
    log.info("6. Click 'Clear Joint Control' to remove GUI panels")

    # import time
    # time.sleep(2)  # Wait for GUI to load

    example_config = {
        "panda_joint1": 0.0,
        "panda_joint2": -1.2,
        "panda_joint3": 0.0,
        "panda_joint4": -2.0,
        "panda_joint5": 0.0,
        "panda_joint6": 0.8,
        "panda_joint7": 0.0,
    }
    visualizer.update_robot_joint_config("franka", example_config)
    log.info("Applied example joint configuration")

    try:
        while True:
            pass

    except KeyboardInterrupt:
        log.info("Shutting down...")


if __name__ == "__main__":
    main()
