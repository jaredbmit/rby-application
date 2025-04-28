""""""

import os
import h5py
import sys
import time
import argparse
import queue
from typing import Optional

import numpy as np
from scipy import interpolate
import scipy.spatial.transform as tf
import matplotlib.pyplot as plt

import rby1_sdk as sdk

D2R = np.pi / 180
MINIMUM_TIME = 2.5
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-5
STOP_POSITION_TRACKING_ERROR = 1e-5

# Optimal Controller Presets
WEIGHT = 0.02  # main parameter - higher will try to track better
CENTERING_WEIGHT = 0.0001
BODY_CENTERING_WEIGHT = 0.001
STOP_COST = 1e-2
VELOCITY_LIMIT_SCALE = 1.0
VELOCITY_TRACKING_GAIN = 0.5
MIN_DELTA_COST = 1e-4
PATIENCE = 10
CONTROL_HOLD_TIME = 300

BEND_ANGLE = 10
# Pouring
Q_HOME = (
    np.array(
        [
            0,
            BEND_ANGLE,
            -2 * BEND_ANGLE,
            BEND_ANGLE,
            0,
            0,
            -15,
            -15,
            -10,
            -95,
            30,
            25,
            45,
            -15,
            15,
            10,
            -95,
            -30,
            25,
            -45,
        ]
    )
    * D2R
)

# # Scooping
# Q_HOME = (
#     np.array(
#         [
#             0,
#             BEND_ANGLE,
#             -2 * BEND_ANGLE,
#             BEND_ANGLE,
#             0,
#             0,
#             0,
#             -15,
#             -10,
#             -95,
#             -60,
#             0,
#             135,
#             0,
#             15,
#             10,
#             -95,
#             60,
#             0,
#             -135,
#         ]
#     )
#     * D2R
# )


def interpolate_trajectory(time_in, time_out, pose):
    interpolate_position = interpolate.interp1d(time_in, pose[:, :3, 3], axis=0)
    position = interpolate_position(time_out)
    interpolate_rotation = tf.Slerp(time_in, tf.Rotation.from_matrix(pose[:, :3, :3]))
    rotation = interpolate_rotation(time_out).as_matrix()

    T = np.zeros((len(time_out), 4, 4))
    T[:, :3, :3] = rotation
    T[:, :3, 3] = position
    T[:, 3, 3] = 1

    return T


class Rainbow:
    def __init__(self, address, power_device, servo):
        self.links = [
            "base",
            "link_torso_5",
            "link_left_arm_1",
            "link_right_arm_1",
            "ee_left",
            "ee_right",
        ]
        self.link_id_map = {
            "base": 0,
            "link_torso_5": 1,
            "link_left_arm_1": 2,
            "link_right_arm_1": 3,
            "ee_left": 4,
            "ee_right": 5,
        }

        self._setup(address, power_device, servo)

    def _setup(self, address, power_device, servo_device):
        print("Attempting to connect to the robot...")

        self.robot = sdk.create_robot_a(address)

        if not self.robot.connect():
            print("Error: Unable to establish connection to the robot.")
            sys.exit(1)

        if not self.robot.is_connected():
            print("Robot is not connected.")
            exit(1)

        # Turn on the robot power
        if not self.robot.is_power_on(power_device):
            rv = self.robot.power_on(power_device)
            if not rv:
                print("Failed to power on.")
                exit(1)

        # Turn on the robot servos
        if not self.robot.is_servo_on(servo_device):
            rv = self.robot.servo_on(servo_device)
            if not rv:
                print("Failed to turn on servos.")
                exit(1)

        # Check and reset control manager faults if necessary
        control_manager_state = self.robot.get_control_manager_state()
        if (
            control_manager_state.state == sdk.ControlManagerState.State.MinorFault
            or control_manager_state.state == sdk.ControlManagerState.State.MajorFault
        ):
            print("Attempting to reset the fault...")
            if not self.robot.reset_fault_control_manager():
                print("Error: Unable to reset the fault in the Control Manager.")
                sys.exit(1)
            print("Fault reset successfully.")

        # Enable the control manager
        if not self.robot.enable_control_manager():
            print("Error: Failed to enable the Control Manager.")
            sys.exit(1)

        self.robot.set_parameter("default.acceleration_limit_scaling", "0.8")
        self.robot.set_parameter("joint_position_command.cutoff_frequency", "5")
        self.robot.set_parameter("cartesian_command.cutoff_frequency", "5")
        self.robot.set_parameter("default.linear_acceleration_limit", "5")
        self.robot.set_time_scale(1.0)

        self.dynamics = self.robot.get_dynamics()
        self.state = self.dynamics.make_state(
            self.links,
            sdk.Model_A().robot_joint_names,
        )

    def reset_pose(self):
        rc = sdk.RobotCommandBuilder().set_command(
            sdk.ComponentBasedCommandBuilder().set_body_command(
                sdk.BodyCommandBuilder().set_command(
                    sdk.JointPositionCommandBuilder()
                    .set_position(Q_HOME)
                    .set_minimum_time(MINIMUM_TIME)
                )
            )
        )

        rv = self.robot.send_command(rc, 10).get()

        if rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to reset pose.")
            return 1

        return 0

    # 3x3 rotation matrix in upper left
    # rightmost column is x y z 1 from top to bottom (homogeneous coordinates)
    def get_pose(self, link_1: str, link_2: str):
        """
        Calculates pose of link_2 wrt link_1 (per self.links defined in init)
        """
        position = self.robot.get_state().position
        self.state.set_q(position)
        self.dynamics.compute_forward_kinematics(self.state)
        link_id_1 = self.link_id_map[link_1]
        link_id_2 = self.link_id_map[link_2]
        pose = self.dynamics.compute_transformation(self.state, link_id_1, link_id_2)
        return pose

    def run_trajectory_and_record(
        self,
        t: np.ndarray,
        T_left: Optional[np.ndarray] = None,
        T_right: Optional[np.ndarray] = None,
        T_torso: Optional[np.ndarray] = None,
    ):
        state_queue = queue.Queue()

        def callback(robot_state, control_manager_state):
            timestamp = time.perf_counter()
            state_queue.put((timestamp, robot_state))

        self.robot.start_state_update(callback, rate=10)

        timestamps_cmd = self.command_optimal_trajectory(
            t, T_left=T_left, T_right=T_right, T_torso=T_torso
        )

        self.robot.stop_state_update()

        # Extract history from queue
        print("Computing state history...")
        timestamps_meas = []
        states = []
        while not state_queue.empty():
            t, state = state_queue.get()
            timestamps_meas.append(t)
            states.append(state)

        T_left_true = []
        T_right_true = []
        T_torso_true = []
        for state in states:
            self.state.set_q(state.position)
            self.dynamics.compute_forward_kinematics(self.state)
            base_link_id = self.link_id_map["base"]
            left_link_id = self.link_id_map["ee_left"]
            right_link_id = self.link_id_map["ee_right"]
            torso_link_id = self.link_id_map["link_torso_5"]
            T_left_true.append(
                self.dynamics.compute_transformation(
                    self.state, base_link_id, left_link_id
                )
            )
            T_right_true.append(
                self.dynamics.compute_transformation(
                    self.state, base_link_id, right_link_id
                )
            )
            T_torso_true.append(
                self.dynamics.compute_transformation(
                    self.state, base_link_id, torso_link_id
                )
            )

        T_left_true = np.array(T_left_true)
        T_right_true = np.array(T_right_true)
        T_torso_true = np.array(T_torso_true)
        print("Finished")

        return timestamps_cmd, timestamps_meas, T_left_true, T_right_true, T_torso_true

    def command_optimal_trajectory(
        self,
        t: np.ndarray,
        T_left: Optional[np.ndarray] = None,
        T_right: Optional[np.ndarray] = None,
        T_torso: Optional[np.ndarray] = None,
    ):
        """
        Commands a trajectory for the 3 main body components
        using RB-Y1's optimal controller.

        Note: Assumes the robot is already at the initial pose!

        Args:
            t (np.ndarray): (n,) Time vector.
            T_left (np.ndarray): (n, 4, 4) Pose sequence for the left end effector.
            T_right (np.ndarray): (n, 4, 4) Pose sequence for the right end effector.
            T_torso (np.ndarray): (n, 4, 4) Pose sequence for the torso.

        Returns:
            (int): 1 if failure, 0 if success or unknown.
        """
        duration = int(max(t)) + 1
        stream = self.robot.create_command_stream(duration)
        timestamps = []

        for i in range(len(t) - 1):
            # print(i, t[i+1], T_right[i + 1])
            dt = float(t[i + 1] - t[i])

            # Using optimal controller
            optimal_command = (
                sdk.OptimalControlCommandBuilder()
                .set_command_header(
                    sdk.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME)
                )
                .add_joint_position_target("torso_0", Q_HOME[0], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("torso_1", Q_HOME[1], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("torso_2", Q_HOME[2], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("torso_3", Q_HOME[3], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("torso_4", Q_HOME[4], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("torso_5", Q_HOME[5], BODY_CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_0", Q_HOME[6], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_1", Q_HOME[7], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_2", Q_HOME[8], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_3", Q_HOME[9], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_4", Q_HOME[10], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_5", Q_HOME[11], CENTERING_WEIGHT)
                .add_joint_position_target("right_arm_6", Q_HOME[12], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_0", Q_HOME[13], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_1", Q_HOME[14], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_2", Q_HOME[15], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_3", Q_HOME[16], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_4", Q_HOME[17], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_5", Q_HOME[18], CENTERING_WEIGHT)
                .add_joint_position_target("left_arm_6", Q_HOME[19], CENTERING_WEIGHT)
                .set_velocity_limit_scaling(VELOCITY_LIMIT_SCALE)
                .set_velocity_tracking_gain(VELOCITY_TRACKING_GAIN)
                .set_stop_cost(STOP_COST)
                .set_min_delta_cost(MIN_DELTA_COST)
                .set_patience(PATIENCE)
            )

            if T_left is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "ee_left", T_left[i + 1], WEIGHT, WEIGHT
                )

            if T_right is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "ee_right", T_right[i + 1], WEIGHT, WEIGHT
                )

            if T_torso is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "link_torso_5", T_torso[i + 1], WEIGHT, WEIGHT
                )

            rc = sdk.RobotCommandBuilder().set_command(
                sdk.ComponentBasedCommandBuilder().set_body_command(optimal_command)
            )

            rv = stream.send_command(rc)

            timestamps.append(time.perf_counter())
            time.sleep(dt)

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return timestamps

        return timestamps


def follow_scooping_trajectory(address, power_device, servo):
    robot = Rainbow(address, power_device, servo)

    if not robot.reset_pose():
        print("Pose reset.")

    # Load example trajectory
    trajectory_file = os.path.expanduser(
        "~/drl/human/data/2025-04-06/scooping_powder/scooping_powder_processed.hdf5"
    )

    with h5py.File(trajectory_file, "r") as f:
        trajectory = f["trajectory_000"]
        data = trajectory["data"]
        t = np.array(data["time"])
        pos_human_to_left_hand_H = np.array(data["pos_human_to_left_hand_H"])
        rot_human_to_left_hand = np.array(data["rot_human_to_left_hand"]).reshape(
            (-1, 3, 3)
        )
        # ref = trajectory["reference"]
        # pos_human_to_bowl_H = np.array(ref["pos_human_to_bowl_H"])
        # pos_human_to_pitcher_H = np.array(ref["pos_human_to_pitcher_H"])
        # theta_human_to_pitcher_H = np.array(ref["angle_human_to_pitcher"])

    # Constants
    rot_robot_to_human = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # rot_right_hand_to_right_gripper = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])

    # Robot holds spoon straight on
    _spoon_tilt_angles_1 = np.array([np.pi / 2, 0, np.pi / 5])
    rot_left_hand_to_spoon = tf.Rotation.from_euler(
        "xyz", _spoon_tilt_angles_1
    ).as_matrix()
    rot_spoon_to_left_gripper = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    rot_left_hand_to_left_gripper = rot_left_hand_to_spoon @ rot_spoon_to_left_gripper

    # pos_right_hand_to_right_gripper_RG = np.array([-0.065, -0.075, 0.08])
    pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0])

    # Rotation trajectory
    rot_robot_to_left_gripper = (
        rot_robot_to_human @ rot_human_to_left_hand @ rot_left_hand_to_left_gripper
    )

    # Static position relations
    pos_human_to_left_hand_init_R = rot_robot_to_human @ pos_human_to_left_hand_H[0]
    pos_robot_to_left_gripper_init_R = robot.get_pose("base", "ee_left")[:3, 3]
    pos_robot_to_human_R = (
        pos_robot_to_left_gripper_init_R
        - rot_robot_to_left_gripper[0] @ pos_left_hand_to_left_gripper_LG
        - pos_human_to_left_hand_init_R
    )

    # Position trajectory
    pos_human_to_left_hand_R = (
        rot_robot_to_human @ pos_human_to_left_hand_H.reshape((-1, 3, 1))
    ).reshape((-1, 3))
    pos_left_hand_to_left_gripper_R = (
        rot_robot_to_left_gripper @ pos_left_hand_to_left_gripper_LG
    )
    pos_robot_to_left_gripper_R = (
        pos_robot_to_human_R
        + pos_human_to_left_hand_R
        + pos_left_hand_to_left_gripper_R
        + np.array([-0.1, 0, 0])
    )

    # Form pose trajectories
    T_left = np.zeros((len(t), 4, 4))
    T_left[:, :3, :3] = rot_robot_to_left_gripper
    T_left[:, :3, 3] = pos_robot_to_left_gripper_R
    T_left[:, 3, 3] = 1

    # Re-interpolate
    factor = 5.0
    dt = 0.01
    time_in = factor * t
    time_out = np.arange(0, np.max(time_in), dt)
    T_left_interp = interpolate_trajectory(time_in, time_out, T_left)

    # Torso static trajectory
    T_torso = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_out),
        axis=0,
    )

    # Command starting pose
    print("Navigating to start pose.")
    duration_mini = 3
    time_mini = np.arange(0, duration_mini, dt)
    T_left_init = robot.get_pose("base", "ee_left")
    T_left_mini = np.stack([T_left_init, T_left[0]])
    T_left_mini_interp = interpolate_trajectory(
        [0, duration_mini], time_mini, T_left_mini
    )
    T_torso_mini = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_mini),
        axis=0,
    )
    robot.command_optimal_trajectory(
        time_mini,
        T_left=T_left_mini_interp,
        T_right=None,
        T_torso=T_torso_mini,
    )
    print("Starting pose reached.")
    time.sleep(2)

    print("Running trajectory.")
    timestamps_cmd, timestamps_true, T_left_true, T_right_true, T_torso_true = (
        robot.run_trajectory_and_record(
            time_out,
            T_left=T_left_interp,
            T_right=None,
            T_torso=T_torso,
        )
    )
    fig, ax = plt.subplots(3)
    time_cmd = [t - timestamps_cmd[0] for t in timestamps_cmd]
    time_true = [t - timestamps_true[0] for t in timestamps_true]
    ax[0].plot(time_cmd, T_left_interp[1:, 0, 3], "--", label="Command")
    ax[0].plot(time_true, T_left_true[:, 0, 3], "-", label="Measured")
    ax[0].set_ylabel("X (m)")
    ax[0].legend()
    ax[1].plot(time_cmd, T_left_interp[1:, 1, 3], "--", label="Command")
    ax[1].plot(time_true, T_left_true[:, 1, 3], "-", label="Measured")
    ax[1].set_ylabel("Y (m)")
    ax[2].plot(time_cmd, T_left_interp[1:, 2, 3], "--", label="Command")
    ax[2].plot(time_true, T_left_true[:, 2, 3], "-", label="Measured")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("Left EE Position Tracking")
    fig.savefig("left_ee_position_tracking.png")
    fig, ax = plt.subplots(3)
    ax[0].plot(time_cmd, T_torso[1:, 0, 3], "--", label="Command")
    ax[0].plot(time_true, T_torso_true[:, 0, 3], "-", label="Measured")
    ax[0].set_ylabel("X (m)")
    ax[0].legend()
    ax[1].plot(time_cmd, T_torso[1:, 1, 3], "--", label="Command")
    ax[1].plot(time_true, T_torso_true[:, 1, 3], "-", label="Measured")
    ax[1].set_ylabel("Y (m)")
    ax[2].plot(time_cmd, T_torso[1:, 2, 3], "--", label="Command")
    ax[2].plot(time_true, T_torso_true[:, 2, 3], "-", label="Measured")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("Torso Position Tracking")
    fig.savefig("torso_position_tracking.png")
    print("Trajectory finished.")
    time.sleep(1)
    time.sleep(1)

    if not robot.reset_pose():
        print("Pose reset.")

    print("Done.")


def follow_pouring_trajectory(address, power_device, servo):
    robot = Rainbow(address, power_device, servo)

    if not robot.reset_pose():
        print("Pose reset.")

    trajectory_file = os.path.expanduser(
        "~/drl/human/data/2025-04-06/pouring/pouring_processed.hdf5"
    )

    with h5py.File(trajectory_file, "r") as f:
        # Traj videos: 0, 36, 42
        trajectory = f["trajectory_042"]
        data = trajectory["data"]
        ref = trajectory["reference"]
        t = np.array(data["time"])
        pos_human_to_right_hand_H = np.array(data["pos_human_to_right_hand_H"])
        rot_human_to_right_hand = np.array(data["rot_human_to_right_hand"]).reshape(
            (-1, 3, 3)
        )
        pos_human_to_glass_rim_H = np.array(ref["pos_human_to_glass_rim_H"])

    # Constants
    rot_robot_to_human = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_right_hand_to_right_gripper = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    # pos_right_hand_to_right_gripper_RG = np.array([-0.065, -0.075, 0.08])
    pos_right_hand_to_right_gripper_RG = np.array([0, 0, 0.125])
    rot_left_hand_to_left_gripper = np.eye(3)  # TODO
    # pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.15])
    pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.175])

    # Right gripper rotation trajectory
    rot_robot_to_right_gripper = (
        rot_robot_to_human @ rot_human_to_right_hand @ rot_right_hand_to_right_gripper
    )

    # Left gripper rotation trajectory
    theta = np.pi / 6
    rot_robot_to_left_gripper_base = np.array(
        [
            [-np.sin(theta), 0, -np.cos(theta)],
            [-np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
        ]
    )
    rot_robot_to_left_gripper = np.repeat(
        rot_robot_to_left_gripper_base[np.newaxis, ...],
        repeats=len(t),
        axis=0,
    )

    # Static position relations
    pos_human_to_right_hand_init_R = rot_robot_to_human @ pos_human_to_right_hand_H[0]
    pos_robot_to_right_gripper_init_R = robot.get_pose("base", "ee_right")[:3, 3]
    pos_robot_to_human_R = (
        pos_robot_to_right_gripper_init_R - pos_human_to_right_hand_init_R
    )

    # Right gripper position trajectory
    pos_human_to_right_hand_R = (
        rot_robot_to_human @ pos_human_to_right_hand_H.reshape((-1, 3, 1))
    ).reshape((-1, 3))
    pos_right_hand_to_right_gripper_R = (
        rot_robot_to_right_gripper @ pos_right_hand_to_right_gripper_RG
    )
    pos_robot_to_right_gripper_R = (
        pos_robot_to_human_R
        + pos_human_to_right_hand_R
        + pos_right_hand_to_right_gripper_R
        + np.array([0, 0.1, 0])
    )

    # Left gripper position trajectory
    pos_human_to_left_hand_H = np.repeat(
        pos_human_to_glass_rim_H[np.newaxis, ...],
        repeats=len(t),
        axis=0,
    )
    pos_human_to_left_hand_R = (
        rot_robot_to_human @ pos_human_to_left_hand_H.reshape((-1, 3, 1))
    ).reshape((-1, 3))
    pos_left_hand_to_left_gripper_R = (
        rot_robot_to_left_gripper @ pos_left_hand_to_left_gripper_LG
    )
    pos_robot_to_left_gripper_R = (
        pos_robot_to_human_R
        + pos_human_to_left_hand_R
        + pos_left_hand_to_left_gripper_R
        + np.array([0, 0.1, 0])
        # + np.array([0, 0, -0.225])
    )

    # Form pose trajectories
    T_right = np.zeros((len(t), 4, 4))
    T_right[:, :3, :3] = rot_robot_to_right_gripper
    T_right[:, :3, 3] = pos_robot_to_right_gripper_R
    T_right[:, 3, 3] = 1
    T_left = np.zeros((len(t), 4, 4))
    T_left[:, :3, :3] = rot_robot_to_left_gripper
    T_left[:, :3, 3] = pos_robot_to_left_gripper_R
    T_left[:, 3, 3] = 1

    # Re-interpolate
    factor = 5.0
    dt = 0.01
    time_in = factor * t
    time_out = np.arange(0, np.max(time_in), dt)
    T_right_interp = interpolate_trajectory(time_in, time_out, T_right)
    T_left_interp = interpolate_trajectory(time_in, time_out, T_left)

    # Torso static trajectory
    T_torso = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_out),
        axis=0,
    )

    # Command starting pose
    print("Navigating to start pose.")
    duration_mini = 3
    time_mini = np.arange(0, duration_mini, dt)
    T_right_init = robot.get_pose("base", "ee_right")
    T_right_mini = np.stack([T_right_init, T_right[0]])
    T_right_mini_interp = interpolate_trajectory(
        [0, duration_mini], time_mini, T_right_mini
    )
    T_left_init = robot.get_pose("base", "ee_left")
    T_left_mini = np.stack([T_left_init, T_left[0]])
    T_left_mini_interp = interpolate_trajectory(
        [0, duration_mini], time_mini, T_left_mini
    )
    T_torso_mini = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_mini),
        axis=0,
    )
    result = robot.command_optimal_trajectory(
        time_mini,
        T_left=T_left_mini_interp,
        T_right=T_right_mini_interp,
        T_torso=T_torso_mini,
    )
    if not result:
        print("Start pose reached.")
    time.sleep(2)

    print("Running trajectory.")
    timestamps_cmd, timestamps_true, T_left_true, T_right_true, T_torso_true = (
        robot.run_trajectory_and_record(
            time_out,
            T_left=T_left_interp,
            T_right=T_right_interp,
            T_torso=T_torso,
        )
    )
    fig, ax = plt.subplots(3)
    time_cmd = [t - timestamps_cmd[0] for t in timestamps_cmd]
    time_true = [t - timestamps_true[0] for t in timestamps_true]
    ax[0].plot(time_cmd, T_left_interp[1:, 0, 3], "--", label="Command")
    ax[0].plot(time_true, T_left_true[:, 0, 3], "-", label="Measured")
    ax[0].set_ylabel("X (m)")
    ax[0].legend()
    ax[1].plot(time_cmd, T_left_interp[1:, 1, 3], "--", label="Command")
    ax[1].plot(time_true, T_left_true[:, 1, 3], "-", label="Measured")
    ax[1].set_ylabel("Y (m)")
    ax[2].plot(time_cmd, T_left_interp[1:, 2, 3], "--", label="Command")
    ax[2].plot(time_true, T_left_true[:, 2, 3], "-", label="Measured")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("Left EE Position Tracking")
    fig.savefig("left_ee_position_tracking.png")
    fig, ax = plt.subplots(3)
    ax[0].plot(time_cmd, T_right_interp[1:, 0, 3], "--", label="Command")
    ax[0].plot(time_true, T_right_true[:, 0, 3], "-", label="Measured")
    ax[0].set_ylabel("X (m)")
    ax[0].legend()
    ax[1].plot(time_cmd, T_right_interp[1:, 1, 3], "--", label="Command")
    ax[1].plot(time_true, T_right_true[:, 1, 3], "-", label="Measured")
    ax[1].set_ylabel("Y (m)")
    ax[2].plot(time_cmd, T_right_interp[1:, 2, 3], "--", label="Command")
    ax[2].plot(time_true, T_right_true[:, 2, 3], "-", label="Measured")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("Right EE Position Tracking")
    fig.savefig("right_ee_position_tracking.png")
    fig, ax = plt.subplots(3)
    ax[0].plot(time_cmd, T_torso[1:, 0, 3], "--", label="Command")
    ax[0].plot(time_true, T_torso_true[:, 0, 3], "-", label="Measured")
    ax[0].set_ylabel("X (m)")
    ax[0].legend()
    ax[1].plot(time_cmd, T_torso[1:, 1, 3], "--", label="Command")
    ax[1].plot(time_true, T_torso_true[:, 1, 3], "-", label="Measured")
    ax[1].set_ylabel("Y (m)")
    ax[2].plot(time_cmd, T_torso[1:, 2, 3], "--", label="Command")
    ax[2].plot(time_true, T_torso_true[:, 2, 3], "-", label="Measured")
    ax[2].set_ylabel("Z (m)")
    ax[2].set_xlabel("Time (s)")
    fig.suptitle("Torso Position Tracking")
    fig.savefig("torso_position_tracking.png")
    print("Trajectory finished.")
    time.sleep(1)

    if not robot.reset_pose():
        print("Pose reset.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default="192.168.12.1:50051",  # localhost:50051 for simulation
        help="Robot address",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=".*",
        help="Power device name regex pattern (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default=".*",
        help="Servo name regex pattern (default: '.*')",
    )
    args = parser.parse_args()

    follow_pouring_trajectory(args.address, args.device, args.servo)
    # follow_scooping_trajectory(args.address, args.device, args.servo)
    # follow_stirring_trajectory(args.address, args.device, args.servo)
