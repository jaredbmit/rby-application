""""""

import os
import h5py
import sys
import time
import argparse
from typing import Optional

import numpy as np
from scipy import interpolate
import scipy.spatial.transform as tf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import rby1_sdk as sdk

D2R = np.pi / 180
MINIMUM_TIME = 2.5
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-5
STOP_POSITION_TRACKING_ERROR = 1e-5

# Optimal Controller Presets
WEIGHT = 0.01
CENTERING_WEIGHT = 0.0001
BODY_CENTERING_WEIGHT = 0.001
STOP_COST = 1e-2
VELOCITY_LIMIT_SCALE = 1.0
VELOCITY_TRACKING_GAIN = 0.01
MIN_DELTA_COST = 1e-4
PATIENCE = 10
CONTROL_HOLD_TIME = 300

# Q_HOME = (
#     np.array(
#         [0, 30, -60, 30, 0, 0, -45, -30, 0, -90, 0, 45, 0, -45, 30, 0, -90, 0, 45, 0]
#     )
#     * D2R
# )

BEND_ANGLE = 5
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
            35,
            45,
            -15,
            15,
            10,
            -95,
            -30,
            35,
            -45,
        ]
    )
    * D2R
)


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

        Also, records true robot trajectories.

        Args:
            t (np.ndarray): (n,) Time vector.
            T_left (np.ndarray): (n, 4, 4) Pose sequence for the left end effector.
            T_right (np.ndarray): (n, 4, 4) Pose sequence for the right end effector.
            T_torso (np.ndarray): (n, 4, 4) Pose sequence for the torso.

        Returns:
            (int): 1 if failure, 0 if success or unknown.
            (np.ndarray): (n, 4, 4) True pose sequence for the left end effector.
            (np.ndarray): (n, 4, 4) True pose sequence for the right end effector.
            (np.ndarray): (n, 4, 4) True pose sequence for the torso end.
        """
        duration = int(max(t)) + 1
        stream = self.robot.create_command_stream(duration)

        T_true_left = []
        T_true_right = []
        T_true_torso = []

        for i in range(len(t) - 1):

            # T_true_left.append(self.get_pose("base", "ee_left"))
            # T_true_right.append(self.get_pose("base", "ee_right"))
            # T_true_torso.append(self.get_pose("base", "link_torso_5"))

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

            time.sleep(dt)

        # T_true_left.append(self.get_pose("base", "ee_left"))
        # T_true_right.append(self.get_pose("base", "ee_right"))
        # T_true_torso.append(self.get_pose("base", "link_torso_5"))

        # T_true_left = np.array(T_true_left)
        # T_true_right = np.array(T_true_right)
        # T_true_torso = np.array(T_true_torso)

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return 1, T_true_left, T_true_right, T_true_torso

        return 0, T_true_left, T_true_right, T_true_torso


def follow_trajectory(address, power_device, servo):
    robot = Rainbow(address, power_device, servo)

    if not robot.reset_pose():
        print("Pose reset.")

    # T_right_init = np.array([[0.45826324, -0.0415582, -0.88784442, 0.48676827], 
    #                          [0.88452268, 0.11939231, 0.4509602, -0.38339947], 
    #                          [0.08726071, -0.99197701, 0.09147228, 1.16882506], 
    #                          [0, 0, 0, 1]])

    # # Load example trajectory
    # trajectory_file = os.path.expanduser(
    #     "~/data/human/scooping_powder/scooping_powder_processed.hdf5"
    # )

    # with h5py.File(trajectory_file, "r") as f:
    #     trajectory = f["trajectory_000"]
    #     data = trajectory["data"]
    #     ref = trajectory["reference"]
    #     t = np.array(data["time"])
    #     pos_human_to_left_hand_H = np.array(data["pos_human_to_left_hand_H"])
    #     rot_human_to_left_hand = np.array(data["rot_human_to_left_hand"]).reshape(
    #         (-1, 3, 3)
    #     )
    #     pos_human_to_bowl_H = np.array(ref["pos_human_to_bowl_H"])
    #     pos_human_to_pitcher_H = np.array(ref["pos_human_to_pitcher_H"])
    #     theta_human_to_pitcher_H = np.array(ref["angle_human_to_pitcher"])

    # print(robot.get_pose("base", "ee_left"))

    # # Constants
    # rot_robot_to_human = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    # rot_right_hand_to_right_gripper = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    # rot_left_hand_to_left_gripper = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    # pos_right_hand_to_right_gripper_RG = np.array([-0.065, -0.075, 0.08])
    # pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.15])

    # # Rotation trajectory
    # rot_robot_to_left_gripper = (
    #     rot_robot_to_human @ rot_human_to_left_hand @ rot_left_hand_to_left_gripper
    # )

    # # Static position relations
    # pos_human_to_left_hand_init_R = rot_robot_to_human @ pos_human_to_left_hand_H[0]
    # pos_robot_to_left_gripper_init_R = robot.get_pose("base", "ee_left")[:3, 3]
    # pos_robot_to_human_R = (
    #     pos_robot_to_left_gripper_init_R
    #     - rot_robot_to_left_gripper[0] @ pos_left_hand_to_left_gripper_LG
    #     - pos_human_to_left_hand_init_R
    # )

    # # Position trajectory
    # pos_human_to_left_hand_R = (
    #     rot_robot_to_human @ pos_human_to_left_hand_H.reshape((-1, 3, 1))
    # ).reshape((-1, 3))
    # pos_left_hand_to_left_gripper_R = (
    #     rot_robot_to_left_gripper @ pos_left_hand_to_left_gripper_LG
    # )
    # pos_robot_to_left_gripper_R = (
    #     pos_robot_to_human_R
    #     + pos_human_to_left_hand_R
    #     + pos_left_hand_to_left_gripper_R
    # )

    # # Re-interpolate
    # dt = 0.01
    # time_in = t
    # time_out = np.arange(0, np.max(time_in), dt)
    # interpolate_position = interpolate.interp1d(
    #     time_in, pos_robot_to_left_gripper_R, axis=0
    # )
    # pos_robot_to_left_gripper_R_interp = interpolate_position(time_out)
    # interpolate_rotation = tf.Slerp(
    #     time_in, tf.Rotation.from_matrix(rot_robot_to_left_gripper)
    # )
    # rot_robot_to_left_gripper_interp = interpolate_rotation(time_out).as_matrix()

    # # Construct pose sequences
    # T_left = np.zeros((len(time_out), 4, 4))
    # T_left[:, :3, :3] = rot_robot_to_left_gripper_interp
    # T_left[:, :3, 3] = pos_robot_to_left_gripper_R_interp
    # T_left[:, 3, 3] = 1
    # T_torso = np.repeat(
    #     robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
    #     repeats=len(time_out),
    #     axis=0,
    # )

    # # Command starting pose
    # print("Navigating to start pose.")
    # result = robot.command_cartesian_pose(
    #     T_left=T_left[0],
    #     T_right=None,
    #     T_torso=T_torso[0],
    # )
    # if not result:
    #     print("Start pose reached.")
    # time.sleep(1)

    # print("Running trajectory.")
    # result, T_true_left, T_true_right, T_true_torso = robot.command_optimal_trajectory(
    #     time_out * 2,
    #     T_left=T_left,
    #     T_right=None,
    #     T_torso=T_torso,
    # )
    # if not result:
    #     print("Trajectory finished.")
    # time.sleep(1)

    # if not robot.reset_pose():
    #     print("Pose reset.")

    # print("Done.")

    trajectory_file = os.path.expanduser(
        "~/data/human/pouring/pouring_processed.hdf5"
    )

    # Traj videos: 0, 36, 42

    with h5py.File(trajectory_file, "r") as f:
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
    pos_right_hand_to_right_gripper_RG = np.array([-0.065, -0.075, 0.08])
    rot_left_hand_to_left_gripper = np.eye(3)  # TODO
    pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.15])

    # Right gripper rotation trajectory
    rot_robot_to_right_gripper = (
        rot_robot_to_human @ rot_human_to_right_hand @ rot_right_hand_to_right_gripper
    )

    # Left gripper rotation trajectory
    # rot_robot_to_left_gripper_init = robot.get_pose("base", "ee_left")[:3, :3]
    # rot_robot_to_left_gripper = np.repeat(
    #     rot_robot_to_left_gripper_init[np.newaxis, ...],
    #     repeats=len(t),
    #     axis=0,
    # )
    theta = np.pi / 12
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
    pos_robot_to_right_gripper_init_R = T_right_init[:3, 3]
    pos_robot_to_human_R = (
        pos_robot_to_right_gripper_init_R
        - pos_human_to_right_hand_init_R
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
        + np.array([0, 0.1, -0.225])
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
    dt = 0.01
    time_out = np.arange(0, np.max(t), dt)
    T_right_interp = interpolate_trajectory(t, time_out, T_right)
    T_left_interp = interpolate_trajectory(t, time_out, T_left)

    # Torso static trajectory
    T_torso = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_out),
        axis=0,
    )

    # Command starting pose
    print("Navigating to start pose.")
    time_mini = np.arange(0, 1, dt)
    T_right_init = robot.get_pose("base", "ee_right")
    T_right_mini = np.stack([T_right_init, T_right[0]])
    T_right_mini_interp = interpolate_trajectory([0, 1], time_mini, T_right_mini)
    T_left_init = robot.get_pose("base", "ee_left")
    T_left_mini = np.stack([T_left_init, T_left[0]])
    T_left_mini_interp = interpolate_trajectory([0, 1], time_mini, T_left_mini)
    T_torso_mini = np.repeat(
        robot.get_pose("base", "link_torso_5")[np.newaxis, ...],
        repeats=len(time_mini),
        axis=0,
    )
    result, _, _, _ = robot.command_optimal_trajectory(
        time_mini * 3,
        T_left=T_left_mini_interp,
        T_right=T_right_mini_interp,
        T_torso=T_torso_mini,
    )
    if not result:
        print("Start pose reached.")
    time.sleep(2)

    print("Running trajectory.")
    result, T_true_left, T_true_right, T_true_torso = robot.command_optimal_trajectory(
        time_out * 1.8,
        T_left=T_left_interp,
        T_right=T_right_interp,
        T_torso=T_torso,
    )
    if not result:
        print("Trajectory finished.")
    time.sleep(1)

    # if not robot.reset_pose():
    #     print("Pose reset.")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", type=str, default="localhost:50051", help="Robot address"
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

    follow_trajectory(args.address, args.device, args.servo)
