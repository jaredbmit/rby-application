""""""

import os
import h5py
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import scipy.spatial.transform as tf
from matplotlib.animation import FuncAnimation


import rby1_sdk as sdk

D2R = np.pi / 180
MINIMUM_TIME = 2.5
LINEAR_VELOCITY_LIMIT = 1.5
ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
ACCELERATION_LIMIT = 1.0
STOP_ORIENTATION_TRACKING_ERROR = 1e-5
STOP_POSITION_TRACKING_ERROR = 1e-5
# WEIGHT = 0.0015
WEIGHT = 0.01
CENTERING_WEIGHT = WEIGHT / 100
STOP_COST = 1e-2
VELOCITY_LIMIT_SCALE = 1.0
# VELOCITY_TRACKING_GAIN = 0.01
VELOCITY_TRACKING_GAIN = 0.1
MIN_DELTA_COST = 1e-4
# PATIENCE = 10
PATIENCE = 1
CONTROL_HOLD_TIME = 300

# Q_HOME = (
#     np.array(
#         [0, 30, -60, 30, 0, 0, -45, -30, 0, -90, 0, 45, 0, -45, 30, 0, -90, 0, 45, 0]
#     )
#     * D2R
# )
Q_HOME = (
    np.array(
        [0, 30, -60, 30, 0, 0, 15, -15, 0, -90, 0, 15, 0, 15, 15, 0, -90, 0, 15, 0]
    )
    * D2R
)


class Rainbow:
    def __init__(self):
        self.links = [
            "base",
            "link_torso_5",
            "ee_left",
            "ee_right",
        ]
        self.link_id_map = {
            "base": 0,
            "link_torso_5": 1,
            "ee_left": 4,
            "ee_right": 5,
        }

    def setup(self, address, power_device, servo_device):
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
        # Build command
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

    def command_hand_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        T_torso,
        T_right,
        side: str,
        controller: str = "cartesian",
    ):
        """Commands a pose for a hand wrt base frame"""
        if controller not in ["cartesian", "optimal"]:
            raise ValueError(
                f"`controller` must be 'cartesian' or 'optimal', not {controller}"
            )
        if side not in ["left", "right"]:
            raise ValueError(f"`side` must be 'left' or 'right', not {side}")

        T = np.eye(4)

        T[:3, 3] = position
        T[:3, :3] = rotation

        if side == "right":
            raise NotImplementedError()
            # if controller == "cartesian":
            #     rc = sdk.RobotCommandBuilder().set_command(
            #         sdk.ComponentBasedCommandBuilder().set_body_command(
            #             sdk.BodyComponentBasedCommandBuilder().set_right_arm_command(
            #                 sdk.CartesianCommandBuilder()
            #                 .add_target(
            #                     "base",
            #                     "ee_right",
            #                     T,
            #                     LINEAR_VELOCITY_LIMIT,
            #                     ANGULAR_VELOCITY_LIMIT,
            #                     ACCELERATION_LIMIT / 2,
            #                 )
            #                 .set_minimum_time(MINIMUM_TIME)
            #                 .set_stop_orientation_tracking_error(
            #                     STOP_ORIENTATION_TRACKING_ERROR
            #                 )
            #                 .set_stop_position_tracking_error(
            #                     STOP_POSITION_TRACKING_ERROR
            #                 )
            #             )
            #         )
            #     )
            # elif controller == "optimal":
            #     rc = sdk.RobotCommandBuilder().set_command(
            #         sdk.ComponentBasedCommandBuilder().set_body_command(
            #             sdk.OptimalControlCommandBuilder()
            #             .add_cartesian_target(
            #                 "base", "link_torso_5", T_torso, WEIGHT, WEIGHT
            #             )
            #             .add_cartesian_target("base", "ee_right", T_right, WEIGHT, WEIGHT)
            #             .add_cartesian_target("base", "ee_left", T, WEIGHT, WEIGHT)
            #             .add_joint_position_target("right_arm_2", np.pi / 2, WEIGHT)
            #             .add_joint_position_target("left_arm_2", -np.pi / 2, WEIGHT)
            #             .set_velocity_limit_scaling(0.05)
            #             .set_velocity_tracking_gain(VELOCITY_TRACKING_GAIN)
            #             .set_stop_cost(STOP_COST)
            #             .set_min_delta_cost(MIN_DELTA_COST)
            #             .set_patience(PATIENCE)
            #         )
            #     )
        elif side == "left":
            if controller == "cartesian":
                rc = sdk.RobotCommandBuilder().set_command(
                    sdk.ComponentBasedCommandBuilder().set_body_command(
                        sdk.BodyComponentBasedCommandBuilder().set_left_arm_command(
                            sdk.CartesianCommandBuilder()
                            .add_target(
                                "base",
                                "ee_left",
                                T,
                                LINEAR_VELOCITY_LIMIT,
                                ANGULAR_VELOCITY_LIMIT,
                                ACCELERATION_LIMIT / 2,
                            )
                            .set_minimum_time(MINIMUM_TIME)
                            .set_stop_orientation_tracking_error(
                                STOP_ORIENTATION_TRACKING_ERROR
                            )
                            .set_stop_position_tracking_error(
                                STOP_POSITION_TRACKING_ERROR
                            )
                        )
                    )
                )
            elif controller == "optimal":
                rc = sdk.RobotCommandBuilder().set_command(
                    sdk.ComponentBasedCommandBuilder().set_body_command(
                        sdk.OptimalControlCommandBuilder()
                        .add_cartesian_target(
                            "base", "link_torso_5", T_torso, WEIGHT, WEIGHT
                        )
                        .add_cartesian_target(
                            "base", "ee_right", T_right, WEIGHT, WEIGHT
                        )
                        .add_cartesian_target(
                            "base", "ee_left", T, WEIGHT * 10, WEIGHT * 10
                        )
                        .set_velocity_limit_scaling(0.05)
                        .set_velocity_tracking_gain(VELOCITY_TRACKING_GAIN)
                        .set_stop_cost(STOP_COST)
                        .set_min_delta_cost(MIN_DELTA_COST)
                        .set_patience(PATIENCE)
                    )
                )

        rv = self.robot.send_command(rc, 10).get()

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
            return 0
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return 1

        return 0

    def command_hand_trajectory(
        self,
        t: np.ndarray,
        position: np.ndarray,
        rotation: np.ndarray,
        T_torso,
        T_right,
        side: str,
        controller: str = "cartesian",
    ):
        """Commands a trajectory for a hand wrt base frame"""
        if controller not in ["cartesian", "optimal"]:
            raise ValueError(
                f"`controller` must be 'cartesian' or 'optimal', not {controller}"
            )
        if side not in ["left", "right"]:
            raise ValueError(f"`side` must be 'left' or 'right', not {side}")

        T = np.eye(4)

        duration = int(max(t)) + 1
        stream = self.robot.create_command_stream(10 * duration)

        for i in range(len(t) - 1):

            dt = float(t[i + 1] - t[i])
            T[:3, 3] = position[i]
            T[:3, :3] = rotation[i]

            if side == "right":
                raise NotImplementedError
                # if controller == "cartesian":
                #     rc = sdk.RobotCommandBuilder().set_command(
                #         sdk.ComponentBasedCommandBuilder().set_body_command(
                #             sdk.BodyComponentBasedCommandBuilder().set_right_arm_command(
                #                 sdk.CartesianCommandBuilder()
                #                 .set_command_header(
                #                     sdk.CommandHeaderBuilder().set_control_hold_time(10)
                #                 )
                #                 .add_target(
                #                     "base",
                #                     "ee_right",
                #                     T,
                #                     LINEAR_VELOCITY_LIMIT,
                #                     ANGULAR_VELOCITY_LIMIT,
                #                     ACCELERATION_LIMIT / 2,
                #                 )
                #                 .set_minimum_time(dt)
                #             )
                #         )
                #     )
                # elif controller == "optimal":
                #     rc = sdk.RobotCommandBuilder().set_command(
                #         sdk.ComponentBasedCommandBuilder().set_body_command(
                #             sdk.OptimalControlCommandBuilder()
                #             .set_command_header(
                #                 sdk.CommandHeaderBuilder().set_control_hold_time(10)
                #             )
                #             .add_cartesian_target("base", "ee_right", T, WEIGHT, WEIGHT)
                #             .set_velocity_limit_scaling(0.05)
                #             .set_velocity_tracking_gain(VELOCITY_TRACKING_GAIN)
                #             .set_stop_cost(STOP_COST)
                #             .set_min_delta_cost(MIN_DELTA_COST)
                #             .set_patience(PATIENCE)
                #         )
                #     )
            elif side == "left":
                if controller == "cartesian":
                    rc = sdk.RobotCommandBuilder().set_command(
                        sdk.ComponentBasedCommandBuilder().set_body_command(
                            sdk.BodyComponentBasedCommandBuilder().set_left_arm_command(
                                sdk.CartesianCommandBuilder()
                                .set_command_header(
                                    sdk.CommandHeaderBuilder().set_control_hold_time(
                                        CONTROL_HOLD_TIME
                                    )
                                )
                                .add_target(
                                    "base",
                                    "ee_left",
                                    T,
                                    LINEAR_VELOCITY_LIMIT,
                                    ANGULAR_VELOCITY_LIMIT,
                                    ACCELERATION_LIMIT / 2,
                                )
                                .set_minimum_time(dt)
                            )
                        )
                    )
                elif controller == "optimal":
                    rc = sdk.RobotCommandBuilder().set_command(
                        sdk.ComponentBasedCommandBuilder().set_body_command(
                            sdk.OptimalControlCommandBuilder()
                            .set_command_header(
                                sdk.CommandHeaderBuilder().set_control_hold_time(
                                    CONTROL_HOLD_TIME
                                )
                            )
                            .add_cartesian_target(
                                "base", "link_torso_5", T_torso, WEIGHT, WEIGHT
                            )
                            .add_cartesian_target("base", "ee_left", T, WEIGHT, WEIGHT)
                            .add_joint_position_target(
                                "left_arm_0", 15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_1", 15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_2", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_3", -90.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_4", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_5", 15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "left_arm_6", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_0", 15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_1", -15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_2", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_3", -90.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_4", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_5", 15.0 * D2R, CENTERING_WEIGHT
                            )
                            .add_joint_position_target(
                                "right_arm_6", 0.0 * D2R, CENTERING_WEIGHT
                            )
                            .set_velocity_limit_scaling(VELOCITY_LIMIT_SCALE)
                            .set_velocity_tracking_gain(VELOCITY_TRACKING_GAIN)
                            .set_stop_cost(STOP_COST)
                            .set_min_delta_cost(MIN_DELTA_COST)
                            .set_patience(PATIENCE)
                        )
                    )

            rv = stream.send_command(rc)

            time.sleep(dt)

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
            return 0
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return 1

        return 0


def follow_trajectory(address, power_device, servo):
    robot = Rainbow()
    robot.setup(address, power_device, servo)
    if not robot.reset_pose():
        print("Pose reset.")

    trajectory_file = os.path.expanduser(
        "~/drl/human/data/stirring/stirring_inference.hdf5"
    )
    with h5py.File(trajectory_file, "r") as f:
        trajectory = f["trajectory_000"]["data"]
        t = np.array(trajectory["time"])
        pos_human_to_left_hand_H = np.array(trajectory["pos_human_to_left_hand_H"])
        rot_human_to_left_hand = np.array(trajectory["rot_human_to_left_hand"]).reshape(
            (-1, 3, 3)
        )

    # Constants
    rot_robot_to_human = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    rot_left_hand_to_left_gripper = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    pos_left_hand_to_left_gripper_LG = np.zeros(3)

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
    pos_robot_to_human_R[0] *= 0.5
    pos_robot_to_human_R[1] *= 0.5

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
    )

    # # Scale trajectory
    # scale_factor = 2 / 3
    # mean = np.mean(pos_robot_to_left_gripper_R, axis=0)
    # pos_robot_to_left_gripper_R = (
    #     scale_factor * (pos_robot_to_left_gripper_R - mean) + mean
    # )

    # Re-interpolate
    dt = 0.01
    time_in = t
    time_out = np.arange(0, np.max(t), dt)
    interpolate_position = interpolate.interp1d(
        time_in, pos_robot_to_left_gripper_R, axis=0
    )
    pos_robot_to_left_gripper_R_interp = interpolate_position(time_out)
    interpolate_rotation = tf.Slerp(
        time_in, tf.Rotation.from_matrix(rot_robot_to_left_gripper)
    )
    rot_robot_to_left_gripper_interp = interpolate_rotation(time_out).as_matrix()

    # # Generate animation
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection="3d")
    # azim_min = 30
    # azim_max = 60

    # def update(frame):
    #     ax.clear()

    #     for i in range(3):
    #         color = ["red", "green", "blue"][i]
    #         ax.quiver(
    #             pos_robot_to_left_gripper_R[frame, 0],
    #             pos_robot_to_left_gripper_R[frame, 1],
    #             pos_robot_to_left_gripper_R[frame, 2],
    #             rot_robot_to_left_gripper[frame, 0, i],
    #             rot_robot_to_left_gripper[frame, 1, i],
    #             rot_robot_to_left_gripper[frame, 2, i],
    #             color=color,
    #             length=0.1,
    #         )

    #     # Timestamp & bounds
    #     ax.set_title(f"t = {t[frame]:.3f}s")
    #     ax.set_xlim([0.2, 0.7])
    #     ax.set_ylim([0.0, 0.5])
    #     ax.set_zlim([0.5, 1.0])
    #     ax.set_xlabel("X")
    #     ax.set_ylabel("Y")
    #     ax.set_zlabel("Z")
    #     angle = azim_min + (azim_max - azim_min) * frame / len(t)
    #     ax.view_init(elev=25, azim=angle)

    # interval_s = (t[-1] - t[0]) / (len(t) + 1)
    # ani = FuncAnimation(
    #     fig, update, frames=len(t), interval=1000 * interval_s, blit=False
    # )
    # ani.save("traj.gif")

    time.sleep(1)

    # if not robot.command_hand_pose(
    #     pos_robot_to_left_gripper_R_interp[0],
    #     rot_robot_to_left_gripper_interp[0],
    #     side="left",
    #     controller="cartesian",
    # ):
    #     print("Finished hand trajectory.")

    # time.sleep(1)

    # if not robot.reset_pose():
    #     print("Pose reset.")

    T_torso = robot.get_pose("base", "link_torso_5")
    T_right = robot.get_pose("base", "ee_right")

    # if not robot.command_hand_pose(
    #     pos_robot_to_left_gripper_R_interp[0],
    #     rot_robot_to_left_gripper_interp[0],
    #     T_torso,
    #     T_right,
    #     "left",
    #     controller="optimal",
    # ):
    #     print("Finished hand trajectory.")

    if not robot.command_hand_trajectory(
        time_out * 1,
        pos_robot_to_left_gripper_R_interp,
        rot_robot_to_left_gripper_interp,
        T_torso,
        T_right,
        side="left",
        controller="optimal",
    ):
        print("Finished hand trajectory.")

    # if not robot.command_hand_trajectory(
    #     t * 2,
    #     pos_robot_to_left_gripper_R,
    #     rot_robot_to_left_gripper,
    #     T_torso,
    #     T_right,
    #     side="left",
    #     controller="optimal",
    # ):
    #     print("Finished hand trajectory.")

    time.sleep(1)

    if not robot.reset_pose():
        print("Pose reset.")

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
