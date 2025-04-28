import os
import h5py
import numpy as np
import time
import matplotlib.pyplot as plt

from RainbowInterface import RainbowInterface

D2R = np.pi / 180


class ExperimentInterface:
    def __init__(self):
        # Initialize the robot.
        robot_ip_address = "192.168.12.1:50051"
        power_device_regex_str = ".*"
        servo_name_regex_str = ".*"
        self._rainbow_interface = RainbowInterface(
            robot_ip_address, power_device_regex_str, servo_name_regex_str
        )
        # Initialize state for the loaded trajectory.
        self._t = None
        self._T_left = None
        self._T_right = None
        self._T_torso = None
        self._speed_reduction_factor = 5
        self._right_gripper_offset_m = np.array([0, 0, 0], dtype=float)
        self._left_gripper_offset_m = np.array([0, 0, 0], dtype=float)
        self._previous_trajectory_activity = None
        self._previous_load_trajectory_command = None

        self._controller_type = "cartesian"
        # self._controller_type = 'optimal'

    def load_pouring_trajectory(self, trajectory_index):

        print("Loading pouring trajectory %d" % (trajectory_index))

        # trajectory_file = r"C:\Users\jdelp\Desktop\ActionSense\code\robot_control\rby-application\data\pouring\pouring_processed.hdf5"
        trajectory_file = os.path.expanduser(
            "~/drl/human/data/2025-04-06/pouring/pouring_processed.hdf5"
        )

        # Define the home positions.
        BEND_ANGLE = 10
        self._rainbow_interface.set_home_position(
            (
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
        )
        self._rainbow_interface.set_home_pose(
            "left",
            [
                [
                    0.34486492049937717,
                    -0.07384813809901812,
                    -0.9357428274415284,
                    0.4940080743827816,
                ],
                [
                    -0.9342627602581264,
                    0.06929264092571862,
                    -0.34978797107448634,
                    0.3677037157440348,
                ],
                [
                    0.09067128213400649,
                    0.9948592776935072,
                    -0.045096964236272234,
                    1.1828204989965654,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        self._rainbow_interface.set_home_pose(
            "right",
            [
                [
                    0.344859060491093,
                    0.07384720848743002,
                    -0.9357450604709672,
                    0.4940085795853877,
                ],
                [
                    0.934264388925526,
                    0.06929915270621888,
                    0.3497823309142603,
                    -0.36770178404107295,
                ],
                [
                    0.09067678855591227,
                    -0.9948588931264642,
                    -0.04509437641626934,
                    1.1828198617983126,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )
        self._rainbow_interface.set_home_pose(
            "torso",
            [
                [0.9999999999999999, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.9999999999999999, 1.2792920781331494],
                [0.0, 0.0, 0.0, 1.0],
            ],
        )

        # Define the human to robot transformations.
        rot_robot_to_human = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_right_hand_to_right_gripper = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        # pos_right_hand_to_right_gripper_RG = np.array([-0.065, -0.075, 0.08])
        pos_right_hand_to_right_gripper_RG = np.array([0, 0, 0.125])
        rot_left_hand_to_left_gripper = np.eye(3)  # TODO
        # pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.15])
        pos_left_hand_to_left_gripper_LG = np.array([0, 0, 0.175])
        left_gripper_theta = np.pi / 6
        if self._previous_trajectory_activity != "pour":
            self._right_gripper_offset_m = np.array([0, 0, 0], dtype=float)
            self._left_gripper_offset_m = np.array(
                [0, 0.1, 0], dtype=float
            )  # np.array([0, 0, -0.225], dtype=float)
        self._previous_trajectory_activity = "pour"

        # Load the trajectory data.
        with h5py.File(trajectory_file, "r") as f:
            # Traj videos: 0, 36, 42
            trajectory = f["trajectory_%03d" % trajectory_index]
            data = trajectory["data"]
            ref = trajectory["reference"]
            t = np.array(data["time"])
            pos_human_to_right_hand_H = np.array(data["pos_human_to_right_hand_H"])
            rot_human_to_right_hand = np.array(data["rot_human_to_right_hand"]).reshape(
                (-1, 3, 3)
            )
            pos_human_to_glass_rim_H = np.array(ref["pos_human_to_glass_rim_H"])

        # Right gripper rotation trajectory
        rot_robot_to_right_gripper = (
            rot_robot_to_human
            @ rot_human_to_right_hand
            @ rot_right_hand_to_right_gripper
        )

        # Left gripper rotation trajectory
        rot_robot_to_left_gripper_base = np.array(
            [
                [-np.sin(left_gripper_theta), 0, -np.cos(left_gripper_theta)],
                [-np.cos(left_gripper_theta), 0, np.sin(left_gripper_theta)],
                [0, 1, 0],
            ]
        )
        rot_robot_to_left_gripper = np.repeat(
            rot_robot_to_left_gripper_base[np.newaxis, ...],
            repeats=len(t),
            axis=0,
        )

        # Static position relations
        pos_human_to_right_hand_init_R = (
            rot_robot_to_human @ pos_human_to_right_hand_H[0]
        )
        pos_robot_to_right_gripper_init_R = self._rainbow_interface.get_home_pose(
            "right"
        )[:3, 3]
        # pos_robot_to_right_gripper_init_R = self._rainbow_interface.get_pose('base', "ee_right")[:3, 3]
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
            + self._right_gripper_offset_m
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
            + self._left_gripper_offset_m
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

        # Torso static trajectory
        T_torso = np.repeat(
            self._rainbow_interface.get_home_pose("torso")[np.newaxis, ...],
            # self._rainbow_interface.get_pose('base', 'link_torso_5')[np.newaxis, ...],
            repeats=len(t),
            axis=0,
        )

        # Store the loaded trajectory.
        self._t = t
        self._T_left = T_left
        self._T_right = T_right
        self._T_torso = T_torso

    def move_to_trajectory_pose(self, time_ratio):
        if (
            self._t is None
            or self._T_right is None
            or self._T_left is None
            or self._T_torso is None
        ):
            return
        time_index = round(time_ratio * len(self._t))
        print("Navigating to the pose at time ratio %g" % time_ratio)
        self._rainbow_interface.move_to_pose(
            T_right=self._T_right[time_index],
            T_left=self._T_left[time_index],
            T_torso=self._T_torso[time_index],
        )

    def run_trajectory(self):
        if self._t is None or self._T_right is None or self._T_left is None:
            return

        # Command starting pose
        print("Navigating to the starting pose")
        self.move_to_trajectory_pose(time_ratio=0)
        user_input = input("Press Enter to run the trajectory, or c to cancel >> ")
        if user_input.strip().lower() in ["c", "cancel", "q", "quit"]:
            return

        print("Running the trajectory")
        (
            timestamps_cmd,
            timestamps_true,
            T_left_true,
            T_right_true,
            T_torso_true,
            T_left_interp,
            T_right_interp,
            T_torso_interp,
        ) = self._rainbow_interface.run_trajectory_and_record(
            self._t,
            T_left=self._T_left,
            T_right=self._T_right,
            T_torso=self._T_torso,
            speed_reduction_factor=self._speed_reduction_factor,
            controller_type=self._controller_type,
        )

        fig, ax = plt.subplots(3)
        time_cmd = [t - timestamps_cmd[0] for t in timestamps_cmd]
        time_true = [t - timestamps_true[0] for t in timestamps_true]
        ax[0].plot(time_cmd, T_left_interp[:, 0, 3], "--", label="Command")
        ax[0].plot(time_true, T_left_true[:, 0, 3], "-", label="Measured")
        ax[0].set_ylabel("X (m)")
        ax[0].legend()
        ax[1].plot(time_cmd, T_left_interp[:, 1, 3], "--", label="Command")
        ax[1].plot(time_true, T_left_true[:, 1, 3], "-", label="Measured")
        ax[1].set_ylabel("Y (m)")
        ax[2].plot(time_cmd, T_left_interp[:, 2, 3], "--", label="Command")
        ax[2].plot(time_true, T_left_true[:, 2, 3], "-", label="Measured")
        ax[2].set_ylabel("Z (m)")
        ax[2].set_xlabel("Time (s)")
        fig.suptitle("Left EE Position Tracking")
        fig.savefig("left_ee_position_tracking.png")
        fig, ax = plt.subplots(3)
        ax[0].plot(time_cmd, T_right_interp[:, 0, 3], "--", label="Command")
        ax[0].plot(time_true, T_right_true[:, 0, 3], "-", label="Measured")
        ax[0].set_ylabel("X (m)")
        ax[0].legend()
        ax[1].plot(time_cmd, T_right_interp[:, 1, 3], "--", label="Command")
        ax[1].plot(time_true, T_right_true[:, 1, 3], "-", label="Measured")
        ax[1].set_ylabel("Y (m)")
        ax[2].plot(time_cmd, T_right_interp[:, 2, 3], "--", label="Command")
        ax[2].plot(time_true, T_right_true[:, 2, 3], "-", label="Measured")
        ax[2].set_ylabel("Z (m)")
        ax[2].set_xlabel("Time (s)")
        fig.suptitle("Right EE Position Tracking")
        fig.savefig("right_ee_position_tracking.png")
        fig, ax = plt.subplots(3)
        ax[0].plot(time_cmd, T_torso_interp[:, 0, 3], "--", label="Command")
        ax[0].plot(time_true, T_torso_true[:, 0, 3], "-", label="Measured")
        ax[0].set_ylabel("X (m)")
        ax[0].legend()
        ax[1].plot(time_cmd, T_torso_interp[:, 1, 3], "--", label="Command")
        ax[1].plot(time_true, T_torso_true[:, 1, 3], "-", label="Measured")
        ax[1].set_ylabel("Y (m)")
        ax[2].plot(time_cmd, T_torso_interp[:, 2, 3], "--", label="Command")
        ax[2].plot(time_true, T_torso_true[:, 2, 3], "-", label="Measured")
        ax[2].set_ylabel("Z (m)")
        ax[2].set_xlabel("Time (s)")
        fig.suptitle("Torso Position Tracking")
        fig.savefig("torso_position_tracking.png")
        print("Trajectory finished.")
        time.sleep(1)

        print("Done.")

    def print_menu(self):
        print(
            """
            menu [m]                                 : Print this menu
            home [h]                                 : Go to home position
            speed # [s #]                            : Set trajectory playback speed reduction factor
            load pour/scoop/stir # [l p/s/r #]       : Load trajectory index #
            time # [t #]                             : Move to the pose at time percent # (0-100)
            run [r]                                  : Run the trajectory
            offset right/left x/y/z # [o r/l x/y/z #]: Increase a gripper offset by the relative # cm
            offset [o]                               : Print the current gripper offsets
            pose [p]                                 : Print the current robot pose
            joints [j]                               : Print the current robot joint positions
            quit [q]                                 : Quit
            """
        )

    def process_user_input(self, user_input):
        user_input = user_input.strip().lower()
        user_input_split = [s.strip() for s in user_input.split(" ")]
        if user_input in ["quit", "q"]:
            return "quit"
        elif user_input in ["menu", "m"]:
            self.print_menu()
        elif user_input in ["home", "h"]:
            self._rainbow_interface.move_to_home()
        elif user_input_split[0] in ["speed", "s"]:
            if len(user_input_split) == 1:
                print(
                    "Current speed reduction factor: %g" % self._speed_reduction_factor
                )
            else:
                try:
                    self._speed_reduction_factor = float(user_input_split[1])
                except:
                    print("Invalid speed specified")
                    return
                # Print the result.
                self.process_user_input("speed")
        elif user_input_split[0] in ["load", "l"]:
            activity_type = user_input_split[1]
            try:
                trajectory_index = int(user_input_split[2])
            except:
                print("Invalid trajectory index specified")
                return
            if activity_type in ["pour", "p"]:
                self.load_pouring_trajectory(trajectory_index)
                self._previous_load_trajectory_command = user_input
            # TODO add other activity types here
        elif user_input_split[0] in ["time", "t"]:
            try:
                time_percent = float(user_input_split[1])
            except:
                print("Invalid time ratio specified")
                return
            if time_percent < 1:
                time_percent = time_percent * 100
            self.move_to_trajectory_pose(time_percent / 100)
        elif user_input_split[0] in ["run", "r"]:
            self.run_trajectory()
        elif user_input_split[0] in ["offset", "o"]:
            if len(user_input_split) == 1:
                print(
                    " Right gripper offset [m]:", self._right_gripper_offset_m.tolist()
                )
                print(
                    " Left  gripper offset [m]:", self._left_gripper_offset_m.tolist()
                )
            else:
                if user_input_split[1] in ["right", "r"]:
                    arms = ["right"]
                elif user_input_split[1] in ["left", "l"]:
                    arms = ["left"]
                elif user_input_split[1] in ["right left", "rightleft", "rl", "both"]:
                    arms = ["right", "left"]
                else:
                    print("Invalid arm side specified")
                    return
                try:
                    axis = ["x", "y", "z"].index(user_input_split[2])
                except:
                    print("Invalid axis specified")
                    return
                try:
                    relative_offset_cm = float(user_input_split[3])
                except:
                    print("Invalid offset amount specified")
                    return
                if "right" in arms:
                    self._right_gripper_offset_m[axis] += relative_offset_cm / 100
                if "left" in arms:
                    self._left_gripper_offset_m[axis] += relative_offset_cm / 100
                # Reload the trajectory to apply the new offset.
                self.process_user_input(self._previous_load_trajectory_command)
                # Print the current offsets.
                self.process_user_input("offset")
        elif user_input_split[0] in ["pose", "p"]:
            print("Right gripper pose:")
            print(self._rainbow_interface.get_pose("base", "ee_right").tolist())
            print("Left gripper pose:")
            print(self._rainbow_interface.get_pose("base", "ee_left").tolist())
            print("Torso pose:")
            print(self._rainbow_interface.get_pose("base", "link_torso_5").tolist())
        elif user_input_split[0] in ["joints", "j"]:
            print(self._rainbow_interface.get_position())
        else:
            print("Unknown menu option")


################################################
################################################
################################################

if __name__ == "__main__":
    experiment_interface = ExperimentInterface()
    experiment_interface.print_menu()
    previous_user_input = "m"
    print()
    experiment_interface.process_user_input("l p 42")
    print()
    while True:
        user_input = input("Enter command >> ").strip()
        if len(user_input) == 0:
            user_input = previous_user_input
        previous_user_input = user_input
        if experiment_interface.process_user_input(user_input) == "quit":
            break
