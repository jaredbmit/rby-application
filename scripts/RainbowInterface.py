import rby1_sdk as sdk
import numpy as np
from scipy import interpolate
import scipy.spatial.transform as tf
import sys
import time
from typing import Optional
import queue

# Optimal Controller Presets
WEIGHT = 0.0015  # main parameter - higher will try to track better
CENTERING_WEIGHT = 0.0001
BODY_CENTERING_WEIGHT = 0.001
STOP_COST = WEIGHT * WEIGHT * 2e-3
VELOCITY_LIMIT_SCALE = 1.0
MIN_DELTA_COST = WEIGHT * WEIGHT * 2e-3
PATIENCE = 10
CONTROL_HOLD_TIME = 1

# Traj interpolation and Control frequency
DT = 1. / 500.


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


class RainbowInterface:
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

        self._q_home_position = None
        self._q_home_pose = {
            "left": None,
            "right": None,
            "torso": None,
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

        self.robot.reset_fault_control_manager()

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

    def set_home_position(self, q_home_position):
        self._q_home_position = q_home_position

    def set_home_pose(self, limb, q_home_pose):
        self._q_home_pose[limb] = np.array(q_home_pose)

    def get_home_pose(self, limb):
        return self._q_home_pose[limb]

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

    def get_position(self):
        position = self.robot.get_state().position
        return position

    def move_to_home(self):
        if self._q_home_position is None:
            print("No home position available.")
            return
        rc = sdk.RobotCommandBuilder().set_command(
            sdk.ComponentBasedCommandBuilder().set_body_command(
                sdk.BodyCommandBuilder().set_command(
                    sdk.JointPositionCommandBuilder()
                    .set_position(self._q_home_position)
                    .set_minimum_time(3)
                )
            )
        )

        rv = self.robot.send_command(rc, 10).get()

        if rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to reset pose.")
            return 1

        return 0

    def move_to_pose(
        self, T_right, T_left, T_torso, duration_s=5, controller_type="optimal"
    ):
        dt_s = DT
        time_mini_s = np.arange(0, duration_s, dt_s)

        T_right_init = self.get_pose("base", "ee_right")
        T_right_mini = np.stack([T_right_init, T_right, T_right])
        T_right_mini_interp = interpolate_trajectory(
            [0, duration_s / 2, duration_s], time_mini_s, T_right_mini
        )
        T_left_init = self.get_pose("base", "ee_left")
        T_left_mini = np.stack([T_left_init, T_left, T_left])
        T_left_mini_interp = interpolate_trajectory(
            [0, duration_s / 2, duration_s], time_mini_s, T_left_mini
        )
        T_torso_init = self.get_pose("base", "link_torso_5")
        T_torso_mini = np.stack([T_torso_init, T_torso, T_torso])
        T_torso_mini_interp = interpolate_trajectory(
            [0, duration_s / 2, duration_s], time_mini_s, T_torso_mini
        )

        if controller_type == "optimal":
            self.command_optimal_trajectory(
                time_mini_s,
                T_left=T_left_mini_interp,
                T_right=T_right_mini_interp,
                T_torso=T_torso_mini_interp,
            )
        elif controller_type == "cartesian":
            self.command_cartesian_trajectory(
                time_mini_s,
                T_left=T_left_mini_interp,
                T_right=T_right_mini_interp,
                T_torso=T_torso_mini_interp,
            )
        else:
            print("Unknown controller type specified")
            return

        time.sleep(1)

    # Main function 1
    def run_trajectory_and_record(
        self,
        t: np.ndarray,
        T_left: Optional[np.ndarray] = None,
        T_right: Optional[np.ndarray] = None,
        T_torso: Optional[np.ndarray] = None,
        speed_reduction_factor: Optional[float] = 1,
        controller_type="optimal",
    ):
        state_queue = queue.Queue()

        def callback(robot_state, control_manager_state):
            timestamp = time.perf_counter()
            state_queue.put((timestamp, robot_state))

        self.robot.start_state_update(callback, rate=10)

        if controller_type == "optimal":
            (timestamps_cmd, T_left_cmd, T_right_cmd, T_torso_cmd) = (
                self.command_optimal_trajectory(
                    t,
                    T_left=T_left,
                    T_right=T_right,
                    T_torso=T_torso,
                    speed_reduction_factor=speed_reduction_factor,
                )
            )
        elif controller_type == "cartesian":
            (timestamps_cmd, T_left_cmd, T_right_cmd, T_torso_cmd) = (
                self.command_cartesian_trajectory(
                    t,
                    T_left=T_left,
                    T_right=T_right,
                    T_torso=T_torso,
                    speed_reduction_factor=speed_reduction_factor,
                )
            )
        else:
            print("Unknown controller type specified")
            return

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

        return (
            timestamps_cmd,
            timestamps_meas,
            T_left_true,
            T_right_true,
            T_torso_true,
            T_left_cmd,
            T_right_cmd,
            T_torso_cmd,
        )

    # Main function 2
    def command_optimal_trajectory(
        self,
        t: np.ndarray,
        T_left: Optional[np.ndarray] = None,
        T_right: Optional[np.ndarray] = None,
        T_torso: Optional[np.ndarray] = None,
        speed_reduction_factor: Optional[float] = 1,
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

        # Re-interpolate to adjust the speed.
        dt = DT
        time_in = speed_reduction_factor * t
        time_out = np.arange(0, np.max(time_in), dt)
        if T_right is not None:
            T_right = interpolate_trajectory(time_in, time_out, T_right)
        if T_left is not None:
            T_left = interpolate_trajectory(time_in, time_out, T_left)
        if T_torso is not None:
            T_torso = interpolate_trajectory(time_in, time_out, T_torso)
        t = time_out

        duration = int(max(t)) + 4
        stream = self.robot.create_command_stream(duration)
        timestamps = []

        for i in range(len(t)):
            if i % round(len(t) / 10) == 0:
                print(
                    "Commanding timestep index %4d/%d (%0.1f%%)"
                    % (i, len(t) - 1, 100 * i / (len(t) - 1))
                )
            iteration_start_time_s = time.time()
            # if i == len(t)-1-1:
            #     print(i, t[i+1], T_right[i + 1])

            if i == 0:
                dt = 2
            else:
                dt = float(t[i] - t[i - 1])

            # Using optimal controller
            optimal_command = (
                sdk.OptimalControlCommandBuilder()
                .set_command_header(
                    sdk.CommandHeaderBuilder().set_control_hold_time(CONTROL_HOLD_TIME)
                )
                .add_joint_position_target(
                    "torso_0", self._q_home_position[0], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "torso_1", self._q_home_position[1], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "torso_2", self._q_home_position[2], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "torso_3", self._q_home_position[3], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "torso_4", self._q_home_position[4], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "torso_5", self._q_home_position[5], BODY_CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_0", self._q_home_position[6], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_1", self._q_home_position[7], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_2", self._q_home_position[8], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_3", self._q_home_position[9], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_4", self._q_home_position[10], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_5", self._q_home_position[11], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "right_arm_6", self._q_home_position[12], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_0", self._q_home_position[13], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_1", self._q_home_position[14], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_2", self._q_home_position[15], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_3", self._q_home_position[16], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_4", self._q_home_position[17], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_5", self._q_home_position[18], CENTERING_WEIGHT
                )
                .add_joint_position_target(
                    "left_arm_6", self._q_home_position[19], CENTERING_WEIGHT
                )
                .set_velocity_limit_scaling(VELOCITY_LIMIT_SCALE)
                .set_stop_cost(STOP_COST)
                .set_min_delta_cost(MIN_DELTA_COST)
                .set_patience(PATIENCE)
            )

            if T_left is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "ee_left", T_left[i], WEIGHT, WEIGHT
                )

            if T_right is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "ee_right", T_right[i], WEIGHT, WEIGHT
                )

            if T_torso is not None:
                optimal_command = optimal_command.add_cartesian_target(
                    "base", "link_torso_5", T_torso[i], WEIGHT, WEIGHT
                )

            rc = sdk.RobotCommandBuilder().set_command(
                sdk.ComponentBasedCommandBuilder().set_body_command(optimal_command)
            )

            rv = stream.send_command(rc)

            timestamps.append(time.perf_counter())

            delay_duration_s = dt - (time.time() - iteration_start_time_s)
            if delay_duration_s > 0:
                time.sleep(delay_duration_s)

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return timestamps

        return (timestamps, T_left, T_right, T_torso)

    def command_cartesian_trajectory(
        self,
        t: np.ndarray,
        T_left: Optional[np.ndarray] = None,
        T_right: Optional[np.ndarray] = None,
        T_torso: Optional[np.ndarray] = None,
        speed_reduction_factor: Optional[float] = 1,
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

        LINEAR_VELOCITY_LIMIT = 1.5
        ANGULAR_VELOCITY_LIMIT = np.pi * 1.5
        ACCELERATION_LIMIT = 1.0
        STOP_ORIENTATION_TRACKING_ERROR = 1e-4
        STOP_POSITION_TRACKING_ERROR = 1e-3

        # Re-interpolate to adjust the speed.
        dt = DT
        time_in = speed_reduction_factor * t
        time_out = np.arange(0, np.max(time_in), dt)
        T_right = interpolate_trajectory(time_in, time_out, T_right)
        T_left = interpolate_trajectory(time_in, time_out, T_left)
        T_torso = interpolate_trajectory(time_in, time_out, T_torso)
        t = time_out

        duration = int(max(t)) + 4
        stream = self.robot.create_command_stream(duration)
        timestamps = []

        for i in range(len(t)):
            if i % round(len(t) / 10) == 0:
                print(
                    "Commanding timestep index %4d/%d (%0.1f%%)"
                    % (i, len(t) - 1, 100 * i / (len(t) - 1))
                )
            iteration_start_time_s = time.time()

            if i == 0:
                dt = 2
            else:
                dt = float(t[i] - t[i - 1])
            minimum_time = dt

            # Using optimal controller
            body_command = sdk.BodyComponentBasedCommandBuilder()
            if T_torso is not None:
                body_command.set_torso_command(
                    sdk.CartesianCommandBuilder()
                    .add_target(
                        "base",
                        "link_torso_5",
                        T_torso[i],
                        LINEAR_VELOCITY_LIMIT,
                        ANGULAR_VELOCITY_LIMIT,
                        ACCELERATION_LIMIT,
                    )
                    .set_minimum_time(minimum_time)
                    .set_stop_orientation_tracking_error(
                        STOP_ORIENTATION_TRACKING_ERROR
                    )
                    .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
                )
            if T_right is not None:
                body_command.set_right_arm_command(
                    sdk.CartesianCommandBuilder()
                    .add_target(
                        "base",
                        "ee_right",
                        T_right[i],
                        LINEAR_VELOCITY_LIMIT,
                        ANGULAR_VELOCITY_LIMIT,
                        ACCELERATION_LIMIT,
                    )
                    .set_minimum_time(minimum_time)
                    .set_stop_orientation_tracking_error(
                        STOP_ORIENTATION_TRACKING_ERROR
                    )
                    .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
                )
            if T_left is not None:
                body_command.set_left_arm_command(
                    sdk.CartesianCommandBuilder()
                    .add_target(
                        "base",
                        "ee_left",
                        T_left[i],
                        LINEAR_VELOCITY_LIMIT,
                        ANGULAR_VELOCITY_LIMIT,
                        ACCELERATION_LIMIT,
                    )
                    .set_minimum_time(minimum_time)
                    .set_stop_orientation_tracking_error(
                        STOP_ORIENTATION_TRACKING_ERROR
                    )
                    .set_stop_position_tracking_error(STOP_POSITION_TRACKING_ERROR)
                )

            rc = sdk.RobotCommandBuilder().set_command(
                sdk.ComponentBasedCommandBuilder().set_body_command(body_command)
            )
            rv = stream.send_command(rc)

            timestamps.append(time.perf_counter())

            delay_duration_s = dt - (time.time() - iteration_start_time_s)
            if delay_duration_s > 0:
                time.sleep(delay_duration_s)

        if rv.finish_code == sdk.RobotCommandFeedback.FinishCode.Unknown:
            print("Control finish unknown.")
        elif rv.finish_code != sdk.RobotCommandFeedback.FinishCode.Ok:
            print("Error: Failed to conduct demo motion.")
            return timestamps

        return (timestamps, T_left, T_right, T_torso)
