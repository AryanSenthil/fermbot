"""Logic to calibrate a robot arm built with feetech motors"""

import time

import numpy as np

from fermbot.common.robot_devices.motors.feetech import (
    CalibrationMode,
    TorqueMode,
    convert_degrees_to_steps,
)
from fermbot.common.robot_devices.motors.utils import MotorsBus


# The following positions are provided in nominal degree range ]-180, +180[
# For more info on these constants, see comments in the code where they get used.
ZERO_POSITION_DEGREE = 0
ROTATED_POSITION_DEGREE = 90


def reset_middle_positions(arm: MotorsBus):
    input("Please move the robot to the new middle position for calibration, then press Enter...")
    # Write 128 to Torque_Enable for all motors.
    arm.write("Torque_Enable", 128)


def assert_drive_mode(drive_mode):
    # `drive_mode` is in [0,1] with 0 means original rotation direction for the motor, and 1 means inverted.
    if not np.all(np.isin(drive_mode, [0, 1])):
        raise ValueError(f"`drive_mode` contains values other than 0 or 1: ({drive_mode})")


def apply_drive_mode(position, drive_mode):
    assert_drive_mode(drive_mode)
    # Convert `drive_mode` from [0, 1] with 0 indicates original rotation direction and 1 inverted,
    # to [-1, 1] with 1 indicates original rotation direction and -1 inverted.
    signed_drive_mode = -(drive_mode * 2 - 1)
    position *= signed_drive_mode
    return position


def move_until_block(arm, motor_name, positive_direction=True, while_move_hook=None):
    count = 0
    while True:
        present_pos = arm.read("Present_Position", motor_name)
        if positive_direction:
            # Move +100 steps every time. Lower the steps to lower the speed at which the arm moves.
            arm.write("Goal_Position", present_pos + 100, motor_name)
        else:
            arm.write("Goal_Position", present_pos - 100, motor_name)

        if while_move_hook is not None:
            while_move_hook()

        present_pos = arm.read("Present_Position", motor_name).item()
        present_speed = arm.read("Present_Speed", motor_name).item()
        present_current = arm.read("Present_Current", motor_name).item()
        # present_load = arm.read("Present_Load", motor_name).item()
        # present_voltage = arm.read("Present_Voltage", motor_name).item()
        # present_temperature = arm.read("Present_Temperature", motor_name).item()

        # print(f"{present_pos=}")
        # print(f"{present_speed=}")
        # print(f"{present_current=}")
        # print(f"{present_load=}")
        # print(f"{present_voltage=}")
        # print(f"{present_temperature=}")

        if present_speed == 0 and present_current > 40:
            count += 1
            if count > 100 or present_current > 300:
                return present_pos
        else:
            count = 0


def move_to_calibrate(
    arm,
    motor_name,
    invert_drive_mode=False,
    positive_first=True,
    in_between_move_hook=None,
    while_move_hook=None,
):
    initial_pos = arm.read("Present_Position", motor_name)

    if positive_first:
        p_present_pos = move_until_block(
            arm, motor_name, positive_direction=True, while_move_hook=while_move_hook
        )
    else:
        n_present_pos = move_until_block(
            arm, motor_name, positive_direction=False, while_move_hook=while_move_hook
        )

    if in_between_move_hook is not None:
        in_between_move_hook()

    if positive_first:
        n_present_pos = move_until_block(
            arm, motor_name, positive_direction=False, while_move_hook=while_move_hook
        )
    else:
        p_present_pos = move_until_block(
            arm, motor_name, positive_direction=True, while_move_hook=while_move_hook
        )

    zero_pos = (n_present_pos + p_present_pos) / 2

    calib_data = {
        "initial_pos": initial_pos,
        "homing_offset": zero_pos if invert_drive_mode else -zero_pos,
        "invert_drive_mode": invert_drive_mode,
        "drive_mode": -1 if invert_drive_mode else 0,
        "zero_pos": zero_pos,
        "start_pos": n_present_pos if invert_drive_mode else p_present_pos,
        "end_pos": p_present_pos if invert_drive_mode else n_present_pos,
    }
    return calib_data


def apply_offset(calib, offset):
    calib["zero_pos"] += offset
    if calib["drive_mode"]:
        calib["homing_offset"] += offset
    else:
        calib["homing_offset"] -= offset
    return calib


def run_arm_manual_calibration(arm: MotorsBus, robot_type: str, arm_name: str, arm_type: str):
    """This function ensures that a neural network trained on data collected on a given robot
    can work on another robot. For instance before calibration, setting a same goal position
    for each motor of two different robots will get two very different positions. But after calibration,
    the two robots will move to the same position.To this end, this function computes the homing offset
    and the drive mode for each motor of a given robot.

    Homing offset is used to shift the motor position to a ]-2048, +2048[ nominal range (when the motor uses 2048 steps
    to complete a half a turn). This range is set around an arbitrary "zero position" corresponding to all motor positions
    being 0. During the calibration process, you will need to manually move the robot to this "zero position".

    Drive mode is used to invert the rotation direction of the motor. This is useful when some motors have been assembled
    in the opposite orientation for some robots. During the calibration process, you will need to manually move the robot
    to the "rotated position".

    After calibration, the homing offsets and drive modes are stored in a cache.

    Example of usage:
    ```python
    run_arm_calibration(arm, "so100", "left", "follower")
    ```
    """
    if (arm.read("Torque_Enable") != TorqueMode.DISABLED.value).any():
        raise ValueError("To run calibration, the torque must be disabled on all motors.")

    print(f"\nRunning calibration of {robot_type} {arm_name} {arm_type}...")

    reset_middle_positions(arm)

    print("\nMove arm to zero position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="zero"))
    input("Press Enter to continue...")

    # We arbitrarily chose our zero target position to be a straight horizontal position with gripper upwards and closed.
    # It is easy to identify and all motors are in a "quarter turn" position. Once calibration is done, this position will
    # correspond to every motor angle being 0. If you set all 0 as Goal Position, the arm will move in this position.
    zero_target_pos = convert_degrees_to_steps(ZERO_POSITION_DEGREE, arm.motor_models)

    # Compute homing offset so that `present_position + homing_offset ~= target_position`.
    zero_pos = arm.read("Present_Position")
    homing_offset = zero_target_pos - zero_pos

    # The rotated target position corresponds to a rotation of a quarter turn from the zero position.
    # This allows to identify the rotation direction of each motor.
    # For instance, if the motor rotates 90 degree, and its value is -90 after applying the homing offset, then we know its rotation direction
    # is inverted. However, for the calibration being successful, we need everyone to follow the same target position.
    # Sometimes, there is only one possible rotation direction. For instance, if the gripper is closed, there is only one direction which
    # corresponds to opening the gripper. When the rotation direction is ambiguous, we arbitrarily rotate clockwise from the point of view
    # of the previous motor in the kinetic chain.
    print("\nMove arm to rotated target position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rotated"))
    input("Press Enter to continue...")

    rotated_target_pos = convert_degrees_to_steps(ROTATED_POSITION_DEGREE, arm.motor_models)

    # Find drive mode by rotating each motor by a quarter of a turn.
    # Drive mode indicates if the motor rotation direction should be inverted (=1) or not (=0).
    rotated_pos = arm.read("Present_Position")
    drive_mode = (rotated_pos < zero_pos).astype(np.int32)

    # Re-compute homing offset to take into account drive mode
    rotated_drived_pos = apply_drive_mode(rotated_pos, drive_mode)
    homing_offset = rotated_target_pos - rotated_drived_pos

    print("\nMove arm to rest position")
    # print("See: " + URL_TEMPLATE.format(robot=robot_type, arm=arm_type, position="rest"))
    input("Press Enter to continue...")
    print()

    # Joints with rotational motions are expressed in degrees in nominal range of [-180, 180]
    calib_modes = []
    for name in arm.motor_names:
        if name == "gripper":
            calib_modes.append(CalibrationMode.LINEAR.name)
        else:
            calib_modes.append(CalibrationMode.DEGREE.name)

    calib_dict = {
        "homing_offset": homing_offset.tolist(),
        "drive_mode": drive_mode.tolist(),
        "start_pos": zero_pos.tolist(),
        "end_pos": rotated_pos.tolist(),
        "calib_mode": calib_modes,
        "motor_names": arm.motor_names,
    }
    return calib_dict