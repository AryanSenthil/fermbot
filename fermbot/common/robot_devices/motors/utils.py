from typing import Protocol 

from fermbot.common.robot_devices.motors.configs import (
    FeetechMotorsBusConfig,
    MotorsBusConfig
)

class MotorsBus(Protocol):
    def motor_names(self): ...
    def set_calibration(self): ...
    def apply_calibration(self): ...
    def revert_calibration(self): ...
    def read(self): ...
    def write(self): ...

def make_motors_buses_from_configs(motors_bus_configs: dict[str, MotorsBusConfig]) -> list[MotorsBus]:
    motors_buses = {}

    for key, cfg in motors_bus_configs.items():
        if cfg.type == "feetech":
            from fermbot.common.robot_devices.motors.feetech import FeetechMotorsBus
            
            motors_buses[key] = FeetechMotorsBus(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.") 
        
    return motors_buses 



def make_motors_bus(motor_type: str, **kwargs) -> MotorsBus:
    if motor_type == "feetech":
        from fermbot.common.robot_devices.motors.feetech import FeetechMotorsBus

        config = FeetechMotorsBusConfig(**kwargs)
        return FeetechMotorsBus(config)

    else:
        raise ValueError(f"The motor type '{motor_type}' is not valid.")