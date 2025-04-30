from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState, START
from typing import List, Dict, Any, Literal, Optional, Tuple 
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from langchain_ollama import ChatOllama
import subprocess
import re
from pydantic import BaseModel, Field

# Define configuration parameters
class MotorConfig(TypedDict):
    arm_type: Literal["follower", "leader"]
    port: str
    motor_ids: List[int]
    current_index: int
    brand: str
    model: str
    baudrate: int
    completed: bool
    error_count: int

# Initialize configuration state
config_state = {
    "follower": {
        "arm_type": "follower",
        "port": "/dev/ttyACM0",
        "motor_ids": [1, 2, 3, 4, 5, 6],
        "current_index": 0,
        "brand": "feetech",
        "model": "sts3215",
        "baudrate": 1000000,
        "completed": False,
        "error_count": 0
    },
    "leader": {
        "arm_type": "leader",
        "port": "/dev/ttyACM1",
        "motor_ids": [1, 2, 3, 4, 5, 6],
        "current_index": 0,
        "brand": "feetech",
        "model": "sts3215",
        "baudrate": 1000000,
        "completed": False,
        "error_count": 0
    }
}


# Track the current arm we're working with 
current_arm = "follower"

# Define the tool to execute motor configuration command 
@tool 
def configure_motor(arm_type: str, motor_id:int):
    """Configure the motor with the given parameters"""
    global config_state, current_arm

    arm_config = config_state[arm_type]

    # Build the command
    cmd = [
        "python", "fermbot/scripts/configure_motor.py",
        "--port", arm_config["port"],
        "--brand", arm_config["brand"],
        "--model", arm_config["model"],
        "--baudrate", str(arm_config["baudrate"]),
        "--ID", str(motor_id)
    ]
    
    try:
        # Execute the command and capture output
        process = subprocess.run(cmd, capture_output=True, text=True)
        output = process.stdout
        
        # Check for success
        if "Motor index found" in output or "Present Position" in output:
            message = f"Motor {motor_id} of the {arm_type} arm configured successfully!\n\n{output}"
            
            # Update the current index if this is the expected next motor
            if motor_id == arm_config["motor_ids"][arm_config["current_index"]]:
                arm_config["current_index"] += 1
                config_state[arm_type] = arm_config
                
                # If we've finished all motors for this arm, mark as completed
                if arm_config["current_index"] >= len(arm_config["motor_ids"]):
                    config_state[arm_type]["completed"] = True
            
            return message
        else:
            # Handle error
            error_message = f"Error configuring motor {motor_id} of the {arm_type} arm:\n\n{output}\n\nPlease check the connection and try again."
            config_state[arm_type]["error_count"] += 1
            return error_message
    
    except Exception as e:
        # Handle exception
        error_message = f"Exception occurred while configuring motor {motor_id} of the {arm_type} arm: {str(e)}"
        config_state[arm_type]["error_count"] += 1
        return error_message

# Define a function to parse user messages for configuration intent
def parse_configuration_intent(message: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Parse user message to detect configuration intent for specific arm/motor
    Returns a tuple of (arm_type, motor_id) or (None, None) if no intent detected
    """
    # Define patterns to match
    follower_pattern = r"(?:configure|setup).*(?:follower|first).*(?:arm|motor)\s*(?:(\d+))?"
    leader_pattern = r"(?:configure|setup).*(?:leader|second).*(?:arm|motor)\s*(?:(\d+))?"
    
    # Check for follower arm intent
    follower_match = re.search(follower_pattern, message.lower())
    if follower_match:
        motor_id = follower_match.group(1)
        return "follower", int(motor_id) if motor_id else None
    
    # Check for leader arm intent
    leader_match = re.search(leader_pattern, message.lower())
    if leader_match:
        motor_id = leader_match.group(1)
        return "leader", int(motor_id) if motor_id else None
    
    return None, None


# Define a function to suggest the next step based on configuration state
def suggest_next_configuration_step() -> Tuple[Optional[str], Optional[int]]:
    """
    Suggests the next logical configuration step based on current state
    Returns a tuple of (arm_type, motor_id) or (None, None) if all done
    """
    global config_state
    
    # Check if follower arm still needs configuration
    if not config_state["follower"]["completed"]:
        arm_type = "follower"
        current_index = config_state["follower"]["current_index"]
        if current_index < len(config_state["follower"]["motor_ids"]):
            return arm_type, config_state["follower"]["motor_ids"][current_index]
    
    # If follower is done, check leader arm
    if not config_state["leader"]["completed"]:
        arm_type = "leader"
        current_index = config_state["leader"]["current_index"]
        if current_index < len(config_state["leader"]["motor_ids"]):
            return arm_type, config_state["leader"]["motor_ids"][current_index]
    
    # If both are done, return None
    return None, None


# Define human interaction class
class AskHuman(BaseModel):
    """Ask the human a question"""
    question: str = Field(..., description="The question to ask the human")
