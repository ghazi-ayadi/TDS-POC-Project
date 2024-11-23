# Imports
from niryo_one_tcp_client import *
import logging

logging.basicConfig(level=logging.INFO)

robot_ip_address = "10.10.10.10"
gripper_used = RobotTool.GRIPPER_2
pick_pose = PoseObject(x=0.003, y=0.286, z=0.202, roll=2.029, pitch=1.461, yaw=-2.097)
place_pose = PoseObject(x=0.286, y=0.003, z=0.202, roll=-3.082, pitch=1.457, yaw=-2.579)
sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]

def move_to_pose(client, pose, height_offset=0):
    """Move to pose with optional height offset."""
    offset_pose = pose.copy_with_offsets(z_offset=height_offset)
    client.move_pose(*client.pose_to_list(offset_pose))

def pick_and_place(client, pick_pose, place_pose, height_offset=0.05, gripper_speed=400):
    """Pick-and-place task."""
    move_to_pose(client, pick_pose, height_offset)
    client.open_gripper(gripper_used, gripper_speed)
    move_to_pose(client, pick_pose)
    client.close_gripper(gripper_used, gripper_speed)
    move_to_pose(client, pick_pose, height_offset)
    
    move_to_pose(client, place_pose, height_offset)
    move_to_pose(client, place_pose)
    client.open_gripper(gripper_used, gripper_speed)
    move_to_pose(client, place_pose, height_offset)

if __name__ == '__main__':
    client = NiryoOneClient()
    try:
        client.connect(robot_ip_address)
        client.change_tool(gripper_used)
        client.calibrate(CalibrateMode.AUTO)
        
        logging.info("Starting pick-and-place task.")
        pick_and_place(client, pick_pose, place_pose)
        
        client.move_joints(*sleep_joints)
        client.set_learning_mode(True)
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        client.quit()
