from niryo_one_tcp_client import *
import logging

logging.basicConfig(level=logging.INFO)

robot_ip_address = "10.10.10.10"
gripper_used = RobotTool.GRIPPER_2
pick_pose = PoseObject(x=-0.098, y=0.3, z=0.21, roll=1.37, pitch=1.35, yaw=-1.408)

# List of place poses+
place_poses = [
    PoseObject(x=0.2925, y=0.035, z=0.162, roll=0.947, pitch=1.502, yaw=2.945),
    PoseObject(x=0.290, y=0.073, z=0.159, roll=1.715, pitch=1.435, yaw=-2.633),
    PoseObject(x=0.275, y=0.096, z=0.158, roll=1.997, pitch=1.519, yaw=-2.298),
    PoseObject(x=0.262, y=0.131, z=0.161, roll=2.919, pitch=1.484, yaw=-1.308)
]

sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]

def move_to_pose(client, pose, height_offset=0):
    """Move to pose with optional height offset."""
    offset_pose = pose.copy_with_offsets(z_offset=height_offset)
    client.move_pose(*client.pose_to_list(offset_pose))

def pick_and_place(client, pick_pose, place_pose, height_offset=0.05, gripper_speed=800):
    """Pick-and-place task."""
    # Move to pick pose (a little above the object)
    move_to_pose(client, pick_pose, height_offset)
    
    # Open gripper before picking
    client.open_gripper(gripper_used, gripper_speed)
    
    # Move to pick pose (contact the object)
    move_to_pose(client, pick_pose)
    
    # Close the gripper to pick the object
    client.close_gripper(gripper_used, gripper_speed)
    
    # Move back to a higher position (away from the object)
    move_to_pose(client, pick_pose, height_offset)
    
    # Move to place pose
    move_to_pose(client, place_pose, height_offset)
    move_to_pose(client, place_pose)
    
    # Open gripper to release the object
    client.open_gripper(gripper_used, gripper_speed)
    
    # Move to a higher position again
    move_to_pose(client, place_pose, height_offset=0.1)
    
    # Close gripper if needed (in case you want to pick again)
    client.close_gripper(gripper_used, gripper_speed)
    
    # Move back to place pose
    move_to_pose(client, place_pose)

if __name__ == '__main__':
    client = NiryoOneClient()
    try:
        client.connect(robot_ip_address)
        client.change_tool(gripper_used)
        client.calibrate(CalibrateMode.AUTO)
        
        logging.info("Starting pick-and-place tasks.")
        
        # Perform the pick-and-place for each place pose
        for place_pose in place_poses:
            pick_and_place(client, pick_pose, place_pose)
        
        client.move_joints(*sleep_joints)
        client.set_learning_mode(True)
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        client.quit()
