from niryo_one_tcp_client import *
from niryo_one_camera import *
import cv2
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

robot_ip_address = "10.10.10.10"
gripper_used = RobotTool.GRIPPER_2
pick_pose = PoseObject(x=-0.098, y=0.3, z=0.21, roll=1.37, pitch=1.35, yaw=-1.408)

# List of place poses initialized with placeholders for dynamic updates
place_poses = [
    PoseObject(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),  # Placeholder for 4 detected places
    PoseObject(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
    PoseObject(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0),
    PoseObject(x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0)
]

sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]

# Function to get the camera intrinsic matrix
def get_camera_intrinsic_matrix():
    fx, fy = 1000, 1000  # Focal lengths (pixels)
    cx, cy = 640, 360    # Principal point (center of image)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K

# Function to detect black circles and get their centers
def detect_black_circles(frame, expected_radius, tolerance=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # Morphological operation to reduce noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        if abs(radius - expected_radius) <= tolerance:
            circularity = cv2.contourArea(contour) / (np.pi * radius**2)
            if 0.8 < circularity <= 1.2:
                centers.append((int(x), int(y)))
                cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), 2)
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
    return centers, frame

# Convert pixel (u, v) coordinates to world (x, y, z) coordinates
def pixel_to_world(u, v, K, Z=0.5):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return X, Y, Z

# Get the camera coordinates and return them
def get_camera_coordinates(niryo_client):
    _, mtx, dist = niryo_client.get_calibration_object()
    K = get_camera_intrinsic_matrix()
    real_diameter = 0.022  # 2.2 cm
    Z = 0.5  # Fixed depth (meters)
    expected_radius = int((real_diameter / 2) * (K[0, 0] / Z))

    # Initialize variables for storing coordinates
    coordinates = []
    
    status, img_compressed = niryo_client.get_img_compressed()
    if not status:
        raise Exception("Error with Niryo One's service")
    
    img_raw = uncompress_image(img_compressed)
    centers, _ = detect_black_circles(img_raw, expected_radius)
    
    for idx, (u, v) in enumerate(centers):
        X, Y, Z_world = pixel_to_world(u, v, K, Z)
        coordinates.append((X, Y, Z_world))
    
    return coordinates

# Function to perform the pick-and-place operation
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

def main():
    client = NiryoOneClient()
    try:
        client.connect(robot_ip_address)
        client.calibrate(CalibrateMode.AUTO)
        client.change_tool(gripper_used)

        logging.info("Starting pick-and-place tasks.")

        # Get the camera coordinates
        camera_coordinates = get_camera_coordinates(client)

        # If four coordinates are detected, assign them to place_poses
        if len(camera_coordinates) == 4:
            for i in range(4):
                x, y, z = camera_coordinates[i]
                place_poses[i] = PoseObject(x=x, y=y, z=z, roll=0, pitch=0, yaw=0)
        
            # Perform the pick-and-place for each detected place pose
            for place_pose in place_poses:
                pick_and_place(client, pick_pose, place_pose)
        
        else:
            logging.error("Could not detect four place positions.")
        
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        client.quit()

if __name__ == '__main__':
    main()
