import cv2
import numpy as np
from niryo_one_tcp_client import *
import logging

logging.basicConfig(level=logging.INFO)

robot_ip_address = "10.10.10.10"
gripper_used = RobotTool.GRIPPER_2

# Initial pick position
pick_pose = PoseObject(x=-0.098, y=0.271, z=0.247, roll=2.759, pitch=1.412, yaw=-1.124)
# Move to this new position after pick
intermediate_pose = PoseObject(x=0.153, y=0.03, z=0.301, roll=-1.095, pitch=1.462, yaw=-0.275)
sleep_joints = [0.0, 0.55, -1.2, 0.0, 0.0, 0.0]

# Function to perform camera calibration (camera intrinsic matrix) - placeholder
def get_camera_intrinsic_matrix():
    fx = 1000  # Focal length in pixels (x-axis)
    fy = 1000  # Focal length in pixels (y-axis)
    cx = 640   # Principal point x (image center)
    cy = 360   # Principal point y (image center)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float32)
    return K

# Function to calculate pixel radius based on real-world diameter
def calculate_pixel_radius(real_diameter, K, Z):
    fx = K[0, 0]
    pixel_radius = (real_diameter / 2) * (fx / Z)
    return int(pixel_radius)

# Detect black circles with specific size constraints
def detect_black_circles(frame, expected_pixel_radius, tolerance=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_circles = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        radius = int(radius)
        
        if abs(radius - expected_pixel_radius) <= tolerance:
            circularity = cv2.contourArea(contour) / (np.pi * radius**2)
            if 0.8 < circularity <= 1.2:  # Circularity threshold
                valid_circles.append((int(x), int(y), radius))
                cv2.circle(frame, (int(x), int(y)), radius, (0, 255, 0), 2)
    
    return valid_circles, frame

# Transform u, v pixel coordinates to world coordinates
def pixel_to_world(u, v, K, Z=0.5):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x_normalized = (u - cx) / fx
    y_normalized = (v - cy) / fy
    X = Z * x_normalized
    Y = Z * y_normalized
    return X, Y, Z

# Move to a given pose with an optional height offset
def move_to_pose(client, pose, height_offset=0):
    offset_pose = pose.copy_with_offsets(z_offset=height_offset)
    client.move_pose(*client.pose_to_list(offset_pose))

# Pick and place task
def pick_and_place(client, pick_pose, intermediate_pose, place_pose, height_offset=0.05, gripper_speed=400):
    move_to_pose(client, pick_pose, height_offset)
    client.open_gripper(gripper_used, gripper_speed)
    move_to_pose(client, pick_pose)
    client.close_gripper(gripper_used, gripper_speed)
    move_to_pose(client, pick_pose, height_offset)
    
    # Move to intermediate position
    move_to_pose(client, intermediate_pose, height_offset)
    move_to_pose(client, intermediate_pose)
    
    # Move to the detected place position
    move_to_pose(client, place_pose, height_offset)
    move_to_pose(client, place_pose)
    client.open_gripper(gripper_used, gripper_speed)
    move_to_pose(client, place_pose, height_offset)

def main():
    client = NiryoOneClient()
    try:
        client.connect(robot_ip_address)
        client.change_tool(gripper_used)
        client.calibrate(CalibrateMode.AUTO)
        
        # Get camera intrinsic matrix
        K = get_camera_intrinsic_matrix()
        
        # Known parameters for the black circle
        real_diameter = 0.022  # Diameter in meters (2.2 cm)
        Z = 0.5  # Depth from the camera in meters
        expected_pixel_radius = calculate_pixel_radius(real_diameter, K, Z)

        # Start camera stream
        cap = cv2.VideoCapture(1)  # Use 1 for external camera, 0 for built-in camera

        if not cap.isOpened():
            logging.error("Error: Cannot open camera")
            return

        logging.info("Starting pick-and-place task.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Cannot read frame")
                break

            valid_circles, output_frame = detect_black_circles(frame, expected_pixel_radius)

            # If a valid circle is detected, update place_pose and execute pick-and-place
            if valid_circles:
                u, v, r = valid_circles[0]  # Pick the first valid circle detected
                X, Y, Z = pixel_to_world(u, v, K)

                # Update place_pose with the coordinates of the detected circle
                place_pose = PoseObject(x=X, y=Y, z=Z, roll=0.0, pitch=1.57, yaw=-1.57)

                # Perform pick-and-place task
                pick_and_place(client, pick_pose, intermediate_pose, place_pose)
                logging.info(f"Moved to place position: x={X:.2f}, y={Y:.2f}, z={Z:.2f}")

            # Display the resulting frame with detected circles
            cv2.imshow('Video Stream', output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        # Move to a sleep position after the task
        client.move_joints(*sleep_joints)
        client.set_learning_mode(True)
    
    except Exception as e:
        logging.error(f"Error: {e}")
    finally:
        client.quit()

if __name__ == '__main__':
    main()
