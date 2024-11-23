from niryo_one_tcp_client import *
from niryo_one_camera import *
import cv2
import numpy as np

# Robot IP and Observation Pose
robot_ip_address = "10.10.10.10"
observation_pose = PoseObject(
    x=0.2, y=0.0, z=0.34,
    roll=0, pitch=1.57, yaw=-0.2,
)

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

def video_stream(niryo_one_client):
    _, mtx, dist = niryo_one_client.get_calibration_object()
    K = get_camera_intrinsic_matrix()  # Use custom intrinsic matrix
    
    real_diameter = 0.022  # 2.2 cm
    Z = 0.5  # Fixed depth (meters)
    expected_radius = int((real_diameter / 2) * (K[0, 0] / Z))

    cv2.namedWindow("Niryo Camera Stream", cv2.WINDOW_NORMAL)

    while True:
        status, img_compressed = niryo_one_client.get_img_compressed()
        if not status:
            print("Error with Niryo One's service")
            break
        
        img_raw = uncompress_image(img_compressed)
        
        centers, output_frame = detect_black_circles(img_raw, expected_radius)
        for (u, v) in centers:
            X, Y, Z_world = pixel_to_world(u, v, K, Z)
            print(f"Circle detected at (u, v): ({u}, {v}) --> World (X, Y, Z): ({X:.2f}, {Y:.2f}, {Z_world:.2f})")
            # Display coordinates on the frame
            cv2.putText(output_frame, f"({X:.2f}, {Y:.2f}, {Z_world:.2f})", (u + 10, v - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Niryo Camera Stream", output_frame)

        if cv2.waitKey(5) in [27, ord("q")]:
            break
    
    niryo_one_client.set_learning_mode(True)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    client = NiryoOneClient()
    client.connect(robot_ip_address)
    client.calibrate(CalibrateMode.AUTO)
    
    video_stream(client)
    
    client.quit()
