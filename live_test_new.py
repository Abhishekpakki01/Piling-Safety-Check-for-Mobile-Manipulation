import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO

# Spot SDK Imports
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2, geometry_pb2

class SpotNoodlePicker(Node):
    def __init__(self, robot_ip, username, password):
        super().__init__('spot_noodle_picker')
        self.model = YOLO('best.pt')
        
        # --- SPOT SDK SETUP ---
        self.sdk = bosdyn.client.create_standard_sdk('NoodlePicker')
        self.robot = self.sdk.create_robot(robot_ip)
        self.robot.authenticate(username, password)
        self.robot.time_sync.wait_for_sync()

        # Get Lease and keep it alive so the robot doesn't shut down
        self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
        self.lease_wallet = self.lease_client.take_lease()
        self.lease_keepalive = LeaseKeepAlive(self.lease_client)
        
        # Manipulation Client
        self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        # -----------------------

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)
        self.create_subscription(CompressedImage, '/spot/hand_color_image/compressed', 
                                 self.image_callback, qos)

        cv2.namedWindow("Gripper_View")
        cv2.setMouseCallback("Gripper_View", self.on_click)
        self.get_logger().info("Authenticated! Lease acquired. Click to pick.")

    def image_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        self.frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        results = self.model(self.frame, conf=0.4, verbose=False)
        annotated = results[0].plot()
        cv2.imshow("Gripper_View", annotated)
        cv2.waitKey(1)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.send_pick_command(x, y)

    def send_pick_command(self, x, y):
        self.get_logger().info(f"Sending Pick Command for pixel ({x}, {y})...")
        
        # Create the grasp request
        pick_vec = geometry_pb2.Vec2(x=x, y=y)
        # Use 'hand_color_image_sensor' as the reference
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            frame_name='hand_color_image_sensor'
        )
        
        request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        self.manip_client.manipulation_api_command(request)

def main():
    rclpy.init()
    # USE YOUR ACTUAL LAB CREDENTIALS HERE
    node = SpotNoodlePicker(robot_ip='192.168.80.3', username='admin', password='password123')
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
