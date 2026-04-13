import os
import time
import rclpy
import threading
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO

# Spot SDK
import bosdyn.client
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2, geometry_pb2

class SpotNoodlePicker(Node):
    def __init__(self):
        super().__init__('spot_noodle_picker')
        print("--- Noodle Picker: Pick, Place, and Retract ---")
        
        self.model = YOLO('best.pt')
        self.hand_frame = None
        self.body_frame = None
        self.is_powered = False 
        self.last_grasp_image = None
        self.last_grasp_pixel = None

        # --- SDK Setup ---
        robot_ip = os.getenv('SPOT_IP', '10.0.0.30')
        username = os.getenv('SPOT_USERNAME', 'admin')
        password = os.getenv('SPOT_PASSWORD', 'password123')

        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodlePickerClient')
            self.robot = self.sdk.create_robot(robot_ip)
            self.robot.authenticate(username, password)
            self.robot.time_sync.wait_for_sync()

            self.lease_client = self.robot.ensure_client('lease')
            self.lease_wallet = self.lease_client.take() 
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            
            print("SDK Success: Connected.")
        except Exception as e:
            print(f"SDK Error: {e}")
            return

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, '/spot/hand_color_image/compressed', self.hand_callback, qos)
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', self.body_callback, qos)

    def hand_callback(self, data):
        self.hand_frame = self.process_image(data)

    def body_callback(self, data):
        self.body_frame = self.process_image(data)

    def process_image(self, data):
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is not None:
                results = self.model(frame, conf=0.4, verbose=False)
                return results[0].plot()
        except:
            return None

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            source_camera = 'hand_color_image' if param == 'hand' else 'frontleft_fisheye_image'
            print(f"Grasping at ({x}, {y})...")

            try:
                image_responses = self.image_client.get_image_from_sources([source_camera])
                image = image_responses[0]
                pick_vec = geometry_pb2.Vec2(x=float(x), y=float(y))
                
                # Save context for returning later
                self.last_grasp_image = image
                self.last_grasp_pixel = pick_vec

                grasp = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=pick_vec,
                    transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                    frame_name_image_sensor=image.shot.frame_name_image_sensor,
                    camera_model=image.source.pinhole
                )
                
                request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
                cmd_response = self.manip_client.manipulation_api_command(request)
                
                threading.Thread(target=self.monitor_and_lift, 
                                 args=(cmd_response.manipulation_cmd_id,), 
                                 daemon=True).start()
                
            except Exception as e:
                print(f"Grasp Error: {e}")

    def monitor_and_lift(self, cmd_id):
        while True:
            try:
                feedback_req = manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_id)
                response = self.manip_client.manipulation_api_feedback_command(feedback_req)
                state = response.current_state
                
                if state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                    print("SUCCESS: Noodle secured! Lifting...")
                    self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                    break
                elif state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED or \
                     state == manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION:
                    print("Grasp ended. Returning to Ready.")
                    self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                    break
                time.sleep(0.1)
            except:
                break

    def stand_sequence(self):
        print("\n[ACTION] Standing...")
        self.robot.power_on()
        self.is_powered = True
        self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(2.0)
        print("[ACTION] Unstowing arm...")
        self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        time.sleep(1.0)

    def original_sit_sequence(self):
        if not self.is_powered: return
        print("\n--- INITIATING RETURN, RELEASE, AND STOW ---")
        try:
            # 1. Lower the arm to where we found the noodle
            print("[ACTION] Moving arm back to original spot...")
            self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
            time.sleep(2.0)

            # 2. Release
            print("[ACTION] Releasing noodle...")
            self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(1.0)
            
            # 3. THE "COME BACK" (Retract arm to body)
            print("[ACTION] Retracting arm to stow position...")
            self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(2.0)
            
            # 4. Final Sit
            print("[ACTION] Sitting and powering off...")
            self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            time.sleep(2.0)
            self.robot.power_off()
            print("Mission Complete.")
        except Exception as e:
            print(f"Shutdown error: {e}")

def main():
    rclpy.init()
    node = SpotNoodlePicker()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cv2.namedWindow("Body_Search_View")
    cv2.namedWindow("Hand_Grasp_View")
    cv2.setMouseCallback("Body_Search_View", node.on_mouse_click, "body")
    cv2.setMouseCallback("Hand_Grasp_View", node.on_mouse_click, "hand")

    try:
        input("\n>>> [ENTER] TO STAND...")
        node.stand_sequence()
        while rclpy.ok():
            if node.body_frame is not None: cv2.imshow("Body_Search_View", node.body_frame)
            if node.hand_frame is not None: cv2.imshow("Hand_Grasp_View", node.hand_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            time.sleep(0.01)
    finally:
        node.original_sit_sequence()
        cv2.destroyAllWindows()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
