import os
import time
import rclpy
import threading
import sys
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
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import manipulation_api_pb2, geometry_pb2, robot_state_pb2, arm_command_pb2, robot_command_pb2, trajectory_pb2
from bosdyn.client.frame_helpers import HAND_FRAME_NAME

class SpotNoodleMaster(Node):
    def __init__(self):
        super().__init__('spot_noodle_master')
        self.model = YOLO('best.pt')
        self.hand_frame = None
        self.body_frame = None
        self.is_standing = False
        self.has_noodle = False # State tracker

        # --- SDK Setup ---
        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodleMaster')
            self.robot = self.sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
            self.robot.authenticate(os.getenv('SPOT_USERNAME', 'admin'), os.getenv('SPOT_PASSWORD', 'password123'))
            self.robot.time_sync.wait_for_sync()

            self.lease_client = self.robot.ensure_client('lease')
            self.lease_wallet = self.lease_client.take() 
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.manip_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            print("--- Noodle Master System Online ---")
        except Exception as e:
            print(f"SDK Error: {e}"); sys.exit(1)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, '/spot/hand_color_image/compressed', self.hand_callback, qos)
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', self.body_callback, qos)

    def hand_callback(self, data):
        self.hand_frame = self.process_image(data)

    def body_callback(self, data):
        self.body_frame = self.process_image(data)

    def process_image(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is not None:
            results = self.model(frame, conf=0.4, verbose=False)
            return results[0].plot()
        return None

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_standing:
                threading.Thread(target=self.stand_sequence, daemon=True).start()
            elif not self.has_noodle:
                # If we don't have the noodle, click performs a GRASP
                threading.Thread(target=self.execute_grasp, args=(x, y, param), daemon=True).start()
            else:
                # If we HAVE the noodle, click performs a POKE
                threading.Thread(target=self.execute_snappy_poke, daemon=True).start()

    def stand_sequence(self):
        print("[ACTION] Powering up and standing...")
        self.robot.power_on()
        self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(2.0)
        self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        self.is_standing = True

    def execute_grasp(self, x, y, camera_type):
        source = 'hand_color_image' if camera_type == 'hand' else 'frontleft_fisheye_image'
        print(f"[GRASP] Target at ({x}, {y}) using {source}...")
        try:
            image_responses = self.image_client.get_image_from_sources([source])
            image = image_responses[0]
            pick_vec = geometry_pb2.Vec2(x=float(x), y=float(y))
            
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec,
                transforms_snapshot_for_camera=image.shot.transforms_snapshot,
                frame_name_image_sensor=image.shot.frame_name_image_sensor,
                camera_model=image.source.pinhole
            )
            request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
            cmd_response = self.manip_client.manipulation_api_command(request)
            
            # Monitor grasp status
            while True:
                feedback = self.manip_client.manipulation_api_feedback_command(
                    manipulation_api_pb2.ManipulationApiFeedbackRequest(manipulation_cmd_id=cmd_response.manipulation_cmd_id))
                if feedback.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                    print("[SUCCESS] Noodle secured!")
                    self.has_noodle = True
                    self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                    break
                elif feedback.current_state in [manipulation_api_pb2.MANIP_STATE_GRASP_FAILED, manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION]:
                    print("[FAILED] Grasp failed.")
                    break
                time.sleep(0.1)
        except Exception as e:
            print(f"Grasp Error: {e}")

    def execute_snappy_poke(self):
        print("[ACTION] Snappy Poke...")
        try:
            dist, duration = 0.40, 0.6
            traj = trajectory_pb2.SE3Trajectory()
            p1 = traj.points.add()
            p1.time_since_reference.seconds = 0
            p1.pose.position.x = 0.0; p1.pose.position.y = 0.0; p1.pose.position.z = 0.0; p1.pose.rotation.w = 1.0
            
            p2 = traj.points.add()
            p2.time_since_reference.nanos = int(duration * 1e9)
            p2.pose.position.x = dist; p2.pose.position.y = 0.0; p2.pose.position.z = 0.0; p2.pose.rotation.w = 1.0

            arm_req = arm_command_pb2.ArmCartesianCommand.Request(root_frame_name=HAND_FRAME_NAME, pose_trajectory_in_task=traj)
            poke_cmd = robot_command_pb2.RobotCommand()
            poke_cmd.synchronized_command.arm_command.arm_cartesian_command.CopyFrom(arm_req)
            
            self.command_client.robot_command(poke_cmd)
            time.sleep(duration + 0.5)
            self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        except Exception as e:
            print(f"Poke Error: {e}")

    def shutdown(self):
        print("\n--- Shutting Down ---")
        try:
            self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(1.0)
            self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            self.robot.power_off()
        except: pass

def main():
    rclpy.init()
    node = SpotNoodleMaster()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cv2.namedWindow("Search_and_Poke")
    cv2.setMouseCallback("Search_and_Poke", node.on_click, "body")

    print("1. Click to Stand\n2. Click noodle to Grasp\n3. Click anywhere to Poke")
    
    try:
        while rclpy.ok():
            if node.body_frame is not None:
                # Show status on screen
                status = "MODE: POKE" if node.has_noodle else "MODE: GRASP"
                if not node.is_standing: status = "CLICK TO STAND"
                cv2.putText(node.body_frame, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Search_and_Poke", node.body_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        node.shutdown(); cv2.destroyAllWindows(); rclpy.shutdown()

if __name__ == '__main__':
    main()
