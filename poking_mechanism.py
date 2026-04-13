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

# Spot SDK
import bosdyn.client
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import HAND_FRAME_NAME
from bosdyn.api import robot_state_pb2, arm_command_pb2, robot_command_pb2, trajectory_pb2

class SpotPokeBodyCam(Node):
    def __init__(self):
        super().__init__('spot_poke_body_cam')
        self.body_frame = None
        self.is_standing = False

        try:
            self.sdk = bosdyn.client.create_standard_sdk('BodyCamPokeTool')
            self.robot = self.sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
            self.robot.authenticate(os.getenv('SPOT_USERNAME', 'admin'), os.getenv('SPOT_PASSWORD', 'password123'))
            self.robot.time_sync.wait_for_sync()
            
            self.lease_client = self.robot.ensure_client('lease')
            self.lease_wallet = self.lease_client.take()
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            print("--- SDK Connected: Body Cam Active ---")
        except Exception as e:
            print(f"Connection Failed: {e}"); sys.exit(1)

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', self.img_callback, qos)

    def img_callback(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        self.body_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def force_stand(self):
        print("[ACTION] Powering on motors and standing...")
        state = self.state_client.get_robot_state()
        if state.power_state.motor_power_state != robot_state_pb2.PowerState.STATE_ON:
            self.robot.power_on()
            time.sleep(2)
        
        self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(3) 
        self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
        self.is_standing = True
        print("[STATE] Robot should now be standing.")

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.is_standing:
                threading.Thread(target=self.force_stand).start()
            else:
                threading.Thread(target=self.execute_poke).start()

    def execute_poke(self):
        print("[ACTION] Executing Snappy Poke...")
        try:
            # Distance (meters) and Duration (seconds)
            dist = 0.40
            duration = 0.6 
            
            # Manual Trajectory Construction
            traj = trajectory_pb2.SE3Trajectory()
            # Point 1: Current position (time 0)
            p1 = traj.points.add()
            p1.time_since_reference.seconds = 0
            p1.time_since_reference.nanos = 0
            p1.pose.position.x = 0.0
            p1.pose.position.y = 0.0
            p1.pose.position.z = 0.0
            p1.pose.rotation.w = 1.0

            # Point 2: Target forward position
            p2 = traj.points.add()
            p2.time_since_reference.seconds = 0
            p2.time_since_reference.nanos = int(duration * 1e9)
            p2.pose.position.x = dist
            p2.pose.position.y = 0.0
            p2.pose.position.z = 0.0
            p2.pose.rotation.w = 1.0

            arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
                root_frame_name=HAND_FRAME_NAME,
                pose_trajectory_in_task=traj
            )
            
            poke_cmd = robot_command_pb2.RobotCommand()
            poke_cmd.synchronized_command.arm_command.arm_cartesian_command.CopyFrom(arm_cartesian_command)
            
            self.command_client.robot_command(poke_cmd)
            time.sleep(duration + 0.5)

            # Retract to ready
            self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
            print("[SUCCESS] Snappy Poke complete.")
        except Exception as e:
            print(f"Poke movement failed: {e}")

    def shutdown(self):
        print("\n--- Mission End ---")
        try:
            self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_command())
            time.sleep(0.5)
            self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            time.sleep(1.0)
            self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            time.sleep(1.5)
            self.robot.power_off()
        except: pass

def main():
    rclpy.init()
    node = SpotPokeBodyCam()
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    cv2.namedWindow("BODY_CAM_VIEW")
    cv2.setMouseCallback("BODY_CAM_VIEW", node.on_click)

    try:
        while rclpy.ok():
            if node.body_frame is not None:
                status = "READY" if node.is_standing else "CLICK TO STAND"
                color = (0, 255, 0) if node.is_standing else (0, 0, 255)
                cv2.putText(node.body_frame, status, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.imshow("BODY_CAM_VIEW", node.body_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        node.shutdown()
        cv2.destroyAllWindows()
        os._exit(0)

if __name__ == '__main__':
    main()
