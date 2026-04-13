import os, time, rclpy, threading, cv2, sys
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage

import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.api import manipulation_api_pb2, geometry_pb2

class SpotNoodleMasterV2(Node):
    def __init__(self):
        super().__init__('spot_noodle_master_v2')
        self.model = YOLO('best.pt')
        self.has_noodle, self.is_standing, self.is_processing = False, False, False
        self.frame = None

        self.ip = os.getenv('SPOT_IP', '10.0.0.30')
        self.username = os.getenv('SPOT_USERNAME', 'rllab')
        self.password = os.getenv('SPOT_PASSWORD', 'robotlearninglab')

        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodleMasterV2')
            self.robot = self.sdk.create_robot(self.ip)
            self.robot.authenticate(self.username, self.password)
            self.robot.time_sync.wait_for_sync()

            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.manip_client = self.robot.ensure_client('manipulation')
            self.lease_client = self.robot.ensure_client('lease')
            self.lease_wallet = self.lease_client.take()
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            print("[SUCCESS] Spot Connected (Compatibility Mode).")
        except Exception as e: 
            print(f"[FATAL] Connection Failed: {e}"); sys.exit(1)

        custom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', self.img_cb, custom_qos)

    def img_cb(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            results = self.model(img, conf=0.4, verbose=False)
            self.frame = results[0].plot()
            mode = "POKE BOX" if self.has_noodle else "GRASP NOODLE"
            cv2.putText(self.frame, f"MODE: {mode}", (10, 40), 1, 1.5, (0, 255, 0), 2)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_processing:
            print(f"[CLICK] X:{x} Y:{y}")
            threading.Thread(target=self.do_action, args=(x, y), daemon=True).start()

    def do_action(self, x, y):
        self.is_processing = True
        try:
            if not self.is_standing:
                self.robot.power_on()
                self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
                time.sleep(2.0)
                self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                self.is_standing = True

            # Manual Depth Extraction
            depth_res = self.image_client.get_image_from_sources(['frontleft_depth_in_visual_frame'])[0]
            depth_buffer = np.frombuffer(depth_res.shot.image.data, dtype=np.uint16)
            depth_img = depth_buffer.reshape(depth_res.shot.image.rows, depth_res.shot.image.cols)
            ix, iy = min(max(int(x), 0), depth_img.shape[1] - 1), min(max(int(y), 0), depth_img.shape[0] - 1)
            dist = depth_img[iy, ix] / 1000.0

            print(f"[DIAGNOSTIC] Distance: {dist:.2f}m")

            # SAFETY GUARD: Because we can't use walk_params, we block clicks too far away
            if dist > 1.0 or dist < 0.1:
                print(f"[REJECTED] Distance {dist:.2f}m is too far for stationary reach. Bring the box closer (approx 0.7m).")
                return

            res = self.image_client.get_image_from_sources(['frontleft_fisheye_image'])[0]
            
            # SIMPLIFIED REQUEST: Removed walk_params for compatibility
            pick_req = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=geometry_pb2.Vec2(x=float(x), y=float(y)),
                transforms_snapshot_for_camera=res.shot.transforms_snapshot,
                frame_name_image_sensor=res.shot.frame_name_image_sensor,
                camera_model=res.source.pinhole
            )
            
            print("[ACTION] Executing manipulation command...")
            self.manip_client.manipulation_api_command(manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_req))
            time.sleep(6.0) 
            
            if not self.has_noodle: 
                self.has_noodle = True
            
            self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())

        except Exception as e: print(f"[ERROR] {e}")
        finally: self.is_processing = False

    def shutdown(self):
        print("\n[SHUTDOWN] Cleanup...")
        try:
            self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
            time.sleep(1.0)
            self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            self.robot.power_off()
        except: pass

def main():
    rclpy.init()
    node = SpotNoodleMasterV2()
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()
    cv2.namedWindow("SPOT_CONTROL")
    cv2.setMouseCallback("SPOT_CONTROL", node.on_click)
    try:
        while True:
            if node.frame is not None: cv2.imshow("SPOT_CONTROL", node.frame)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
    finally:
        node.shutdown()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
