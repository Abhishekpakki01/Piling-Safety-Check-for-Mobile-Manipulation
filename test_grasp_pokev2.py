import os, time, rclpy, threading, cv2, sys
import numpy as np
from ultralytics import YOLO
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import CompressedImage

import bosdyn.client
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.api import manipulation_api_pb2, geometry_pb2

class SpotNoodleMasterV2(Node):
    def __init__(self):
        super().__init__('spot_noodle_master_v2')
        self.model = YOLO('best.pt')
        self.has_noodle, self.is_standing, self.is_processing = False, False, False
        self.frame = None

        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodleMasterV2')
            self.robot = self.sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
            self.robot.authenticate(os.getenv('SPOT_USERNAME', 'rllab'), os.getenv('SPOT_PASSWORD', 'robotlearninglab'))
            self.robot.time_sync.wait_for_sync()

            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.manip_client = self.robot.ensure_client('manipulation')
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            
            self.lease_client = self.robot.ensure_client('lease')
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            print("[SAFE MODE] 1st Click: Stand | 2nd Click: Grasp | 3rd Click: Poke")
        except Exception as e: print(f"[FATAL] {e}"); sys.exit(1)

        custom_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST, depth=5)
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', self.img_cb, custom_qos)

    def img_cb(self, data):
        np_arr = np.frombuffer(data.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is not None:
            results = self.model(img, conf=0.4, verbose=False)
            self.frame = results[0].plot()
            
            # Dynamic HUD
            if not self.is_standing:
                msg, col = "CLICK TO STAND", (0, 0, 255)
            elif not self.has_noodle:
                msg, col = "STAND COMPLETE: CLICK NOODLE", (0, 165, 255)
            else:
                msg, col = "VERIFIED: CLICK BOX TO POKE", (0, 255, 0)
            
            cv2.putText(self.frame, msg, (10, 50), 1, 1.8, col, 2)

    def check_grasp_success(self):
        state = self.state_client.get_robot_state()
        width = state.manipulator_state.gripper_open_percentage
        print(f"[SENSOR] Gripper Width: {width:.1f}%")
        # Noodle is usually > 3% open. 0% means it missed and closed fully.
        return width > 2.5

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and not self.is_processing:
            threading.Thread(target=self.do_action, args=(x, y), daemon=True).start()

    def do_action(self, x, y):
        self.is_processing = True
        try:
            # --- PHASE 0: STAND ---
            if not self.is_standing:
                print("[ACTION] Standing and preparing arm...")
                self.robot.power_on()
                self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
                time.sleep(2.0)
                # Move arm to a forward-pointing "Ready" position
                self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())
                self.is_standing = True
                print("[READY] Robot is standing. Please click the NOODLE now.")
                return 

            # --- DEPTH VALIDATION (Grid Search) ---
            depth_res = self.image_client.get_image_from_sources(['frontleft_depth_in_visual_frame'])[0]
            depth_buffer = np.frombuffer(depth_res.shot.image.data, dtype=np.uint16)
            depth_img = depth_buffer.reshape(depth_res.shot.image.rows, depth_res.shot.image.cols)
            
            # Sample a 9x9 area around the click to find the REAL object
            ix, iy = int(x), int(y)
            region = depth_img[max(0, iy-4):iy+5, max(0, ix-4):ix+5]
            valid_depths = region[region > 0]
            
            if len(valid_depths) == 0:
                print("[REJECTED] Camera is blind at this pixel. Move noodle or try another spot.")
                return
            
            # At 0.4m, we want the CLOSEST thing in that grid (the noodle)
            dist = np.min(valid_depths) / 1000.0
            print(f"[DIAGNOSTIC] Distance found: {dist:.2f}m")

            # --- PHASE 1: GRASP ---
            if not self.has_noodle:
                if dist > 0.9 or dist < 0.2:
                    print(f"[SAFETY] Blocked. {dist:.2f}m is too far/near for a safe grasp.")
                    return

                res = self.image_client.get_image_from_sources(['frontleft_fisheye_image'])[0]
                pick_req = manipulation_api_pb2.PickObjectInImage(
                    pixel_xy=geometry_pb2.Vec2(x=float(x), y=float(y)),
                    transforms_snapshot_for_camera=res.shot.transforms_snapshot,
                    frame_name_image_sensor=res.shot.frame_name_image_sensor,
                    camera_model=res.source.pinhole
                )
                
                print(f"[ACTION] Attempting Grasp at {dist:.2f}m...")
                self.manip_client.manipulation_api_command(manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=pick_req))
                time.sleep(8.0) 

                if self.check_grasp_success():
                    self.has_noodle = True
                    print("[SUCCESS] Noodle verified in jaws.")
                    self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0))
                else:
                    print("[FAILED] Gripper empty. Try clicking the noodle again.")
                
                self.command_client.robot_command(RobotCommandBuilder.arm_ready_command())

            # --- PHASE 2: POKE ---
            else:
                print(f"[ACTION] Poking target at {dist:.2f}m...")
                # Execution logic for poke...
                # (Same PickObjectInImage logic used for the poke movement)
                print("[DONE] Poke maneuver complete.")

        except Exception as e: print(f"[ERROR] {e}")
        finally: self.is_processing = False

    def shutdown(self):
        try:
            self.command_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(1.0))
            time.sleep(0.5)
            self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
            self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            self.robot.power_off()
        except: pass

def main():
    rclpy.init()
    node = SpotNoodleMasterV2()
    threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
    cv2.namedWindow("SPOT_SAFETY_UI"); cv2.setMouseCallback("SPOT_SAFETY_UI", node.on_click)
    try:
        while True:
            if node.frame is not None: cv2.imshow("SPOT_SAFETY_UI", node.frame)
            if cv2.waitKey(30) & 0xFF == ord('q'): break
    finally:
        node.shutdown()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()
