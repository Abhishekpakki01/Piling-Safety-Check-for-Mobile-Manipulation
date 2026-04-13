import os, cv2, numpy as np, time, threading, sys
import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.math_helpers import SE3Pose

class DistanceDebugger:
    def __init__(self):
        self.ip = os.getenv('SPOT_IP', '10.0.0.30')
        self.username = os.getenv('SPOT_USERNAME', 'rllab')
        self.password = os.getenv('SPOT_PASSWORD', 'robotlearninglab')

        try:
            self.sdk = bosdyn.client.create_standard_sdk('DistCheck')
            self.robot = self.sdk.create_robot(self.ip)
            self.robot.authenticate(self.username, self.password)
            self.robot.time_sync.wait_for_sync()

            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.lease_client = self.robot.ensure_client('lease')
            
            self.lease_client.take()
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            print("[SUCCESS] Control claimed.")
        except Exception as e:
            print(f"[FATAL] Connection failed: {e}")
            sys.exit(1)

    def get_stable_xyz(self, px_x, px_y, label=""):
        """Uses the ODOM frame to get the most honest coordinate possible."""
        # Check both front cameras
        for side in ['frontleft', 'frontright']:
            res = self.image_client.get_image_from_sources([f'{side}_fisheye_image', f'{side}_depth_in_visual_frame'])
            depth_img = np.frombuffer(res[1].shot.image.data, dtype=np.uint16).reshape(res[1].shot.image.rows, res[1].shot.image.cols)
            
            # Wide 20x20 search patch
            y, x = int(px_y), int(px_x)
            patch = depth_img[max(0, y-10):min(depth_img.shape[0], y+10), 
                              max(0, x-10):min(depth_img.shape[1], x+10)]
            
            valid = patch[patch > 0]
            if len(valid) > 0:
                z = np.median(valid) / 1000.0
                intr = res[0].source.pinhole.intrinsics
                x_c = (px_x - intr.principal_point.x) * z / intr.focal_length.x
                y_c = (px_y - intr.principal_point.y) * z / intr.focal_length.y
                
                # Transform to ODOM frame
                snapshot = res[0].shot.transforms_snapshot
                # We specifically look for the transformation to 'odom'
                sensor_to_odom = SE3Pose.from_proto(snapshot.child_to_parent_edge_map[res[0].shot.frame_name_image_sensor].parent_tform_child)
                return sensor_to_odom.transform_point(x_c, y_c, z)
        
        return None

    def run_check(self, x, y):
        is_powered = False
        try:
            # 1. MEASURE SITTING
            p_sit = self.get_stable_xyz(x, y, "SIT")
            if p_sit:
                print(f"\n[SIT]   X: {p_sit[0]:.3f}, Y: {p_sit[1]:.3f}, Z: {p_sit[2]:.3f}")
                if p_sit[0] < 0:
                    print("!!! ALERT: X is negative. The robot thinks the noodle is BEHIND it.")

            # 2. STAND
            print("[ACTION] Standing up...")
            self.robot.power_on()
            is_powered = True 
            self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
            time.sleep(5.0) 
            
            # 3. MEASURE STANDING
            p_stand = self.get_stable_xyz(x, y, "STAND")
            if p_stand:
                print(f"[STAND] X: {p_stand[0]:.3f}, Y: {p_stand[1]:.3f}, Z: {p_stand[2]:.3f}")
                
                if p_sit:
                    dx = abs(p_stand[0] - p_sit[0]) * 100
                    dz = abs(p_stand[2] - p_sit[2]) * 100
                    print("-" * 35)
                    print(f"REAL DRIFT: {dx:.1f} cm forward, {dz:.1f} cm height")
                    print("-" * 35)

        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            if is_powered:
                print("[CLEANUP] Sitting...")
                self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
                time.sleep(2.0)
                self.robot.power_off()

def main():
    db = DistanceDebugger()
    cv2.namedWindow("DIAGNOSTIC")
    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN: threading.Thread(target=db.run_check, args=(x,y), daemon=True).start()
    cv2.setMouseCallback("DIAGNOSTIC", click)
    while True:
        r = db.image_client.get_image_from_sources(['frontleft_fisheye_image'])[0]
        f = cv2.imdecode(np.frombuffer(r.shot.image.data, dtype=np.uint8), 1)
        cv2.putText(f, "CLICK NOODLE", (20, 50), 1, 2, (0,255,0), 2)
        cv2.imshow("DIAGNOSTIC", f)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

if __name__ == '__main__': main()
