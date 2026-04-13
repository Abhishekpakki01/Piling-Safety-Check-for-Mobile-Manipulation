import os, cv2, numpy as np, time, threading, sys
import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.math_helpers import SE3Pose

class NoodleTruthv10:
    def __init__(self):
        # Using environment variables for safety as discussed
        self.ip = os.getenv('SPOT_IP', '10.0.0.30')
        self.username = os.getenv('SPOT_USERNAME', 'rllab')
        self.password = os.getenv('SPOT_PASSWORD', 'robotlearninglab')
        self.target_odom = None
        self.is_standing = False

        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodleTruth_v10')
            self.robot = self.sdk.create_robot(self.ip)
            self.robot.authenticate(self.username, self.password)
            self.robot.time_sync.wait_for_sync()

            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.lease_client = self.robot.ensure_client('lease')
            
            self.lease_client.take()
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(self.lease_client, must_acquire=True)
            print("[SUCCESS] v10 Truth Mode Active. Ready for Wednesday's final check.")
        except Exception as e:
            print(f"[FATAL] Connection failed: {e}"); sys.exit(1)

    def stand_up(self):
        """Forces the robot to stand to get better IR sensor angles."""
        print("[ACTION] Standing up to high-visibility pose...")
        self.robot.power_on()
        self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(6) # Give it extra time to stop swaying
        self.is_standing = True
        print("[READY] Robot stable. Click the noodle now.")

    def get_stable_xyz(self, px_x, px_y):
        """Aggressive 25th percentile sampling to ignore floor-bleed at long distances."""
        samples = []
        for _ in range(20):
            try:
                res = self.image_client.get_image_from_sources(['frontleft_fisheye_image', 'frontleft_depth_in_visual_frame'])
                depth_img = np.frombuffer(res[1].shot.image.data, dtype=np.uint16).reshape(res[1].shot.image.rows, res[1].shot.image.cols)
                
                # Using a 30px patch since the object is further away (153cm)
                patch = depth_img[max(0, px_y-15):min(depth_img.shape[0], px_y+15), 
                                  max(0, px_x-15):min(depth_img.shape[1], px_x+15)]
                
                # Expand gate to 2.0m since you mentioned 153cm
                valid = patch[(patch > 300) & (patch < 2000)]
                if len(valid) > 0:
                    # 25th percentile hits the 'front' of the noodle
                    z_cam = np.percentile(valid, 25) / 1000.0
                    intr = res[0].source.pinhole.intrinsics
                    x_c = (px_x - intr.principal_point.x) * z_cam / intr.focal_length.x
                    y_c = (px_y - intr.principal_point.y) * z_cam / intr.focal_length.y
                    
                    tform = res[0].shot.transforms_snapshot.child_to_parent_edge_map[res[0].shot.frame_name_image_sensor].parent_tform_child
                    sensor_to_odom = SE3Pose.from_proto(tform)
                    samples.append(sensor_to_odom.transform_point(x_c, y_c, z_cam))
            except: continue
            time.sleep(0.05)
        
        return np.median(samples, axis=0) if len(samples) >= 3 else None

    def handle_click(self, x, y):
        if not self.is_standing:
            print("[WARNING] Robot must be STANDING to get accurate depth at 1.5m!")
            return
        
        print("[LOCKING] Calculating 3D Vector...")
        p = self.get_stable_xyz(x, y)
        if p is not None:
            self.target_odom = p
            # Euclidean distance from robot center (Odom origin) to target
            dist = np.linalg.norm(p) 
            print("\n" + "="*45)
            print(f"WORLD COORDS: X:{p[0]:.3f} Y:{p[1]:.3f} Z:{p[2]:.3f}")
            print(f"SENSOR DISTANCE: {dist:.3f} meters")
            print(f"RULER ERROR: {abs(dist - 1.53)*100:.1f} cm offset")
            print("="*45)
            if abs(dist - 1.53) > 0.15:
                print("CAUTION: Sensor and Ruler disagree by >15cm. Do not poke.")
        else:
            print("[ABORT] Sensor returned 0 depth. Try clicking the darkest part of the noodle.")

def main():
    db = NoodleTruthv10()
    cv2.namedWindow("v10_TRUTH_CHECK")
    
    def click(e, x, y, f, p):
        if e == cv2.EVENT_LBUTTONDOWN:
            threading.Thread(target=db.handle_click, args=(x,y), daemon=True).start()
            
    cv2.setMouseCallback("v10_TRUTH_CHECK", click)

    while True:
        try:
            res = db.image_client.get_image_from_sources(['frontleft_fisheye_image'])[0]
            f = cv2.imdecode(np.frombuffer(res.shot.image.data, dtype=np.uint8), 1)
            
            # Reprojection Logic for the Visual 'Lock'
            if db.target_odom is not None:
                try:
                    snapshot = res.shot.transforms_snapshot
                    odom_to_sensor = SE3Pose.from_proto(snapshot.child_to_parent_edge_map[res.shot.frame_name_image_sensor].parent_tform_child).inverse()
                    p_cam = odom_to_sensor.transform_point(*db.target_odom)
                    intr = res.source.pinhole.intrinsics
                    px_x = int((p_cam[0] * intr.focal_length.x / p_cam[2]) + intr.principal_point.x)
                    px_y = int((p_cam[1] * intr.focal_length.y / p_cam[2]) + intr.principal_point.y)
                    
                    if 0 <= px_x < f.shape[1] and 0 <= px_y < f.shape[0]:
                        cv2.circle(f, (px_x, px_y), 15, (0, 0, 255), 3) # Red for Truth
                        cv2.putText(f, "LOCKED", (px_x+20, px_y), 1, 1, (0,0,255), 2)
                except: pass

            cv2.putText(f, "1. Press 'S' to Stand | 2. Click Noodle", (20, 40), 1, 1.2, (0,255,0), 2)
            cv2.imshow("v10_TRUTH_CHECK", f)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'): threading.Thread(target=db.stand_up, daemon=True).start()
            if key == ord('q'): break
        except Exception: continue

    # Shutdown
    if db.is_standing:
        db.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2); db.robot.power_off()

if __name__ == '__main__': main()
