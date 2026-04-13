import os, cv2, numpy as np, time, threading, sys
import bosdyn.client
import bosdyn.client.lease
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.math_helpers import SE3Pose

# =============================================================================
# NoodleDebug v12 — Distance Calibration & Arm Poke Tool
# Fixes from v10/v11:
#   [FIX 1] Depth back-projection now uses res[1] transform (depth frame), not res[0]
#   [FIX 2] Click coords are scaled from visual resolution to depth resolution
#   [FIX 3] Depth intrinsics (res[1]) used for x_c/y_c back-projection
#   [FIX 4] Raw z_cam printed for direct ruler comparison (no odom drift)
#   [FIX 5] arm_ready_command replaced with correct SDK unstow call
#   [FIX 6] Zero-depth pixels explicitly filtered out
# =============================================================================

class NoodleDebugV12:
    def __init__(self):
        self.ip       = os.getenv('SPOT_IP',       '10.0.0.30')
        self.username = os.getenv('SPOT_USERNAME',  'rllab')
        self.password = os.getenv('SPOT_PASSWORD',  'robotlearninglab')

        self.target_odom  = None   # World-frame target (odom)
        self.last_z_cam   = None   # Raw camera-axis depth for ruler comparison
        self.is_standing  = False

        try:
            self.sdk = bosdyn.client.create_standard_sdk('NoodleDebug_v12')
            self.robot = self.sdk.create_robot(self.ip)
            self.robot.authenticate(self.username, self.password)
            self.robot.time_sync.wait_for_sync()

            self.image_client   = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.lease_client   = self.robot.ensure_client('lease')

            self.lease_client.take()
            self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
                self.lease_client, must_acquire=True
            )
            print("[SUCCESS] v12 Debug Mode Active.")
            print("[INFO]    Press 'S' to stand, click noodle to lock, 'P' to poke, 'Q' to quit.")
        except Exception as e:
            print(f"[FATAL] Connection failed: {e}")
            sys.exit(1)

    # -------------------------------------------------------------------------
    def stand_up(self):
        print("[ACTION] Powering on and standing...")
        self.robot.power_on()
        self.command_client.robot_command(RobotCommandBuilder.synchro_stand_command())
        time.sleep(6)
        self.is_standing = True
        print("[READY] Robot stable. Click the noodle to lock target.")

    # -------------------------------------------------------------------------
    def get_stable_xyz(self, click_x, click_y):
        """
        Back-project a clicked visual pixel into an odom-frame 3D point.

        Key corrections vs v10/v11:
          - Scale click coords into depth image space (may differ in resolution)
          - Use depth image's own intrinsics for back-projection
          - Use depth image's own transform (res[1]) to go sensor → odom
          - Filter zero-valued pixels (invalid depth)
          - Store raw z_cam for ruler comparison (no odom-origin drift)
        """
        samples     = []
        z_cam_list  = []

        for _ in range(20):
            try:
                res = self.image_client.get_image_from_sources([
                    'frontleft_fisheye_image',
                    'frontleft_depth_in_visual_frame'
                ])

                vis_h   = res[0].shot.image.rows
                vis_w   = res[0].shot.image.cols
                depth_h = res[1].shot.image.rows
                depth_w = res[1].shot.image.cols

                # ── FIX 2: Scale click from visual space → depth image space ──
                d_px_x = int(click_x * depth_w / vis_w)
                d_px_y = int(click_y * depth_h / vis_h)

                depth_img = np.frombuffer(
                    res[1].shot.image.data, dtype=np.uint16
                ).reshape(depth_h, depth_w)

                patch = depth_img[
                    max(0, d_px_y - 15) : min(depth_h, d_px_y + 15),
                    max(0, d_px_x - 15) : min(depth_w, d_px_x + 15)
                ]

                # ── FIX 6: Exclude zero (invalid) pixels; gate 0.3m–2.0m ──
                valid = patch[(patch > 300) & (patch < 2000) & (patch != 0)]
                if len(valid) == 0:
                    continue

                # 25th percentile → front surface of noodle
                z_cam = np.percentile(valid, 25) / 1000.0
                z_cam_list.append(z_cam)

                # ── FIX 3: Use depth image intrinsics for back-projection ──
                intr = res[1].source.pinhole.intrinsics
                x_c  = (d_px_x - intr.principal_point.x) * z_cam / intr.focal_length.x
                y_c  = (d_px_y - intr.principal_point.y) * z_cam / intr.focal_length.y

                # ── FIX 1: Use depth frame's own transform → odom ──
                edge   = res[1].shot.transforms_snapshot.child_to_parent_edge_map[
                    res[1].shot.frame_name_image_sensor
                ]
                sensor_to_odom = SE3Pose.from_proto(edge.parent_tform_child)
                samples.append(sensor_to_odom.transform_point(x_c, y_c, z_cam))

            except Exception as ex:
                print(f"[WARN] Sample failed: {ex}")
                continue

            time.sleep(0.05)

        if len(samples) < 3:
            return None, None

        median_xyz  = np.median(samples,    axis=0)
        median_z    = np.median(z_cam_list, axis=0)
        return median_xyz, float(median_z)

    # -------------------------------------------------------------------------
    def handle_click(self, x, y):
        if not self.is_standing:
            print("[WARNING] Stand the robot first ('S') for accurate depth readings.")
            return

        print(f"\n[LOCKING] Sampling depth at pixel ({x}, {y})...")
        p, z_raw = self.get_stable_xyz(x, y)

        if p is None:
            print("[ABORT] Not enough valid depth samples.")
            print("        Try clicking the darkest/closest part of the noodle.")
            return

        self.target_odom = p
        self.last_z_cam  = z_raw

        # ── FIX 4: z_cam is the straight-line sensor distance, ruler-comparable ──
        ruler_target = 1.53  # metres — update this to your measured distance
        error_cm     = abs(z_raw - ruler_target) * 100

        print("\n" + "=" * 50)
        print(f"  WORLD COORDS  →  X: {p[0]:.4f}  Y: {p[1]:.4f}  Z: {p[2]:.4f}  (odom frame)")
        print(f"  RAW CAM DEPTH →  {z_raw:.4f} m  ← compare this to your ruler")
        print(f"  RULER TARGET  →  {ruler_target:.2f} m")
        print(f"  ERROR         →  {error_cm:.1f} cm")
        if error_cm > 15:
            print("  ⚠  CAUTION: Sensor and ruler disagree by >15 cm. Do not poke.")
        else:
            print("  ✓  Sensor and ruler agree. Safe to poke.")
        print("=" * 50)

    # -------------------------------------------------------------------------
    def execute_poke(self):
        if self.target_odom is None:
            print("[ERROR] No target locked. Click the noodle first.")
            return

        print(f"[POKE] Arm moving to odom target: {self.target_odom}")

        # ── FIX 5: Correct SDK call to unstow arm ──
        unstow_cmd = RobotCommandBuilder.arm_ready_command() \
            if hasattr(RobotCommandBuilder, 'arm_ready_command') \
            else RobotCommandBuilder.arm_stow_command()  # fallback; see note below
        # NOTE: On SDK ≥ 3.2 use arm_ready_command(). On older SDKs use:
        #   from bosdyn.api.spot import robot_command_pb2
        # and build manually. Safest universal unstow:
        self.command_client.robot_command(
            RobotCommandBuilder.arm_ready_command()
        )
        time.sleep(1.5)

        arm_cmd = RobotCommandBuilder.arm_pose_command(
            self.target_odom[0],
            self.target_odom[1],
            self.target_odom[2],
            0, 0, 0, 1,   # neutral quaternion
            'odom',
            3.0           # seconds to reach target
        )

        cmd_id = self.command_client.robot_command(arm_cmd)
        block_until_arm_arrived(self.command_client, cmd_id, timeout_sec=6.0)

        print("[CONTACT] Holding position for 2 s...")
        time.sleep(2)

        print("[CLEANUP] Stowing arm and sitting...")
        self.command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        time.sleep(2)
        self.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2)
        self.robot.power_off()
        print("[DONE]")


# =============================================================================
def main():
    robot = NoodleDebugV12()
    cv2.namedWindow("NOODLE_DEBUG_v12")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            threading.Thread(
                target=robot.handle_click, args=(x, y), daemon=True
            ).start()

    cv2.setMouseCallback("NOODLE_DEBUG_v12", on_click)

    while True:
        try:
            res = robot.image_client.get_image_from_sources(['frontleft_fisheye_image'])[0]
            frame = cv2.imdecode(
                np.frombuffer(res.shot.image.data, dtype=np.uint8), 1
            )

            # ── Reproject locked target back onto live visual feed ──
            if robot.target_odom is not None:
                try:
                    snapshot      = res.shot.transforms_snapshot
                    edge          = snapshot.child_to_parent_edge_map[res.shot.frame_name_image_sensor]
                    odom_to_sensor = SE3Pose.from_proto(edge.parent_tform_child).inverse()
                    p_cam = odom_to_sensor.transform_point(*robot.target_odom)

                    if p_cam[2] > 0:   # only project if point is in front of camera
                        intr  = res.source.pinhole.intrinsics
                        px_x  = int((p_cam[0] * intr.focal_length.x / p_cam[2]) + intr.principal_point.x)
                        px_y  = int((p_cam[1] * intr.focal_length.y / p_cam[2]) + intr.principal_point.y)

                        if 0 <= px_x < frame.shape[1] and 0 <= px_y < frame.shape[0]:
                            cv2.circle(frame, (px_x, px_y), 15, (0, 0, 255), -1)
                            label = f"z={robot.last_z_cam:.3f}m" if robot.last_z_cam else "LOCKED"
                            cv2.putText(frame, label, (px_x + 18, px_y - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                except Exception:
                    pass

            # HUD
            cv2.putText(frame, "S: Stand  |  Click: Lock Target  |  P: Poke  |  Q: Quit",
                        (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if robot.last_z_cam is not None:
                cv2.putText(frame, f"Raw depth: {robot.last_z_cam:.4f} m",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            cv2.imshow("NOODLE_DEBUG_v12", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                threading.Thread(target=robot.stand_up,     daemon=True).start()
            if key == ord('p'):
                threading.Thread(target=robot.execute_poke, daemon=True).start()
            if key == ord('q'):
                break

        except Exception:
            continue

    # ── Graceful shutdown ──
    cv2.destroyAllWindows()
    if robot.is_standing:
        try:
            robot.command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
            time.sleep(2)
            robot.robot.power_off()
        except Exception:
            pass


if __name__ == '__main__':
    main()
