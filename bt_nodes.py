from ultralytics import YOLO
import py_trees
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from std_msgs.msg import String
import time
import numpy as np
import ultralytics
import cv2


ros_node = None

class SetupLocalization(py_trees.behaviour.Behaviour):
    def __init__(self, name="SetupLocalization"):
        super().__init__(name)

    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or rclpy.create_node('bt_node')
        self.logger.info("[1] SetupLocalization ready")

    def update(self):
        self.logger.info("[1] Detecting fiducials...")
        self.blackboard.set("fiducials_detected", True)
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.5
        pose.pose.orientation.w = 1.0
        self.blackboard.set("robot_pose", pose)
        return py_trees.common.Status.SUCCESS

class DetectPile(py_trees.behaviour.Behaviour):
    def __init__(self, name="DetectPile"):
        super().__init__(name)
        # Load YOLO model (use 'yolo11n.pt' or your custom 'best.pt' for 'box' class)
        self.model = YOLO("yolo11n.pt")

    def update(self):
        self.logger.info("[2] Detecting pile with YOLO vision...")
        
        # Placeholder: Load image from file (replace with ROS2 subscriber for camera feed)
        image_path = "path/to/your/camera_image.jpg"  # e.g., from /camera/image_raw
        img = cv2.imread(image_path)
        if img is None:
            self.logger.error("[2] Failed to load image!")
            return py_trees.common.Status.FAILURE

        # Run YOLO inference
        results = self.model(img)

        boxes = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                if cls_name == 'box':  # Filter for 'box' class; adjust as needed
                    box_dict = {
                        'x': float(box.xyxy[0][0]),
                        'y': float(box.xyxy[0][1]),
                        'width': float(box.xyxy[0][2] - box.xyxy[0][0]),
                        'height': float(box.xyxy[0][3] - box.xyxy[0][1])
                    }
                    boxes.append(box_dict)

                    # Visualize: Draw box on image
                    cv2.rectangle(img, (int(rect.x), int(rect.y)), 
                                  (int(rect.x + rect.width), int(rect.y + rect.height)), 
                                  (0, 255, 0), 2)
                    cv2.putText(img, f"{cls_name} {box.conf:.2f}", (int(rect.x), int(rect.y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save visualized image
        cv2.imwrite("detected_piles.jpg", img)
        self.logger.info("[2] Visualized and saved 'detected_piles.jpg'")

        self.blackboard.set("pile_boxes", boxes)
        self.logger.info(f"[2] Detected {len(boxes)} piles")
        return py_trees.common.Status.SUCCESS if boxes else py_trees.common.Status.FAILURE

class SegmentProbingArea(py_trees.behaviour.Behaviour):
    def update(self):
        boxes = self.blackboard.get("pile_boxes")
        if not boxes: return py_trees.common.Status.FAILURE
        points = []
        for box in boxes:
            p = Point(x=box['x'] + box['width'] / 2.0, y=box['y'] + box['height'] / 2.0, z=0.0)
            points.append(p)
        self.blackboard.set("probing_points", points)
        self.logger.info(f"[3] Generated {len(points)} probing points")
        return py_trees.common.Status.SUCCESS

class ApproachProbingPoint(py_trees.behaviour.Behaviour):
    def update(self):
        point = self.blackboard.get("current_probing_point")
        if not point: return py_trees.common.Status.FAILURE
        self.logger.info(f"[4.1] Moving to ({point.x:.1f}, {point.y:.1f})")
        time.sleep(0.8)
        return py_trees.common.Status.SUCCESS

class ExecuteProbe(py_trees.behaviour.Behaviour):
    def update(self):
        self.logger.info("[4.2] Probing with noodle...")
        time.sleep(1.0)
        img = Image()
        img.header.frame_id = "noodle_tip"
        img.header.stamp = self.node.get_clock().now().to_msg()
        img.height = 480; img.width = 640; img.encoding = "bgr8"
        img.step = 640 * 3
        img.data = np.full((480, 640, 3), 100, dtype=np.uint8).tobytes()
        self.blackboard.set("tactile_data", img)
        return py_trees.common.Status.SUCCESS

class RetractSafely(py_trees.behaviour.Behaviour):
    def update(self):
        self.logger.info("[4.3] Retracting safely...")
        time.sleep(0.6)
        return py_trees.common.Status.SUCCESS

class SelectFallbackPoint(py_trees.behaviour.Behaviour):
    def update(self):
        points = self.blackboard.get("probing_points")
        if not points or len(points) < 2:
            self.logger.warning("[4.F] No fallback point!")
            return py_trees.common.Status.FAILURE
        self.blackboard.set("current_probing_point", points[1])
        self.logger.info("[4.F] Using fallback point")
        return py_trees.common.Status.SUCCESS

class AnalyzeDeformation(py_trees.behaviour.Behaviour):
    def update(self):
        self.logger.info("[5] Analyzing deformation...")
        self.blackboard.set("deformation_metrics", 0.07)
        return py_trees.common.Status.SUCCESS

class CheckStability(py_trees.behaviour.Behaviour):
    def update(self):
        m = self.blackboard.get("deformation_metrics", 0.0)
        threshold = 0.05
        stable = m < threshold
        self.logger.info(f"[6] {'STABLE' if stable else 'UNSTABLE'} ({m:.2f} < {threshold})")
        return py_trees.common.Status.SUCCESS if stable else py_trees.common.Status.FAILURE

class AlertOperator(py_trees.behaviour.Behaviour):
    def setup(self, **kwargs):
        self.pub = self.node.create_publisher(String, '/alert', 10)
    def update(self):
        m = self.blackboard.get("deformation_metrics", 0.0)
        msg = String(data=f"UNSTABLE PILE! Deformation: {m*100:.1f}%")
        self.pub.publish(msg)
        self.logger.error(f"ALERT: {msg.data}")
        return py_trees.common.Status.SUCCESS

class RecordData(py_trees.behaviour.Behaviour):
    def update(self):
        self.logger.info("[7.2] Saving tactile data + video...")
        self.blackboard.set("video_log", "/home/abhishek/pile_logs/session_001.mp4")
        return py_trees.common.Status.SUCCESS

class ResetSystem(py_trees.behaviour.Behaviour):
    def update(self):
        self.logger.info("[8] System reset")
        time.sleep(0.4)
        return py_trees.common.Status.SUCCESS
