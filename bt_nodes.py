from ultralytics import YOLO
import py_trees
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import time
import numpy as np
import cv2
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.api import manipulation_api_pb2
from bosdyn.api import geometry_pb2

ros_node = None

class SetupLocalization(py_trees.behaviour.Behaviour):
    def __init__(self, name="SetupLocalization"):
        super().__init__(name)
        self.blackboard = self.attach_blackboard_client(name=name)
        self.blackboard.register_key(
            key="fiducials_detected",
            access=py_trees.common.Access.WRITE
        )
        self.blackboard.register_key(
            key="robot_pose",
            access=py_trees.common.Access.WRITE
        )

    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or kwargs.get('node')
        self.logger.info("[1] SetupLocalization ready")
        return True

    def update(self):
        self.logger.info("[1] Detecting fiducials...")
        self.blackboard.fiducials_detected = True
        
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = 1.0
        pose.pose.position.y = 0.5
        pose.pose.orientation.w = 1.0
        self.blackboard.robot_pose = pose
        
        return py_trees.common.Status.SUCCESS

class DetectNoodleWithYOLO(py_trees.behaviour.Behaviour):
   
    def __init__(self, name="DetectNoodleWithYOLO", model_path="best.pt"):
        super().__init__(name)
        self.model_path = model_path
        self.model = None
        self.latest_image = None
        self.bridge = None
        
    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or kwargs.get('node')
        
        # Load finetuned YOLO model
        self.logger.info(f"📦 Loading finetuned YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.model.to('cpu')
        
        # Subscribe to Spot's camera
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
        self.bridge = CvBridge()
        
        self.subscription = self.node.create_subscription(
            Image,
            '/spot/camera/hand_color/image',  # Spot's hand camera
            self._camera_callback,
            10
        )
        
        self.logger.info("✅ YOLO loaded for swimming noodle detection")
        return True
    
    def _camera_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.logger.error(f"Image conversion failed: {e}")
    
    def update(self):
        self.logger.info("🔍 Detecting swimming noodle...")
        
        if self.latest_image is None:
            self.logger.warning("⏳ Waiting for camera image...")
            return py_trees.common.Status.RUNNING
        
        img = self.latest_image.copy()
        
        # Run YOLO detection
        results = self.model(img, verbose=False)
        
        # Find swimming noodle with highest confidence
        best_detection = None
        best_confidence = 0.0
        
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                confidence = float(box.conf)
                
                if cls_name == 'swimming_noodle' and confidence > best_confidence:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    best_detection = {
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'center': [center_x, center_y],
                        'confidence': confidence,
                        'image': img
                    }
                    best_confidence = confidence
        
        if best_detection:
            # Store detection in blackboard
            blackboard = py_trees.blackboard.Client(name=self.name)
            blackboard.set("noodle_detection", best_detection, overwrite=True)
            
            self.logger.info(f"✅ Found swimming noodle! Confidence: {best_confidence:.2f}")
            self.logger.info(f"   Center: ({best_detection['center'][0]:.0f}, {best_detection['center'][1]:.0f})")
            
            # Save visualization
            x1, y1, x2, y2 = best_detection['bbox']
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            cv2.circle(img, (int(best_detection['center'][0]), int(best_detection['center'][1])), 5, (0, 0, 255), -1)
            cv2.imwrite('/tmp/noodle_detected.jpg', img)
            
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.warning("⚠️ No swimming noodle detected")
            return py_trees.common.Status.FAILURE


class GraspNoodleWithSpot(py_trees.behaviour.Behaviour):
    """
    Grasp swimming noodle using Spot's arm
    Uses detection from blackboard and Spot SDK arm_grasp
    """
    
    def __init__(self, name="GraspNoodleWithSpot", 
                 spot_ip="192.168.80.3",  # Default Spot IP
                 username="user",
                 password="password"):
        super().__init__(name)
        self.spot_ip = spot_ip
        self.username = username
        self.password = password
        self.robot = None
        self.manipulation_client = None
        
    def setup(self, **kwargs):
        self.logger.info("🤖 Connecting to Spot...")
        
        try:
            # Create Spot SDK robot instance
            sdk = bosdyn.client.create_standard_sdk('GraspNoodleBT')
            self.robot = sdk.create_robot(self.spot_ip)
            
            # Authenticate
            bosdyn.client.util.authenticate(self.robot)
            
            # Time sync
            self.robot.time_sync.wait_for_sync()
            
            # Create manipulation API client
            self.manipulation_client = self.robot.ensure_client(
                ManipulationApiClient.default_service_name
            )
            
            self.logger.info("✅ Connected to Spot successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to Spot: {e}")
            return False
    
    def update(self):
        self.logger.info("🦾 Attempting to grasp swimming noodle...")
        
        # Get detection from blackboard
        blackboard = py_trees.blackboard.Client(name=self.name)
        detection = blackboard.get("noodle_detection")
        
        if not detection:
            self.logger.error("❌ No noodle detection found in blackboard")
            return py_trees.common.Status.FAILURE
        
        try:
            # Get grasp point (center of detected noodle)
            grasp_x, grasp_y = detection['center']
            
            self.logger.info(f"🎯 Grasping at pixel: ({grasp_x:.0f}, {grasp_y:.0f})")
            
            # Build manipulation request
            # This mimics arm_grasp.py logic
            manipulation_request = self._build_grasp_request(
                grasp_x, grasp_y, detection['image']
            )
            
            # Execute grasp
            response = self.manipulation_client.manipulation_api_command(
                manipulation_request
            )
            
            # Wait for grasp to complete
            self._wait_for_grasp_completion(response.manipulation_cmd_id)
            
            self.logger.info("✅ Grasp completed successfully!")
            return py_trees.common.Status.SUCCESS
            
        except Exception as e:
            self.logger.error(f"❌ Grasp failed: {e}")
            return py_trees.common.Status.FAILURE
    
    def _build_grasp_request(self, pixel_x, pixel_y, image):
        """
        Build manipulation API request for grasping
        Based on arm_grasp.py example
        """
        # Convert pixel coordinates to pick point
        # This is simplified - see arm_grasp.py for full implementation
        
        pick_vec = geometry_pb2.Vec2(x=pixel_x, y=pixel_y)
        
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=None,  # Use latest
            frame_name_image_sensor='hand_color_image_sensor',
            camera_model=None  # Use latest
        )
        
        # Wrap in manipulation request
        request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp
        )
        
        return request
    
    def _wait_for_grasp_completion(self, cmd_id, timeout=10.0):
        """Wait for grasp command to complete"""
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            feedback = self.manipulation_client.manipulation_api_feedback_command(cmd_id)
            
            if feedback.current_state == manipulation_api_pb2.MANIP_STATE_DONE:
                self.logger.info("✓ Grasp execution complete")
                return True
            elif feedback.current_state == manipulation_api_pb2.MANIP_STATE_FAILED:
                self.logger.error("✗ Grasp execution failed")
                return False
            
            time.sleep(0.1)
        
        self.logger.warning("⏱ Grasp timeout")
        return False
    
    def terminate(self, new_status):
        """Cleanup Spot connection"""
        if self.robot:
            self.logger.info("Disconnecting from Spot")
            # Spot SDK handles cleanup automatically


# Example usage in behavior tree
def create_noodle_grasping_tree(spot_ip, spot_user, spot_pass):
    """
    Complete behavior tree: Detect noodle → Grasp noodle
    """
    root = py_trees.composites.Sequence("NoodleGraspingProtocol", memory=True)
    
    root.add_children([
        DetectNoodleWithYOLO(
            name="DetectNoodle",
            model_path="noodle_training/yolo11n_noodle/weights/best.pt"
        ),
        GraspNoodleWithSpot(
            name="GraspNoodle",
            spot_ip=spot_ip,
            username=spot_user,
            password=spot_pass
        )
    ])
    
    return root

class DetectPile(py_trees.behaviour.Behaviour):
    def __init__(self, name="DetectPile"):
        super().__init__(name)
        self.model = None
        self.bridge = None
        self.latest_image = None
        
    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or kwargs.get('node')
        
        self.logger.info("📦 Loading YOLO model...")
        self.model = YOLO("yolo11n.pt")
        
        # IMPORTANT: Force CPU mode (GPU has compatibility issues)
        import torch
        device = 'cpu'
        self.model.to(device)
        self.logger.info(f"🖥️ Using device: {device}")
        
        self.bridge = CvBridge()
        
        # Subscribe to camera topic
        self.subscription = self.node.create_subscription(
            Image,
            '/camera/image_raw',
            self._camera_callback,
            10
        )
        self.logger.info("✅ YOLO loaded, subscribed to /camera/image_raw")
        return True
    
    def _camera_callback(self, msg):
        """Store latest camera image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.logger.error(f"Image conversion failed: {e}")
    
    def update(self):
        self.logger.info("[2] Detecting piles with YOLO...")
        
        # Check for camera image
        if self.latest_image is None:
            self.logger.warning("⏳ Waiting for camera image...")
            return py_trees.common.Status.RUNNING
        
        img = self.latest_image.copy()
        
        # Run YOLO detection
        results = self.model(img, verbose=False)
        
        # Process detections
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_name = r.names[int(box.cls)]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf)
                
                box_dict = {
                    'x': float(x1),
                    'y': float(y1),
                    'width': float(x2 - x1),
                    'height': float(y2 - y1),
                    'confidence': confidence,
                    'class': cls_name
                }
                boxes.append(box_dict)
                
                # Draw visualization
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(img, f"{cls_name} {confidence:.2f}", (int(x1), int(y1)-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save visualization
        import os
        os.makedirs("/tmp/bt_detections", exist_ok=True)
        cv2.imwrite("/tmp/bt_detections/detected.jpg", img)
        
        # Store in blackboard
        blackboard = py_trees.blackboard.Client(name=self.name)
        blackboard.set("pile_boxes", boxes, overwrite=True)
        
        if boxes:
            self.logger.info(f"✅ [2] Detected {len(boxes)} objects")
            for i, box in enumerate(boxes):
                self.logger.info(f"   {i+1}. {box['class']} ({box['confidence']:.2f})")
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.warning("⚠️ [2] No objects detected")
            return py_trees.common.Status.FAILURE

# FIXED: All classes now use consistent blackboard access
class SegmentProbingArea(py_trees.behaviour.Behaviour):
    def __init__(self, name="SegmentProbingArea"):  
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        boxes = blackboard.get("pile_boxes")
        if not boxes:
            return py_trees.common.Status.FAILURE
        
        points = []
        for box in boxes:
            p = Point(x=box['x'] + box['width'] / 2.0, 
                     y=box['y'] + box['height'] / 2.0, z=0.0)
            points.append(p)
        
        blackboard.set("probing_points", points, overwrite=True)
        self.logger.info(f"[3] Generated {len(points)} probing points")
        return py_trees.common.Status.SUCCESS

class SetCurrentProbingPoint(py_trees.behaviour.Behaviour):
    def __init__(self, name="SetCurrentProbingPoint"):
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        points = blackboard.get("probing_points")
        if points and len(points) > 0:
            blackboard.set("current_probing_point", points[0], overwrite=True)
            self.logger.info(f"[4.0] Set current probing point: {points[0]}")
            return py_trees.common.Status.SUCCESS
        else:
            self.logger.warning("[4.0] No probing points available")
            return py_trees.common.Status.FAILURE

class ApproachProbingPoint(py_trees.behaviour.Behaviour):
    def __init__(self, name="ApproachProbingPoint"):
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        point = blackboard.get("current_probing_point")
        if not point:
            self.logger.warning("[4.1] No probing point set")
            return py_trees.common.Status.FAILURE
        
        self.logger.info(f"[4.1] Moving to probing point ({point.x:.1f}, {point.y:.1f})")
        import time
        time.sleep(0.8)
        return py_trees.common.Status.SUCCESS

# FIXED: Added setup() and consistent blackboard access
class ExecuteProbe(py_trees.behaviour.Behaviour):
    def __init__(self, name="ExecuteProbe"):
        super().__init__(name)
    
    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or kwargs.get('node')
        return True
    
    def update(self):
        self.logger.info("[4.2] Probing with noodle...")
        time.sleep(1.0)
        img = Image()
        img.header.frame_id = "noodle_tip"
        img.header.stamp = self.node.get_clock().now().to_msg()
        img.height = 480
        img.width = 640
        img.encoding = "bgr8"
        img.step = 640 * 3
        img.data = np.full((480, 640, 3), 100, dtype=np.uint8).tobytes()
        
        blackboard = py_trees.blackboard.Client(name=self.name)
        blackboard.set("tactile_data", img, overwrite=True)
        return py_trees.common.Status.SUCCESS

class RetractSafely(py_trees.behaviour.Behaviour):
    def __init__(self, name="RetractSafely"):
        super().__init__(name)
    
    def update(self):
        self.logger.info("[4.3] Retracting safely...")
        time.sleep(0.6)
        return py_trees.common.Status.SUCCESS

# FIXED: Consistent blackboard access
class SelectFallbackPoint(py_trees.behaviour.Behaviour):
    def __init__(self, name="SelectFallbackPoint"):
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        points = blackboard.get("probing_points")
        if not points or len(points) < 2:
            self.logger.warning("[4.F] No fallback point!")
            return py_trees.common.Status.FAILURE
        
        blackboard.set("current_probing_point", points[1], overwrite=True)
        self.logger.info("[4.F] Using fallback point")
        return py_trees.common.Status.SUCCESS

# FIXED: Consistent blackboard access
class AnalyzeDeformation(py_trees.behaviour.Behaviour):
    def __init__(self, name="AnalyzeDeformation"):
        super().__init__(name)
    
    def update(self):
        self.logger.info("[5] Analyzing deformation...")
        blackboard = py_trees.blackboard.Client(name=self.name)
        blackboard.set("deformation_metrics", 0.07, overwrite=True)
        return py_trees.common.Status.SUCCESS

# FIXED: Consistent blackboard access
class CheckStability(py_trees.behaviour.Behaviour):
    def __init__(self, name="CheckStability"):
        super().__init__(name)
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        m = blackboard.get("deformation_metrics")
        if m is None:
            m = 0.0
        threshold = 0.05
        stable = m < threshold
        self.logger.info(f"[6] {'STABLE' if stable else 'UNSTABLE'} ({m:.2f} < {threshold})")
        return py_trees.common.Status.SUCCESS if stable else py_trees.common.Status.FAILURE

# FIXED: Consistent blackboard access
class AlertOperator(py_trees.behaviour.Behaviour):
    def __init__(self, name="AlertOperator"):
        super().__init__(name)
        self.pub = None
    
    def setup(self, **kwargs):
        global ros_node
        self.node = ros_node or kwargs.get('node')
        if self.node:
            self.pub = self.node.create_publisher(String, '/alert', 10)
            self.logger.info("✅ AlertOperator ready")
        return True
    
    def update(self):
        blackboard = py_trees.blackboard.Client(name=self.name)
        m = blackboard.get("deformation_metrics")
        if m is None:
            m = 0.0
        msg = String(data=f"UNSTABLE PILE! Deformation: {m*100:.1f}%")
        if self.pub:
            self.pub.publish(msg)
        self.logger.error(f"🚨 ALERT: {msg.data}")
        return py_trees.common.Status.SUCCESS

# FIXED: Consistent blackboard access
class RecordData(py_trees.behaviour.Behaviour):
    def __init__(self, name="RecordData"):
        super().__init__(name)
    
    def update(self):
        self.logger.info("[7.2] Saving tactile data + video...")
        blackboard = py_trees.blackboard.Client(name=self.name)
        blackboard.set("video_log", "/home/abhishek/pile_logs/session_001.mp4", overwrite=True)
        return py_trees.common.Status.SUCCESS

class ResetSystem(py_trees.behaviour.Behaviour):
    def __init__(self, name="ResetSystem"):
        super().__init__(name)
    
    def update(self):
        self.logger.info("[8] System reset")
        time.sleep(0.4)
        return py_trees.common.Status.SUCCESS