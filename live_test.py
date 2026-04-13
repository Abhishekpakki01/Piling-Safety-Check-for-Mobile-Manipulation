import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from ultralytics import YOLO

class SpotNoodleDetector(Node):
    def __init__(self):
        super().__init__('spot_noodle_detector')
        self.model = YOLO('best.pt') 
        
        # Define a QoS profile that matches Spot's 'Best Effort' stream
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 1. Gripper Camera
        self.create_subscription(CompressedImage, '/spot/hand_color_image/compressed', 
                                 lambda msg: self.listener_callback(msg, "Hand_Camera"), 
                                 qos_profile)
        
        # 2. Front Left Camera
        self.create_subscription(CompressedImage, '/spot/frontleft_fisheye_image/compressed', 
                                 lambda msg: self.listener_callback(msg, "Front_Left"), 
                                 qos_profile)
        
        # 3. Front Right Camera
        self.create_subscription(CompressedImage, '/spot/frontright_fisheye_image/compressed', 
                                 lambda msg: self.listener_callback(msg, "Front_Right"), 
                                 qos_profile)

        self.get_logger().info('QoS Fixed! Now listening to Spot cameras...')

    def listener_callback(self, data, camera_name):
        np_arr = np.frombuffer(data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        results = self.model(image, conf=0.4, verbose=False)
        
        annotated_frame = results[0].plot()
        cv2.imshow(camera_name, annotated_frame)
        cv2.waitKey(1)

def main():
    rclpy.init()
    node = SpotNoodleDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
