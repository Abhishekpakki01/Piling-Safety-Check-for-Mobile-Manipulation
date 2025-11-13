#!/usr/bin/env python3
import rclpy
import py_trees
import py_trees_ros
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def create_tree():
    root = py_trees.composites.Sequence("SafetyCheckProtocol", memory=True)
    root.add_children([SetupLocalization(), DetectPile(), SegmentProbingArea()])

    probe_sel = py_trees.composites.Selector("ProbeSequence", memory=False)
    primary = py_trees.composites.Sequence("PrimaryProbe", memory=True)
    primary.add_children([
        py_trees.behaviours.SetBlackboardVariable("current_probing_point", setter=lambda bb: bb.get("probing_points")[0]),
        ApproachProbingPoint(), ExecuteProbe(), RetractSafely()
    ])
    probe_sel.add_children([primary, SelectFallbackPoint()])

    root.add_children([probe_sel, AnalyzeDeformation(),
                       py_trees.decorators.Inverter(child=CheckStability()),
                       py_trees.composites.Parallel("ResponseHandling", policy=py_trees.common.ParallelPolicy.SuccessOnAll())])
    parallel = root.children[-1]
    parallel.add_children([AlertOperator(), RecordData()])
    root.add_child(ResetSystem())
    return root

def main():
    global ros_node
    rclpy.init()
    ros_node = rclpy.create_node('safety_bt_node')

    # Initialize CvBridge and camera subscription
    bridge = CvBridge()
    def camera_callback(msg):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite("camera_image.jpg", cv_img)  # Save for DetectPile

    ros_node.create_subscription(Image, '/camera/image_raw', camera_callback, 10)

    tree = py_trees_ros.trees.BehaviourTree(root=create_tree(), unicode_tree_debug=True)
    tree.setup(node=ros_node, timeout=15.0)
    print("Piling Safety BT STARTED")
    try:
        while rclpy.ok():
            tree.tick()
            rclpy.spin_once(ros_node, timeout_sec=0.1)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        tree.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()