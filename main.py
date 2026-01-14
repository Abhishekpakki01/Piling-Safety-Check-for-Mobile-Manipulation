#!/usr/bin/env python3
"""
Main entry point for Piling Safety and Noodle Grasping Behavior Trees
"""
import rclpy
import py_trees
import py_trees_ros
import time
from bt_nodes import *

# Global ROS node
ros_node = None


def create_tree():
    """
    Original piling safety check behavior tree
    """
    root = py_trees.composites.Sequence("SafetyCheckProtocol", memory=True)
    
    # Phase 1-3: Setup and Detection
    root.add_children([
        SetupLocalization(name="SetupLocalization"), 
        DetectPile(name="DetectPile"), 
        SegmentProbingArea(name="SegmentProbingArea")
    ])
    
    # Phase 4: Probing sequence with fallback
    probe_sel = py_trees.composites.Selector("ProbeSequence", memory=False)
    primary = py_trees.composites.Sequence("PrimaryProbe", memory=True)
    primary.add_children([
        SetCurrentProbingPoint(name="SetCurrentProbingPoint"),
        ApproachProbingPoint(name="ApproachProbingPoint"), 
        ExecuteProbe(name="ExecuteProbe"), 
        RetractSafely(name="RetractSafely")
    ])
    
    probe_sel.add_children([
        primary, 
        SelectFallbackPoint(name="SelectFallbackPoint")
    ])
    
    root.add_child(probe_sel)
    
    # Phase 5-6: Analysis
    root.add_children([
        AnalyzeDeformation(name="AnalyzeDeformation"),
        py_trees.decorators.Inverter(
            name="InvertStability", 
            child=CheckStability(name="CheckStability")
        )
    ])
    
    # Phase 7: Parallel response
    parallel = py_trees.composites.Parallel(
        "ResponseHandling",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll()
    )
    parallel.add_children([
        AlertOperator(name="AlertOperator"), 
        RecordData(name="RecordData")
    ])
    root.add_child(parallel)
    
    # Phase 8: Reset
    root.add_child(ResetSystem(name="ResetSystem"))
    
    return root


def create_noodle_grasping_tree():
    """
    Noodle detection and grasping behavior tree
    Requires Spot SDK and spot_bt_nodes.py
    """
    try:
        from spot_bt_nodes import ConnectToSpot, DetectNoodleWithYOLO, GraspNoodle
        
        root = py_trees.composites.Sequence("NoodleGraspingProtocol", memory=True)
        
        root.add_children([
            ConnectToSpot(name="ConnectToSpot"),
            DetectNoodleWithYOLO(name="DetectNoodle"),
            GraspNoodle(name="GraspNoodle")
        ])
        
        return root
        
    except ImportError as e:
        print(f"\n⚠️  Spot integration not available: {e}")
        print("Make sure spot_bt_nodes.py and Spot SDK are installed")
        print("Run in safety check mode: python3 main.py\n")
        return None


def main():
    """Main execution"""
    global ros_node
    import sys
    
    print("="*70)
    print("PILING SAFETY + NOODLE GRASPING BEHAVIOR TREE")
    print("="*70)
    
    # Initialize ROS2
    rclpy.init()
    ros_node = rclpy.create_node('piling_safety_bt_node')
    
    # Choose tree based on command line argument
    if len(sys.argv) > 1 and sys.argv[1] == 'noodle':
        print("\n🎯 Mode: Noodle Grasping with Spot")
        tree_root = create_noodle_grasping_tree()
        
        if tree_root is None:
            print("❌ Cannot run noodle mode. Exiting.")
            rclpy.shutdown()
            return
    else:
        print("\n🎯 Mode: Piling Safety Check")
        tree_root = create_tree()
    
    # Create behavior tree
    tree = py_trees_ros.trees.BehaviourTree(
        root=tree_root,
        unicode_tree_debug=True
    )
    
    # Setup
    try:
        tree.setup(node=ros_node, timeout=15.0)
        print("✅ Behavior Tree Initialized Successfully\n")
    except Exception as e:
        print(f"❌ Tree setup failed: {e}")
        rclpy.shutdown()
        return
    
    print("Press Ctrl+C to stop\n")
    
    # Main execution loop
    try:
        while rclpy.ok():
            tree.tick()
            rclpy.spin_once(ros_node, timeout_sec=0.1)
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
        
    finally:
        tree.shutdown()
        ros_node.destroy_node()
        rclpy.shutdown()
        print("✅ Clean shutdown complete\n")


if __name__ == '__main__':
    main()