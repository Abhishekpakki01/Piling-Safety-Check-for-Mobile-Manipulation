import os, time, sys
import bosdyn.client
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, block_until_arm_arrives

def force_shutdown():
    sdk = bosdyn.client.create_standard_sdk('ForceRelax')
    robot = sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
    robot.authenticate(os.getenv('SPOT_USERNAME', 'rllab'), os.getenv('SPOT_PASSWORD', 'robotlearninglab'))
    
    # Forcefully take the lease away from any crashed scripts
    lease_client = robot.ensure_client('lease')
    lease_wallet = lease_client.take()
    lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True)
    
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    print("[1/3] Folding the arm...")
    stow_id = command_client.robot_command(RobotCommandBuilder.arm_stow_command())
    time.sleep(2.0) # Give it time to tuck
    
    print("[2/3] Sitting down (relaxing legs)...")
    command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
    time.sleep(3.0) 
    
    print("[3/3] Cutting motor power.")
    robot.power_off()
    print("[SUCCESS] Robot is now relaxed and safe to move.")

if __name__ == '__main__':
    force_shutdown()
