import os, time, sys
import bosdyn.client
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient

def run_kill_switch():
    sdk = bosdyn.client.create_standard_sdk('KillSwitch')
    robot = sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
    robot.authenticate(os.getenv('SPOT_USERNAME', 'rllab'), os.getenv('SPOT_PASSWORD', 'robotlearninglab'))
    
    lease_client = robot.ensure_client('lease')
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)

    print("\n" + "!"*40)
    print("!!! SPOT EMERGENCY STOP READY !!!")
    print("Press ENTER to STOW ARM, SIT, and KILL POWER.")
    print("!"*40 + "\n")

    input("MONITORING... [PRESS ENTER TO STOP]")

    try:
        # Take the lease to ensure we can command it
        lease_wallet = lease_client.take()
        lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True)
        
        print("[SHUTDOWN] Stowing arm...")
        command_client.robot_command(RobotCommandBuilder.arm_stow_command())
        time.sleep(2.0)

        print("[SHUTDOWN] Sitting down...")
        command_client.robot_command(RobotCommandBuilder.synchro_sit_command())
        time.sleep(2.0) 

        robot.power_off()
        print("[SAFE] Power cut.")
    except Exception as e:
        print(f"Error: {e}")
        robot.power_off_cutdown() # Last resort hard cut
    finally:
        os._exit(0)

if __name__ == '__main__':
    run_kill_switch()
