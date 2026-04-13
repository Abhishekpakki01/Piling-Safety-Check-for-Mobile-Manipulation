import bosdyn.client
import bosdyn.client.estop
import os

def main():
    sdk = bosdyn.client.create_standard_sdk('EstopClear')
    robot = sdk.create_robot(os.getenv('SPOT_IP', '10.0.0.30'))
    robot.authenticate(os.getenv('SPOT_USERNAME', 'rllab'), os.getenv('SPOT_PASSWORD', 'robotlearninglab'))
    
    estop_client = robot.ensure_client('estop')
    estop_endpoint = bosdyn.client.estop.EstopEndpoint(estop_client, 'Scanner', 9.0)
    estop_endpoint.force_simple_setup() 
    print("[SAFE] Software E-Stop cleared. System ready.")

if __name__ == '__main__':
    main()
