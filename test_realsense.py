import pyrealsense2 as rs

def check_realsense_connection():
    # Create a context object. This manages all connected RealSense devices.
    ctx = rs.context()
    
    # Query all connected devices
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices found.")
        return False
    
    print(f"Detected {len(devices)} device(s):")
    
    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        sn = dev.get_info(rs.camera_info.serial_number)
        fw = dev.get_info(rs.camera_info.firmware_version)
        print(f"[{i}] {name} (S/N: {sn}) | FW: {fw}")
        
    return True

if __name__ == "__main__":
    check_realsense_connection()
