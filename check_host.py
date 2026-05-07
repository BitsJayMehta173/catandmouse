import socket
import os
import network_utils
import subprocess

def get_ips_via_ipconfig():
    print("Running ipconfig...")
    try:
        output = subprocess.check_output("ipconfig", shell=True).decode()
        for line in output.split('\n'):
            if "IPv4 Address" in line:
                print(f"  Found via ipconfig: {line.split(':')[-1].strip()}")
    except:
        pass

def check():
    print("--- Advanced Host Diagnostic ---")
    
    # Method 1: socket.getaddrinfo
    print("\nMethod 1: Internal Hostname lookup")
    hostname = socket.gethostname()
    print(f"Hostname: {hostname}")
    try:
        ips = []
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if "." in ip and ip not in ips:
                ips.append(ip)
        print(f"Detected IPs: {ips}")
    except Exception as e:
        print(f"Error in Method 1: {e}")

    # Method 2: UDP trick to find primary outgoing IP
    print("\nMethod 2: Primary Network Interface (Internet Route)")
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't actually connect, just finds the interface that would route to this IP
        s.connect(("8.8.8.8", 80))
        primary_ip = s.getsockname()[0]
        print(f"Primary IP: {primary_ip}  <-- USE THIS IP ON THE CLIENT!")
        s.close()
    except Exception as e:
        print(f"Method 2 failed (no internet?): {e}")

    # Method 3: ipconfig
    print("\nMethod 3: All Windows Interfaces")
    get_ips_via_ipconfig()
    
    print("\n--- Port Availability Check ---")
    for name, port in [("UDP", network_utils.UDP_PORT), ("TCP", network_utils.TCP_PORT), ("GAZE", network_utils.GAZE_PORT)]:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM if name != "UDP" else socket.SOCK_DGRAM)
        try:
            s.bind(('0.0.0.0', port))
            print(f"  [OK] {name} Port {port} is AVAILABLE.")
        except Exception as e:
            print(f"  [X] {name} Port {port} is BUSY/BLOCKED: {e}")
        finally:
            s.close()

if __name__ == "__main__":
    check()
