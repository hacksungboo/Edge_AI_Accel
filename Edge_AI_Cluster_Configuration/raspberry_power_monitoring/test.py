import bluetooth
import struct

# J7-C 블루투스 장치의 MAC 주소
DEVICE_MAC = "65:97:56:84:5E:08"  # 실제 장치 주소로 변경

def connect_device(mac_address, port=1):
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    try:
        sock.connect((mac_address, port))
        print(f"블루투스 장치({mac_address})에 연결되었습니다.")
        return sock
    except Exception as e:
        print(f"블루투스 연결 실패: {e}")
        return None

def parse_packet(packet):
    if len(packet) < 36:
        return None, None, None
    # 전압(4~6번째 바이트)
    voltage_raw = (packet[4] << 16) | (packet[5] << 8) | packet[6]
    voltage = voltage_raw / 100.0  # V
    # 전류(7~9번째 바이트)
    current_raw = (packet[7] << 16) | (packet[8] << 8) | packet[9]
    current = current_raw / 100.0  # A
    # 전력 계산(W)
    power = voltage * current
    return voltage, current, power

def main():
    sock = connect_device(DEVICE_MAC)
    if sock is None:
        return

    try:
        while True:
            data = sock.recv(1024)
            if len(data) >= 36:
                # 패킷 파싱
                voltage, current, power = parse_packet(data)
                if voltage is not None:
                    print(f"전압: {voltage:.2f} V, 전류: {current:.2f} A, 전력: {power:.2f} W")
    except KeyboardInterrupt:
        print("종료합니다.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
