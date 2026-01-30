import asyncio
from bleak import BleakClient

DEVICE_MAC = "65:97:56:84:5E:08"  # 실제 장치 MAC 주소
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"  # 데이터 특성 UUID

def parse_ble_data(data):
    if len(data) >= 10:
        voltage_raw = int.from_bytes(data[4:7], byteorder='little', signed=False)
        current_raw = int.from_bytes(data[7:10], byteorder='little', signed=False)
        voltage = voltage_raw / 100.0  # V
        current = current_raw / 100.0  # A
        power = voltage * current  # W
        print(f"Voltage: {voltage:.2f} V, Current: {current:.2f} A, Power: {power:.2f} W")
    else:
        print("Invalid data length")

def parse_ble_data2(data):
    if len(data) >= 27:  # 최소 27바이트 필요 (24~26번 인덱스 시간 정보 포함)
        # 전압 파싱 (4~6번 인덱스)
        voltage_raw = (data[4] << 16) | (data[5] << 8) | data[6]
        voltage = voltage_raw / 100.0  # V
        
        # 전류 파싱 (7~9번 인덱스)
        current_raw = (data[7] << 16) | (data[8] << 8) | data[9]
        current = current_raw / 100.0  # A
        
        # 전력 계산
        power = voltage * current  # W
        
        # 시간 파싱 (24~26번 인덱스)
        hour = data[24]
        minute = data[25]
        second = data[26]
        
        print(f"[{hour:02d}:{minute:02d}:{second:02d}] 전압: {voltage:.2f} V, 전류: {current:.2f} A, 전력: {power:.2f} W")
    else:
        print("잘못된 데이터 길이")

async def run(address):
    def notification_handler(sender, data):
        print("Received data:", data.hex())
        parse_ble_data2(data)

    async with BleakClient(address) as client:
        connected = await client.is_connected()
        print(f"Connected: {connected}")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        await asyncio.sleep(60)
        await client.stop_notify(CHARACTERISTIC_UUID)

if __name__ == "__main__":
    try:
        asyncio.run(run(DEVICE_MAC))
    except Exception as e:
        print(f"Error: {e}")
