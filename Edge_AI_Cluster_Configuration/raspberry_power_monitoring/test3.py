import asyncio
from bleak import BleakClient

DEVICE_MAC = "65:97:56:84:5E:08"  # 실제 장치 MAC 주소
CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"  # 데이터 특성 UUID

def parse_ble_data(data):
    if len(data) >= 27:  # 최소 27바이트 필요
        voltage_raw = (data[4] << 16) | (data[5] << 8) | data[6]
        voltage = voltage_raw / 100.0  # V
        current_raw = (data[7] << 16) | (data[8] << 8) | data[9]
        current = current_raw / 100.0  # A
        hour = data[24]
        minute = data[25]
        second = data[26]
        print(f"[{hour:02d}:{minute:02d}:{second:02d}] 전압: {voltage:.2f} V, 전류: {current:.2f} A, 전력: {(voltage * current):.2f} W")
    else:
        print("잘못된 데이터 길이")

async def run(address):
    count = 0
    max_count = 30

    def notification_handler(sender, data):
        nonlocal count
        parse_ble_data(data)
        count += 1
        # 30번 데이터를 받으면 이벤트 루프 종료
        if count >= max_count:
            loop = asyncio.get_event_loop()
            loop.stop()

    async with BleakClient(address) as client:
        connected = await client.is_connected()
        print(f"Connected: {connected}")
        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)
        # 최대 60초 동안 대기 (혹시 데이터가 덜 오면 종료)
        await asyncio.sleep(60)
        await client.stop_notify(CHARACTERISTIC_UUID)

if __name__ == "__main__":
    try:
        asyncio.run(run(DEVICE_MAC))
    except Exception as e:
        print(f"Error: {e}")
    print("프로그램 종료")
