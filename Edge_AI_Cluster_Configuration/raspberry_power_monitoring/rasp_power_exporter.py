import asyncio
from bleak import BleakClient
from prometheus_client import start_http_server, Gauge

# Prometheus 메트릭 정의
VOLTAGE = Gauge('raspberry_power_voltage', 'Voltage (V)')
CURRENT = Gauge('raspberry_power_current', 'Current (A)')
POWER = Gauge('raspberry_power_watt', 'Power (W)')


DEVICE_MAC='65:97:56:84:5E:08'
UUID='0000ffe1-0000-1000-8000-00805f9b34fb'

def parse_ble_data(data):
    if len(data) >= 27:
        voltage_raw = (data[4] << 16) | (data[5] << 8) | data[6]
        voltage = voltage_raw / 100.0
        current_raw = (data[7] << 16) | (data[8] << 8) | data[9]
        current = current_raw / 100.0
        hour = data[24]
        minute = data[25]
        second = data[26]
        # 메트릭 업데이트
        VOLTAGE.set(voltage)
        CURRENT.set(current)
        POWER.set(voltage * current)
        print(f"[{hour:02d}:{minute:02d}:{second:02d}] 전압: {voltage:.2f} V, 전류: {current:.2f} A, 전력: {(voltage * current):.2f} W")
    else:
        print("잘못된 데이터 길이")

async def run(address):
    def notification_handler(sender, data):
        parse_ble_data(data)

    async with BleakClient(address) as client:
        connected = await client.is_connected()
        print(f"Connected: {connected}")
        await client.start_notify(UUID, notification_handler)
        # 무한 루프로 계속 데이터 수신
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    # Prometheus 서버 시작
    start_http_server(8000)
    try:
        asyncio.run(run(DEVICE_MAC))
    except Exception as e:
        print(f"Error: {e}")
    print("프로그램 종료")
