import asyncio
import signal
from bleak import BleakClient
from prometheus_client import start_http_server, Gauge

VOLTAGE = Gauge('raspberry_power_voltage', 'Voltage (V)')
CURRENT = Gauge('raspberry_power_current', 'Current (A)')
POWER = Gauge('raspberry_power_watt', 'Power (W)')

async def notification_handler(sender, data):
    if len(data) >= 27:
        voltage_raw = (data[4] << 16) | (data[5] << 8) | data[6]
        current_raw = (data[7] << 16) | (data[8] << 8) | data[9]
        VOLTAGE.set(voltage_raw / 100.0)
        CURRENT.set(current_raw / 100.0)
        POWER.set((voltage_raw * current_raw) / 10000.0)

async def connect_and_collect(mac, uuid):
    client = BleakClient(mac)
    try:
        await client.connect()
        await client.start_notify(uuid, notification_handler)

        # 종료 이벤트를 기다리는 future 생성
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                # Windows나 일부 환경에서는 signal handler 추가가 불가
                pass

        # 종료 이벤트 대기
        await stop_event.wait()
        print("Gracefully shutting down...")
        await client.stop_notify(uuid)
        await client.disconnect()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if client.is_connected:
            await client.disconnect()

async def main():
    # MAC 주소 파일에서 읽기
    with open('/etc/power-exporter/power-meter-mac', 'r') as f:
        mac = f.read().strip()
    uuid = "0000ffe1-0000-1000-8000-00805f9b34fb"
    start_http_server(8000)
    await connect_and_collect(mac, uuid)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}")
    print("프로그램 종료")
