# custom_ exporter.py
from prometheus_client import start_http_server, Gauge
from jtop import jtop
import time

board_voltage = Gauge('jetson_board_voltage_volt', 'Jetson board total voltage in volts')
board_current = Gauge('jetson_board_current_ampere', 'Jetson board total current in amperes')
board_power = Gauge('jetson_board_power_watt', 'Jetson board total power consumption in watts')

def collect_metrics():
    with jtop() as jetson:
        while jetson.ok():
            power = jetson.power
            if "tot" in power:
                tot = power["tot"]
                volt = tot.get("volt")
                curr = tot.get("curr")
                watt = tot.get("power")
                if volt is not None:
                    board_voltage.set(volt / 1000.0)
                else:
                    board_voltage.set(0)
                if curr is not None:
                    board_current.set(curr / 1000.0)
                else:
                    board_current.set(0)
                if watt is not None:
                    board_power.set(watt / 1000.0)
                else:
                    board_power.set(0)
            else:
                board_voltage.set(0)
                board_current.set(0)
                board_power.set(0)
            time.sleep(1)

if __name__ == "__main__":
    start_http_server(8000)
    collect_metrics()
