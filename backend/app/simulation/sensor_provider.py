"""
sensor_provider.py
==================
Abstract SensorProvider interface.

ARCHITECTURE DESIGN:
    SensorProvider (abstract)
        |
        +-- SimulatedSensorProvider  ← current implementation
        |
        +-- FutureRealSensorProvider ← placeholder for real hardware

This separation means the rest of the application NEVER depends directly
on the simulation. When real sensors are available, only the provider
implementation changes — the API, ML pipeline, and frontend stay the same.
"""
from abc import ABC, abstractmethod
from ..schemas.sensor import SensorReading


class SensorProvider(ABC):
    """
    Abstract interface for sensor data sources.

    Implementors:
        - SimulatedSensorProvider (this prototype)
        - FutureRealSensorProvider (ESP32/BLE hardware — not yet implemented)
    """

    @abstractmethod
    def get_reading(self) -> SensorReading:
        """Return the latest sensor reading."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the provider state (e.g., time offset)."""
        ...

    @property
    @abstractmethod
    def source_label(self) -> str:
        """Human-readable label for the data source (e.g., 'SIMULATED')."""
        ...


class FutureRealSensorProvider(SensorProvider):
    """
    Placeholder for future real hardware integration.

    When an ESP32/microcontroller is available:
      - This class would connect via Bluetooth / Wi-Fi / serial
      - Parse incoming sensor packets
      - Return SensorReading objects in the same format

    Current status: NOT IMPLEMENTED — hardware interface planned for future.
    """

    def get_reading(self) -> SensorReading:
        raise NotImplementedError(
            "Real sensor hardware not connected. "
            "Hardware interface planned for future integration. "
            "Use SimulatedSensorProvider for the prototype."
        )

    def reset(self) -> None:
        pass

    @property
    def source_label(self) -> str:
        return "REAL_HARDWARE_NOT_CONNECTED"
