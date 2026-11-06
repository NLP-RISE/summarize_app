from src.device import available_devices

def test_devices_returns_list():
    assert isinstance(available_devices(), list)
