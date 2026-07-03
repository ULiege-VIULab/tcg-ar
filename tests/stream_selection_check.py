"""Verify selectable stream production: set_stream_active flips both the activation
flag and the RTSP-sender wait_event; the Stream_panel checkboxes drive it; and the
lean-default pattern leaves only the first camera's broadcast shot active. Headless
Qt (offscreen)."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import types
import numpy as np

from PySide6 import QtWidgets
from inference import user_interface as ui


class FakeEvent:
    def __init__(self): self._set = True
    def set(self): self._set = True
    def clear(self): self._set = False
    def is_set(self): return self._set


class FakeDetected:
    def __init__(self, n): self.n = n
    def get_nb_cam_available(self): return self.n


class FakeSCW(QtWidgets.QWidget):
    """Minimal stand-in exposing what Stream_panel + the real set_stream_active need."""
    def __init__(self, nb):
        super().__init__()
        self.detected_cam = FakeDetected(nb)
        self.camera_activated = [True] * (3 * nb)
        self.output_buffer_wait_events = [FakeEvent() for _ in range(3 * nb)]
    # bind the real implementation so we exercise production code
    set_stream_active = ui.Show_camera_widget.set_stream_active


def main():
    app = QtWidgets.QApplication([])
    nb = 2

    # --- real set_stream_active toggles flag + event together ---
    obj = types.SimpleNamespace(camera_activated=[True, True, True],
                                output_buffer_wait_events=[FakeEvent(), FakeEvent(), FakeEvent()])
    ui.Show_camera_widget.set_stream_active(obj, 1, False)
    assert obj.camera_activated[1] is False and obj.output_buffer_wait_events[1].is_set() is False
    ui.Show_camera_widget.set_stream_active(obj, 1, True)
    assert obj.camera_activated[1] is True and obj.output_buffer_wait_events[1].is_set() is True

    # --- lean default: only index 2*nb (cam-0 broadcast) stays active ---
    scw = FakeSCW(nb)
    default_index = 2 * nb
    for index in range(len(scw.camera_activated)):
        scw.set_stream_active(index, index == default_index)
    assert sum(scw.camera_activated) == 1
    assert scw.camera_activated[default_index] is True
    assert all(scw.output_buffer_wait_events[i].is_set() == (i == default_index)
               for i in range(3 * nb))

    # --- Stream_panel checkbox drives production; sync reflects external changes ---
    panel = ui.Stream_panel(scw)
    assert len(panel.checks) == 3 * nb
    # tick the cam-1 AR shot (index = 1 + 1*nb)
    ar_cam1 = 1 + 1 * nb
    panel.checks[ar_cam1].setChecked(True)
    assert scw.camera_activated[ar_cam1] is True
    assert scw.output_buffer_wait_events[ar_cam1].is_set() is True
    # external change (e.g. tile click) -> panel re-syncs checkbox state
    scw.set_stream_active(ar_cam1, False)
    panel._sync()
    assert panel.checks[ar_cam1].isChecked() is False

    # --- RTSP address overlay + toggle ---
    frame = np.zeros((20, 30, 3), np.uint8)
    buttons = [ui.Picture_button(scw, frame, i) for i in range(3)]
    for i, b in enumerate(buttons):
        b.address = "rtsp://localhost:8554/ptcgAR/%d" % i
    holder = types.SimpleNamespace(button_list=buttons, show_stream_addresses=False,
                                   get_stream_addresses_shown=ui.Show_camera_widget.get_stream_addresses_shown,
                                   toggle_stream_addresses=ui.Show_camera_widget.toggle_stream_addresses)
    assert all(b.show_address is False for b in buttons)
    ui.Show_camera_widget.toggle_stream_addresses(holder)
    assert holder.show_stream_addresses is True and all(b.show_address for b in buttons)
    ui.Show_camera_widget.toggle_stream_addresses(holder)
    assert holder.show_stream_addresses is False and all(b.show_address is False for b in buttons)

    print("STREAM_SELECTION_OK")


if __name__ == "__main__":
    main()
