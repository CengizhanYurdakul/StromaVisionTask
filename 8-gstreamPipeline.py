from threading import Thread
import gi
import cv2
import numpy as np
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

def detect(pad, info):
    buf = info.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)
    
    numpy_frame = np.ndarray(
        shape=(640, 640, 4),
        dtype=np.uint8,
        buffer=map_info.data)
    
    return Gst.PadProbeReturn.OK

Gst.init()

main_loop = GLib.MainLoop()
thread = Thread(target=main_loop.run)
thread.start()

pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! decodebin ! videoconvert ! fakesink name=s")
pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    detect
)

pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        msg = pipeline.get_bus().timed_pop_filtered(
                Gst.SECOND,
                Gst.MessageType.EOS | Gst.MessageType.ERROR
            )
        if msg:
            text = msg.get_structure().to_string() if msg.get_structure() else ''
            msg_type = Gst.message_type_get_name(msg.type)
            print(f'{msg.src.name}: [{msg_type}] {text}')
            break
except KeyboardInterrupt:
    pass

pipeline.set_state(Gst.State.NULL)
main_loop.quit()