# gst-launch-1.0 v4l2src device=/dev/video0 ! videoconvert! videoscale ! video/x-raw, width=2592, height=600 ! autovideosink -v
import os, sys
import gi
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
gi.require_version('Gst', '1.0')
from gi.repository import Gst

model = YOLO("runs/detect/Optimizer-Adam_LR-0.01_Pretrained-True_Augmentation-True/weights/best.pt")
model.to("cuda")

frame_format = 'RGBA'

Gst.init()
pipeline = Gst.parse_launch(f'''
    filesrc location=challenge/images/val/val.mp4 !
    decodebin !
    videoconvert !
    video/x-raw,format=RGBA !
    fakesink name=s
''')

def on_frame_probe(pad, info):
    buf = info.get_buffer()
    success, map_info = buf.map(Gst.MapFlags.READ)
    
    numpy_frame = np.ndarray(
        shape=(640, 640, 4),
        dtype=np.uint8,
        buffer=map_info.data)
    
    results = model(cv2.cvtColor(numpy_frame, cv2.COLOR_RGBA2RGB))
    xyxy = results[0].boxes.xyxy
    
    for i in xyxy:
        cv2.rectangle(
                    numpy_frame,
                    (int(i[0]), int(i[1])),
                    (int(i[2]), int(i[3])),
                    (255, 0, 0),
                    3
                    )
    
    cv2.imwrite("deneme/%s.jpg" % time(), numpy_frame)
    
    print(f'[{buf.pts / Gst.SECOND:6.2f}]')
    return Gst.PadProbeReturn.OK

pipeline.get_by_name('s').get_static_pad('sink').add_probe(
    Gst.PadProbeType.BUFFER,
    on_frame_probe
)

pipeline.set_state(Gst.State.PLAYING)

startTime = time()
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
    
endTime = time()
print("TOTAL TIME: ", round(endTime - startTime, 4)) # 26.8523
pipeline.set_state(Gst.State.NULL)