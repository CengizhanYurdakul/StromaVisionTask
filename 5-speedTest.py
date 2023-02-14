import time
import torch
import argparse
import numpy as np

from models import TRTModule
from utils.YoloOnnx import *
from ultralytics import YOLO
from models.utils import blob, letterbox
        
def profileTorch(args):
    model = YOLO(args.torchPath)
    model.to("cuda")
    # random_input = np.random.rand(640, 640, 3).astype(np.uint8)
    random_input = cv2.imread("datasets/val/images/0593.jpg")
    times = []
    for i in range(200):
        _ = model(random_input)


def profileOnnx(args):
    model = YOLOv8(args.onnxPath)
    # random_input = np.random.rand(640, 640, 3).astype(np.uint8)
    random_input = cv2.imread("datasets/val/images/0593.jpg")
    
    times = []
    for i in range(200):
        _ = model(random_input)

    meanTime = np.mean(model.times)
    return meanTime
    

def profileTRT(args):
    device = "cuda"
    Engine = TRTModule(args.trtPath, device)
    H, W = Engine.inp_info[0].shape[-2:]
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])
    # random_input = torch.randn(Engine.inp_info[0].shape, device=device)
    random_input = cv2.imread("datasets/val/images/0593.jpg")
    random_input, ratio, dwdh = letterbox(random_input, (W, H))
    random_input = cv2.cvtColor(random_input, cv2.COLOR_BGR2RGB)
    tensor = blob(random_input, return_seg=False)
    dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
    tensor = torch.tensor(tensor, device=device)

    times = []
    for i in range(200):
        if i < 10:
            _ = Engine(tensor.half())
        else:
            s = time.time()
            _ = Engine(tensor.half())
            times.append(time.time()-s)
        
    meanTime = np.mean(times)
    return meanTime

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-to", "--torchPath", default="runs/detect/Optimizer-Adam_LR-0.001_Pretrained-True_Augmentation-True/weights/best.pt", help="Weights of converted Torch model")
    parser.add_argument("-t", "--trtPath", default="runs/detect/Optimizer-Adam_LR-0.001_Pretrained-True_Augmentation-True/weights/best.engine", help="Weights of converted TRT model")
    parser.add_argument("-o", "--onnxPath", default="runs/detect/Optimizer-Adam_LR-0.01_Pretrained-True_Augmentation-True/weights/best.onnx", help="Weights of converted Onnx model")
    args = parser.parse_args()
    
    meanOnnx = profileOnnx(args)
    meanTRT = profileTRT(args)
    profileTorch(args)
    
    print("Onnx: ", meanOnnx, "seconds")
    print("TRT: ", meanTRT, "seconds")