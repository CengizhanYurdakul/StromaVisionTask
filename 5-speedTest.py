import time
import argparse
import numpy as np

from utils.YoloTRT import *
from utils.YoloOnnx import *
from ultralytics import YOLO
        
def profileTorch(args):
    model = YOLO(args.torchPath)
    model.to("cuda")
    random_input = np.random.rand(640, 640, 3).astype(np.uint8)
    times = []
    for i in range(200):
        _ = model(random_input)


def profileOnnx(args):
    model = YOLOv8(args.onnxPath)
    random_input = np.random.rand(640, 640, 3).astype(np.uint8)
    
    times = []
    for i in range(200):
        _ = model(random_input)

    meanTime = np.mean(model.times)
    return meanTime
    

def profileTRT(args):
    device = torch.device("cuda")
    Engine = TRTModule(args.trtPath, device)
    profiler = TRTProfilerV0()
    Engine.set_profiler(profiler)
    random_input = torch.randn(Engine.inp_info[0].shape, device=device)

    times = []
    for i in range(200):
        if i < 10:
            _ = Engine(random_input)
        else:
            s = time.time()
            _ = Engine(random_input)
            times.append(time.time()-s)
        
    meanTime = np.mean(times)
    return meanTime



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-to", "--torchPath", default="runs/detect/Optimizer-Adam_LR-0.01_Pretrained-False_Augmentation-False/weights/best.pt", help="Weights of converted Torch model")
    parser.add_argument("-t", "--trtPath", default="runs/detect/Optimizer-Adam_LR-0.01_Pretrained-False_Augmentation-False/weights/best.engine", help="Weights of converted TRT model")
    parser.add_argument("-o", "--onnxPath", default="runs/detect/Optimizer-Adam_LR-0.01_Pretrained-False_Augmentation-False/weights/best.onnx", help="Weights of converted Onnx model")
    args = parser.parse_args()
    
    meanOnnx = profileOnnx(args)
    meanTRT = profileTRT(args)
    profileTorch(args)
    
    print("Onnx: ", meanOnnx, "seconds")
    print("TRT: ", meanTRT, "seconds")