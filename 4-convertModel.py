import os
import argparse
from tqdm import tqdm

def convert(args):
    path = args.inputPath
    dirs = os.listdir(path)
    
    for modelPath in tqdm(dirs):
        model = os.path.join(path, modelPath, "weights/best.pt")
        os.system("yolo mode=export model=%s format=onnx opset=11 simplify=True" % model)
        os.system("yolo mode=export model=%s format=openvino simplify=True opset=11" % model)
        os.system("yolo mode=export model=%s format=engine simplify=True opset=11 device=0" % model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputPath", default="runs/detect", help="Input path of training experiments")
    args = parser.parse_args()
    convert(args)