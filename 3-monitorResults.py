import os
import argparse
from tqdm import tqdm
from ultralytics import YOLO

def monitor(args):
    """
    Calculating some metrics to understand the success of the models
    """

    path = args.inputPath
    dirs = os.listdir(path)

    for modelPath in tqdm(dirs):
        model = YOLO(os.path.join(path, modelPath, "weights/best.pt"))
        for configPath in ["configs/test.yaml", "configs/val.yaml"]:
            print("\n%s %s\n" % (modelPath, configPath.upper()))
            args = dict(model=model, data=configPath)
            model.val(**args)
            print("\n------------------------------------------------\n")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputPath", default="runs/detect", help="Input path of training experiments")
    args = parser.parse_args()
    monitor(args)