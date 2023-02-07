import os
import argparse

def train(args):
    """
    
    """

    for optimizer in args.optimizers:
        for learningRate in args.learningRates:
            for pretrained in args.pretrainedStatus:
                for augmentation in args.augmentation:
                    name = "Optimizer-%s_LR-%s_Pretrained-%s_Augmentation-%s" % (optimizer, learningRate, pretrained, augmentation)
                    if augmentation:
                        os.system("yolo task=detect mode=train model=yolov8n.pt data=./custom.yaml epochs=1 batch=-1 imgsz=640 pretrained=%s hsv_h=0 hsv_s=0 hsv_v=0 translate=0 scale=0 mosaic=0 optimizer=%s lr0=%s name=%s" % (pretrained, optimizer, learningRate, name))
                    else:
                        os.system("yolo task=detect mode=train model=yolov8n.pt data=./custom.yaml epochs=1 batch=-1 imgsz=640 pretrained=%s optimizer=%s lr0=%s name=%s" % (pretrained, optimizer, learningRate, name))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optimizers", default=["Adam", "SGD"], help="Optimizer functions to be tested during training", type=list)
    parser.add_argument("-lr", "--learningRates", default=[0.01, 0.001, 0.0001], help="Learning rate values to be tested during training", type=list)
    parser.add_argument("-p", "--pretrainedStatus", default=[False, True], help="Pretrained status to be tested during training", type=list)
    parser.add_argument("-a", "--augmentation", default=[False, True], help="Augmentation status to be tested during training", type=list)
    args = parser.parse_args()
    train(args)