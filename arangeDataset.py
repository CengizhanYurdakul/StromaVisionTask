import os
import cv2
import json
import argparse
from glob import glob
from tqdm import tqdm

def convertBox2Yolo(box):
    dw = 1/640
    dh = 1/640
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def main(args):
    """
    Pipeline for transferring the training dataset from video and switching from COCO format to YOLO format for YOLO training
    """

    videoNames = glob(os.path.join(args.inputPath, "**/**/*.mp4"))

    for videoName in tqdm(videoNames):
        videoCapture = cv2.VideoCapture(videoName)
        
        c = 0
        
        if not os.path.exists(os.path.join(args.outputPath, "%s/images" % videoName.split("/")[-2])):
            os.makedirs(os.path.join(args.outputPath, "%s/images" % videoName.split("/")[-2]))
        
        while True:
            ret, frame = videoCapture.read()
            if ret == True:
                cv2.imwrite(os.path.join(args.outputPath, "%s/images" % videoName.split("/")[-2], "%04d.jpg" % c), frame)
                c += 1
            else:
                break
            
        f = open(os.path.join("%s/annotations" % args.inputPath, "instances_%s.json" % videoName.split("/")[-2]))
        annotations = json.load(f)
        
        if not os.path.exists(os.path.join(args.outputPath, "%s/labels" % videoName.split("/")[-2])):
            os.makedirs(os.path.join(args.outputPath, "%s/labels" % videoName.split("/")[-2]))
        
        for annotation in annotations["annotations"]:
            imageName = annotation["image_id"] - 1
            bbox = annotation["bbox"]
            classID = annotation["category_id"] - 1
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            
            f1, f2, f3, f4 = convertBox2Yolo((x, x+w, y, y+h))
            
            if os.path.exists(os.path.join(args.outputPath, videoName.split("/")[-2], "labels/%04d.txt" % imageName)):
                f = open(os.path.join(args.outputPath, videoName.split("/")[-2], "labels/%04d.txt" % imageName), "a")
            else:
                f = open(os.path.join(args.outputPath, videoName.split("/")[-2], "labels/%04d.txt" % imageName), "w")
            f.write("%s %s %s %s %s\n" % (classID, f1, f2, f3, f4))
            f.close()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputPath", default="challenge", help="Input path of raw dataset")
    parser.add_argument("-o", "--outputPath", default="datasets", help="Output path for aranged dataset")
    args = parser.parse_args()
    main(args)