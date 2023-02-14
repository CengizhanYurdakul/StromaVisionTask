import os
import cv2
import torch
import argparse
from glob import glob
from tqdm import tqdm

from models.torch_utils import det_postprocess
from models.utils import blob, letterbox, path_to_list

from models import TRTModule

CLASSES = ('0', '1')

def countMet(image, tracker, line=200, count1=0, count2=0):
    
    image = cv2.line(image, (0, line), (1920,line), (255,255,0), 1)

    for track in tracker:
        x1, y1, x2, y2 = track.get_state()[0]
        if not hasattr(track, 'lastY'):
            track.lastY  = y1 + ((y2 - y1 )/ 2 )
            track.count_flag = False

            if y1 + ((y2 - y1 )/ 2 ) > line:
                track.reverse = True
            else:
                track.reverse = False

        if not track.reverse and not track.count_flag:
                
            if track.lastY < line and (y1 + ((y2 - y1 )/ 2 ) ) > line :

                track.count_flag = True
                object = track.objectClass
                if object.item() == 0:
                    count1 += 1
                else:
                    count2 += 1
        
        # if track.reverse and not track.count_flag:
        #     if track.matched:
        #         if track.lastY > line and (track.bbox[1] + ((track.bbox[3] - track.bbox[1] )/ 2 ) ) < line :
        #             track.count_flag = True
        #             count_reverse +=1

        track.lastY  = y1 + ((y2 - y1 )/ 2 )
        
    image = cv2.putText(
                image,
                "0: %s" % count1,
                (int(50), int(50)),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (255, 0, 0),
                1,
            )
    
    image = cv2.putText(
                image,
                "1: %s" % count2,
                (int(50), int(100)),
                cv2.FONT_HERSHEY_DUPLEX,
                1,
                (0, 0, 255),
                1,
            )
    
    return image, count1, count2

def main(args):
    # model = YOLO(args.modelPath)
    # model.to("cuda")

    device = "cuda"
    Engine = TRTModule(args.modelPath, device)
    H, W = Engine.inp_info[0].shape[-2:]
    Engine.set_desired(['num_dets', 'bboxes', 'scores', 'labels'])

    videoNames = glob(os.path.join(args.inputPath, "**/**/*.mp4"))
    for videoName in videoNames:
        videoCapture = cv2.VideoCapture(videoName)
        
        from utils.sort import Sort
        tracker = Sort(
        max_age=1,
        min_hits=3,
        iou_threshold=0.3
        )
        
        count1 = 0
        count2 = 0
        
        frameW = int(videoCapture.get(3) / 1)
        frameH = int(videoCapture.get(4) / 1)
        total = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        out = cv2.VideoWriter(
            "%s/%s" % (args.outputPath, videoName.split("/")[-1]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            30,
            (frameW ,frameH),
        )
        
        for _ in tqdm(range(total)):
            ret, frame = videoCapture.read()
            
            
            if ret == True:
                frame, ratio, dwdh = letterbox(frame, (W, H))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = blob(frame, return_seg=False)
                dwdh = torch.tensor(dwdh * 2, dtype=torch.float32, device=device)
                tensor = torch.tensor(tensor, device=device)
                data = Engine(tensor)
                
                bboxes, scores, labels = det_postprocess(data)
                bboxes -= dwdh
                bboxes /= ratio

                trackers = tracker.update((bboxes).to("cpu").numpy(), labels)
                frame, count1, count2 = countMet(frame, trackers, 450, count1, count2)
                    
                for track in trackers:
                    x1, y1, x2, y2 = track.get_state()[0]
                    
                    if track.objectClass.item() == 0:
                        color = (255, 0, 0)
                    else:
                        color = (0, 0, 255)
                        
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        color,
                        3
                    )
                    cv2.putText(
                        frame,
                        "ID: %s" % int(track.id),
                        (int(x1), int(y1)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        color,
                        2
                    )
            else:
                break
            
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
        del tracker
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--modelPath", default="runs/detect/Optimizer-Adam_LR-0.0001_Pretrained-True_Augmentation-True/weights/best.engine", help="Weights of model")
    parser.add_argument("-i", "--inputPath", default="challenge", help="Input path of videos")
    parser.add_argument("-o", "--outputPath", default="outs", help="Output path of processed videos")
    args = parser.parse_args()
    
    main(args)
    