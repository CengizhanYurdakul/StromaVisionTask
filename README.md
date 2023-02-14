# `1-arangeDataset.py`
```
python arangeDataset.py -i <Input dataset directory> -o <Processed dataset directory>
```

# `2-gridSearch.py`
```
python 2-gridSearch.py -o <Optimizer functions> -lr <Learning rate values> -p <Pretrained status(True/False)> -a <Augmentation status(True/False)>
```

# `3-monitorResults.py`
```
python 3-monitorResults.py -i <Models weight directory>
```

# `4-convertModel.py`
```
python 4-convertModel.py -i <Models weight directory>
```

# `5-speedTest.py`
```
python 5-speedTest.py -to <Torch weight file path> -t <TensorRT weight file path> -o <Onnx weight file path>
```

# `6-trackObject.py`
```
python 6-trackObject.py -w <Torch model path> -i <Video path> -o <Video save path>
```

# `9-convert2onnx.py`
```
python 9-convert2onnx.py -w <Torch model weights> --iou-thres <IOU threshold for NMS> --conf-thres <Confidence for NMS> --topk <Number of maximum detection> --opset <Onnx opset> --sim <Simplify onnx(True/False)> --input-shape <Model input shape> --device <CUDA/CPU>
```

# `10-convert2trt.py`
```
python 10-convert2trt.py -w <Torch model weights> --iou-thres <IOU threshold for NMS> --conf-thres <Confidence for NMS> --topk <Number of maximum detection> --input-shape <Model input shape> --fp16 <fp16 precision(True/False)> --device <CUDA/CPU>
```

# `11-trackObjectWithTRT.py`
```
python 6-trackObject.py -w <TensorRT weight path> -i <Video path> -o <Video save path>
```