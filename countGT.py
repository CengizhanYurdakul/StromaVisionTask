import os
import json

path = "challenge/annotations"
dirs = os.listdir(path)

for fileName in dirs:
    f = open(os.path.join(path, fileName))
    annotations = json.load(f)
    
    class1 = []
    class2 = []
    print("%s calculating..." % fileName.split("_")[-1])
    for annotation in annotations["annotations"]:
        if (annotation["category_id"] == 1) and (annotation["track_id"] not in class1):
            class1.append(annotation["track_id"])
        elif (annotation["category_id"] == 2) and (annotation["track_id"] not in class2):
            class2.append(annotation["track_id"])
            
    print("Class0: %s - Class1: %s" % (len(class1), len(class2)))