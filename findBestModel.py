import os
import pandas as pd
from tqdm import tqdm

path = "runs/detect"
dirs = os.listdir(path)

for modelPath in tqdm(dirs):
    resultPath = os.path.join(path, modelPath, "results.csv")
    df = pd.read_csv(resultPath, index_col=0)
    
    print("%s: %s" % (modelPath, df.iloc[:, 6].max()))