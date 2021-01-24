import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('robot_log.csv',delimiter=';',decimal='.')
#Change path of the original dataset
df['Path']=df['Path'].str.slice(start=16)
path_to_list=df['Path'].tolist()


class Databucket():
    def __init__(self):
        self.images = path_to_list  
        self.xpos = df["X_Position"].values
        self.ypos = df["Y_Position"].values
        self.yaw = df["Yaw"].values
        self.count = 0 # This will be a running index
        self.worldmap = np.zeros((200, 200, 3)).astype(np.float)
        #self.ground_truth = ground_truth_3d # Ground truth worldmap
data = Databucket()


print df['Path'][5]