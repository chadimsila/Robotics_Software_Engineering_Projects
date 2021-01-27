import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('robot_log.csv',delimiter=',',decimal='.')

df['Path']=df['Path'].str.slice(start=55)
img=cv2.imread(df['Path'][1],cv2.IMREAD_GRAYSCALE)
print(img)
