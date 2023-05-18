import csv
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import re

def eyefaceanalysis(file1,file2):
    # my_data = pd.read_csv('eyedata_actual_zhiqi.csv')
    # print(my_data.head())
    SCREEN_WIDTH = 2560
    SCREEN_HEIGHT = 1440

    with open(file1, newline='') as actual_csvfile:
        actual_data = list(csv.reader(actual_csvfile))
    with open(file2, newline='') as desired_csvfile:
        desired_data = list(csv.reader(desired_csvfile))

    actual_data_1 = actual_data[0]
    actual_data_2 = actual_data[2]
    actual_data_3 = actual_data[4]
    desired_data_1 = [(int)((float)(desired_data[0][0])*SCREEN_WIDTH-1),(int)((float)(desired_data[0][1])*SCREEN_HEIGHT-1)]
    desired_data_2 = [(int)((float)(desired_data[2][0])*SCREEN_WIDTH-1),(int)((float)(desired_data[2][1])*SCREEN_HEIGHT-1)]
    desired_data_3 = [(int)((float)(desired_data[4][0])*SCREEN_WIDTH-1),(int)((float)(desired_data[4][1])*SCREEN_HEIGHT-1)]
    residual_data_1 = []
    residual_data_2 = []
    residual_data_3 = []
    for i in range(len(actual_data_1)):
        string_xy = re.findall(r'\(x=([\d]+), y=([\d]+)\)',actual_data_1[i])[0]
        actual_data_1[i] = [(int)(string_xy[0]),(int)(string_xy[1])]
        residual_data_1.append(np.sqrt((desired_data_1[0]-actual_data_1[i][0])**2+(desired_data_1[1]-actual_data_1[i][1])**2))

        string_xy = re.findall(r'\(x=([\d]+), y=([\d]+)\)',actual_data_2[i])[0]
        actual_data_2[i] = [(int)(string_xy[0]),(int)(string_xy[1])]
        residual_data_2.append(np.sqrt((desired_data_2[0]-actual_data_2[i][0])**2+(desired_data_2[1]-actual_data_2[i][1])**2))

        string_xy = re.findall(r'\(x=([\d]+), y=([\d]+)\)',actual_data_3[i])[0]
        actual_data_3[i] = [(int)(string_xy[0]),(int)(string_xy[1])]
        residual_data_3.append(np.sqrt((desired_data_3[0]-actual_data_3[i][0])**2+(desired_data_3[1]-actual_data_3[i][1])**2))

    actual_data_1 = np.array(actual_data_1)
    actual_data_2 = np.array(actual_data_2)
    actual_data_3 = np.array(actual_data_3)
    average_residual = np.mean([residual_data_1,residual_data_2,residual_data_3])
    # print(actual_data_1[:,0])
    plt.figure(0)
    plt.scatter(actual_data_1[:,0],actual_data_1[:,1],label='Point 1')
    plt.scatter(actual_data_2[:,0],actual_data_2[:,1],label='Point 2')
    plt.scatter(actual_data_3[:,0],actual_data_3[:,1],label='Point 3')
    plt.scatter(desired_data_1[0],desired_data_1[1],c='red')
    plt.scatter(desired_data_2[0],desired_data_2[1],c='red')
    plt.scatter(desired_data_3[0],desired_data_3[1],c='red')
    plt.xlabel("Screen Location X")
    plt.ylabel("Screen Location Y")
    plt.title("Face tracking cursor location actual vs desired(red)")
    plt.legend()
    # plt.show()
    plt.figure(1)
    plt.hist(residual_data_1,alpha=0.5,label="Point 1",density=True)
    plt.hist(residual_data_2,alpha=0.5,label="Point 2",density=True)
    plt.hist(residual_data_3,alpha=0.5,label="Point 3",density=True)
    plt.axvline(x=average_residual,c='red',label='average_residual')
    plt.xlabel("Residual")
    plt.ylabel("Probability Density")
    plt.title("Residual between actual and desired cursor location")
    plt.legend()
    plt.show()


if __name__ =='__main__':
    # eyefaceanalysis('eyedata_actual_zhiqi.csv','eyedata_desired_zhiqi.csv')
    eyefaceanalysis('facedata_actual_zhiqi.csv','facedata_desired_zhiqi.csv')