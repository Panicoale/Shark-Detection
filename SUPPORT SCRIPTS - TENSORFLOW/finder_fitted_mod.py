import os
import csv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import matplotlib


folder= '/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/EVAL/validation'
folder='/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/evaluation_data'
folder='/home/mueavi-pc-01/Desktop/tf/WS_shark/SSD_INCEPTION/EVAL/evaluation'
folder='/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/evaluation_data'
folder= '/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/EVAL/validation'

folder='/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/evaluation_data'


elements = os.listdir(folder)

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

STEP=[]
VALUE=[]
TIME=[]
IOU=[]
for element in elements:
    if 'PerformanceByCategory' in element:
        with open(os.path.join(folder,element), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            step=[]
            value=[]
            time=[]
            n=0
            for row in csv_reader:
                #print(row)
                n+=1

                if line_count == 0:
                    #print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
                    #print(row)
                    line_count += 1
                    step.append(row[1])
                    value.append(row[2])
                    time.append(row[0])
            STEP.append(step)
            VALUE.append(value)
            TIME.append(time)
    if 'OpenImagesV2_Precision_mAP' in element:
        with open(os.path.join(folder,element), mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                line_count += 1
                if line_count > 1:
                    
                    IOU.append(100*float(row[2]))
print(n)
SUPPORT=[]
for i in range(n-1):
    SUPPORT.append(0)
#print(STEP)    
for i,step_i in enumerate(STEP):
    value_i=VALUE[i]
    #print(value_i)
    for j,value_j in enumerate(value_i):
        #print(value_j)
        SUPPORT[j]+=SUPPORT[j]+float(value_j)
mAP=[]
for value in SUPPORT:
    mAP.append(round(value/n,2))

#print((mAP))
print((IOU))

cost=[]
for i,value in enumerate(mAP):
    cost.append(0.5*(value+float(IOU[i])))


# print(cost)

index_max = np.argmax(cost)
# print(cost[index_max])
# print(time[index_max])
# print(step[index_max])

SMALL_SIZE = 17
MEDIUM_SIZE = 25
BIGGER_SIZE = 27
VERY_BIGGER_SIZE = 28

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=VERY_BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=VERY_BIGGER_SIZE) 


figure(num=None, figsize=(20, 15), dpi=80, facecolor='w', edgecolor='k')
#MAP_PLOT=plt.plot(step,mAP, label = "Performance")
MAP_PLOT=plt.plot(step,IOU, label = "mAP")
#MAP_PLOT=plt.plot(step,cost, label = "Cost Function")
MAP_PLOT=plt.title('faster R-CNN - ResNet 101') 
MAP_PLOT=plt.xlabel("Training Steps")
MAP_PLOT=plt.ylabel("mAP")
plt.legend() 

#plt.show(MAP_PLOT)
#lt.close()
plt.savefig(os.path.join(folder,'results.png'))
