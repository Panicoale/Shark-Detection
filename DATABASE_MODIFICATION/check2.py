import csv
import cv2 
import os
import numpy as np

FOLDER = './train'
CSV_FILE = './annotation/train_labels.csv'
CSV_OUTPUT = './annotation/train_labels3.csv'


FOLDER = './test'
CSV_FILE = './annotation/test_labels.csv'
CSV_OUTPUT = './annotation/test_labels3.csv'

FOLDER = './train'
CSV_FILE = './annotation/train_labels.csv'
CSV_OUTPUT = './annotation/train_labels3.csv'

FOLDER = './validation'
CSV_FILE = './annotation/validation_labels.csv'
CSV_OUTPUT = './annotation/validation_labels3.csv'


lines=[]
lines.append(['filename','width','height','class','xmin','ymin','xmax','ymax'])

with open(CSV_FILE, 'r') as fid:
    
    print('Checking file:', CSV_FILE, 'in folder:', FOLDER)
    
    file = csv.reader(fid, delimiter=',')
    first = True
    
    cnt = 0
    error_cnt = 0
    error = False

    for row in file:
        if error == True:
            error_cnt += 1
            error = False
            
        if first == True:
            first = False
            continue
        
        cnt += 1
        if (cnt)%500==0:
            print('progress: '+str(cnt))
        
        name, width, height, xmin, ymin, xmax, ymax = row[0], int(row[1]), int(row[2]), int(row[4]), int(row[5]), int(row[6]), int(row[7])
        
        path = os.path.join(FOLDER, name)
        img = cv2.imread(path)
        
        
        if type(img) == type(None):
            error = True
            print('Could not read image', img)
            continue
        
        org_height_0, org_width_0 = img.shape[:2]
        org_height= org_height_0
        org_width = org_width_0
        
        if org_width_0 != width:
            error = True
            width=org_width
        
        if org_height_0 != height:
            error = True
            heigth=org_weight
        
        if xmin >= org_width:
            error = True
            xmin = org_width-2
            
        if xmax >= org_width:
            error = True
            xmax = org_width-1
            
        if ymin >= org_height:
            error = True
            ymin = org_height-2
        
        if ymax >= org_height:
            error = True
            ymax = org_height-1
        
        lines.append([name, str(width), str(height), row[3], str(xmin), str(ymin), str(xmax), str(ymax)])
        

with open(CSV_OUTPUT, 'w') as writeFile:
    writer = csv.writer(writeFile)
    writer.writerows(lines)
