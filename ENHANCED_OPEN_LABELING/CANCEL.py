import os
import sys
import xml.etree.ElementTree as ET

path="/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/PASCAL_VOC/Drone Footage Captures Tiger Shark Roaming Close to Swimmers in Miami's South Beach Shore_mp4"
path2="/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/YOLO_darknet/Drone Footage Captures Tiger Shark Roaming Close to Swimmers in Miami's South Beach Shore_mp4"

path="/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/PASCAL_VOC/Aerial Dolphin Footage San Felipe_  Baja, Mexico_mp4"
path2="/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/YOLO_darknet/Aerial Dolphin Footage San Felipe_  Baja, Mexico_mp4"



N=1244
case=2
if case==1:
    os.chdir(path)
else:
    os.chdir(path2)

def convert(list):
    # Converting integer list to string list
    s = [str(i) for i in list]

    # Join list items using join()
    res = int("".join(s))

    return (res)


#q=['4', '_', '4', '2', '.', 'x', 'm', 'l']
qq=[]
for (dirpath, dirnames, f) in os.walk(path):

    for file in f:
        file2=list(file)
        q = file2[-8:]
       # print(q)
        q2=q
        del q[-4:]

        q3=q
        if q[3]=='_':
            del q[0:4]
            q4=q
        elif q[2]=='_':
            del q[0:3]
            q4=q
        elif q[1]=='_':
            del q[0:2]
            q4=q
        elif q[0]=='_':
            del q[0:1]
            q5 = q
        numbers = convert(q)
        qq.append(numbers)
        if numbers > N:
            if case==1:
                tree = ET.parse(file)
                root = tree.getroot()
                foos = tree.findall('object')
                for foo in foos:
                    root.remove(foo)

                tree.write(file)
            else:
                    file2[-3:] = 'txt'
                    file3 = ''.join(file2)
                    f = open(file3, 'w')
                    f.write('')
                    f.close()
