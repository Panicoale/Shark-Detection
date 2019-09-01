import os
from shutil import copyfile



#input_path='/home/alessandro/Shark/OPEN-LABELLING/completed/video'
#input_path='/home/alessandro/Shark/OPEN-LABELLING/completed'
#input_path='/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/TBEmoded/'
#input_path='/home/alessandro/Shark/0-DATABASE/'

input_path='/media/alessandro/ULTRA64/2019-07-23/MIXED'
output_path='/home/alessandro/Shark/00-DATABASE'


selection = 0
if selection == 0:
    files = []
    #text='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj/'
    # r=root, d=directories, f = files
    for r, d, f in os.walk(input_path):
        L=len(f)
        n=0
        for file in f:
            n+=1
            x= n % 1000
            if x == 0:
                print (str(n/L*100) + '%')
            if '.jpg' in file:
                element1 = os.path.join(r, file)
                element2 = os.path.join(output_path, file)
                copyfile(element1, element2)
#elif selection == 1:
 #   for r, d, f in os.walk(input_path):
  #      for file in f:
            if '.txt' in file:
                element1 = os.path.join(r, file)
                element2 = os.path.join(output_path, file)
                copyfile(element1, element2)
            if '.xml' in file:
                element1 = os.path.join(r, file)
                element2 = os.path.join(output_path, file)
                copyfile(element1, element2)

elif selection == 2:
    new_input_path_VOC='/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/output/PASCAL_VOC/'
    new_input_path_DARK = '/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/output/YOLO_darknet/'
    for directory in os.listdir(input_path):

        if os.path.isdir(input_path+directory):
            for r, d, f in os.walk(new_input_path_VOC+directory):
                for file in f:
                    if '.txt' in file:
                        element1 = os.path.join(r, file)
                        element2 = os.path.join(output_path, file)
                        copyfile(element1, element2)

                    if '.xml' in file:
                        element1 = os.path.join(r, file)
                        element2 = os.path.join(output_path, file)
                        copyfile(element1, element2)

            for r, d, f in os.walk(new_input_path_DARK+directory):
                for file in f:
                    if '.txt' in file:
                        element1 = os.path.join(r, file)
                        element2 = os.path.join(output_path, file)
                        copyfile(element1, element2)

                    if '.xml' in file:
                        element1 = os.path.join(r, file)
                        element2 = os.path.join(output_path, file)
                        copyfile(element1, element2)

