import os
from os.path import isfile, join

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

# load class list
CLASS_LIST_PATH='/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/class_list.txt'

with open(CLASS_LIST_PATH) as f:
    CLASS_LIST = list(nonblank_lines(f))
# print(CLASS_LIST)
last_class_index = len(CLASS_LIST) - 1
class_number=[]
class_null=0
for i in range(last_class_index+1):
    class_number.append(0)

#folder='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/test_video/'
folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj/'
folder='/home/alessandro/Shark/RAW_DATABASE/'
folder='/home/alessandro/Shark/00-DATABASE'
folder='/home/alessandro/Shark/00_DATABASE_FILTERED_AUGMENTED'
folder='/home/alessandro/Shark/00-DATABASE'
images_dir=os.listdir(folder)

# get all elements of the folder

# open all the txt files'
number_pictures=0
number_tags=0
null_files=0
for element in images_dir:
    if element[-4:] =='.txt':
        if element[0:1] != '.':
            number_pictures += 1
            #print(element[0:1])
            file = join(folder, element)
            with open(folder+element) as f:
                OBJECT_LIST = list(nonblank_lines(f))
                if OBJECT_LIST == []:
                    class_null+=1
                else:
                    for number in OBJECT_LIST:
                        if number[1]==' ':
                            class_number[int(number[0])]+=1
                            number_tags+=1
                        else:
                            class_number[int(10+number[0])]+=1
                            number_tags += 1
        else:
            null_files+=1

#print(class_number)
#print(CLASS_LIST)
CLASS_LIST_2=CLASS_LIST
CLASS_LIST_2.append('background')
class_number_2=class_number
class_number_2.append(class_null)

dictionary = dict(zip(CLASS_LIST_2, class_number_2))
print('\n###TAGS DISTRIBUTION###\n')
for key, car in dictionary.items():
    print('{} : {}'.format(key, car))
print('\n###DATABASE STATISTICS###\n')
print('The total number of images is: ' + str(number_pictures) +'\n')
print('The total number of tags is: ' + str(number_tags)+'\n')
print('The total number of tags and background images (no tags) is: ' + str(number_tags + class_null)+'\n')

print(len(images_dir))
print(null_files)