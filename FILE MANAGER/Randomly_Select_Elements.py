import os
import random
from os.path import isfile, join

folder='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/test_video/'
folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'
folder ='/home/alessandro/Shark/00_DATABASE_FILTERED'
images_dir=os.listdir(folder)

output_folder='/home/alessandro/Shark/'
# get all elements of the folder

# open all the txt files'
list_txt=[]
for element in images_dir:
    if element[-4:]=='.txt':
        if element [0:1] != '.':
            element_path = join(folder, element)
            list_txt.append(element_path)

length_list_txt=len(list_txt)

Training=0
Limit_Training=length_list_txt*0.7
Training_list=[]
while Training < Limit_Training:
    Training+=1
    i = random.randrange(len(list_txt)) # get random index
    list_txt[i], list_txt[-1] = list_txt[-1], list_txt[i]    # swap with the last element
    output = list_txt.pop()                  # pop last element O(1)
    output2=output[:-3]+'jpg'
    if isfile(output2) == 1:
        Training_list.append(output2)


Validation=0
Limit_Validation=length_list_txt*0.15
Validation_list=[]
while Validation < Limit_Validation:
    Validation+=1
    i = random.randrange(len(list_txt))  # get random index
    list_txt[i], list_txt[-1] = list_txt[-1], list_txt[i]    # swap with the last element
    output = list_txt.pop()                  # pop last element O(1)
    # Validation_list.append(output)
    output2 = output[:-3] + 'jpg'
    if isfile(output2) == 1:
        Validation_list.append(output2)

#print(list_txt)
Testing_list=[]
for value in list_txt:
    output = value[:-3] + 'jpg'
    print(output)
    if isfile(output) == 1:
        Testing_list.append(output)


# print(Testing_list)
# print(Validation_list)
# print(Training_list)

Training_path= join(output_folder, 'Training.txt')
Validation_path= join(output_folder, 'Validation.txt')
Testing_path= join(output_folder, 'Testing.txt')

with open(Training_path, 'w') as f:
    for item in Training_list:
        f.write("%s\n" % item)

with open(Validation_path, 'w') as f:
    for item in Validation_list:
        f.write("%s\n" % item)

with open(Testing_path, 'w') as f:
    for item in Testing_list:
        f.write("%s\n" % item)