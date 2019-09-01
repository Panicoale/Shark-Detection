import os
import random
from os.path import isfile, join


#### INPUT - OUTPUT - OTHER ALGORITHM DATA
folder='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/test_video/'
folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'
folder ='/home/alessandro/Shark/00_DATABASE'
folder ='/home/alessandro/Shark/00_DATABASE_FILTERED_AUGMENTED'
#folder ='/home/alessandro/Shark/00_DATABASE_FILTERED'

output_folder='/home/alessandro/Shark/lists'
# get all elements of the folder

Training_ratio=0.7
Validation_ratio=0.10
Testing_ratio=0.20

# load class list
CLASS_LIST_PATH='/home/alessandro/Shark/SCRIPT/OPEN-LABELLING/OpenLabeling/main/class_list_shark.txt'

##### SUPPORT FUNCTIONS

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line


def counter(CLASS_LIST,folder ):

    # print(CLASS_LIST)
    last_class_index = len(CLASS_LIST) - 1
    class_number=[]
    class_null=0
    for i in range(last_class_index+1):
        class_number.append(0)

    images_dir=os.listdir(folder)

# get all elements of the folder

# open all the txt files'
    number_pictures=0
    number_tags=0
    null_files=0
    list_txt = []
    Tag_txt=[]
    for element in images_dir:
        if element[-4:] =='.txt':
            if element[0:1] != '.':
                element_path = join(folder, element[:-3]+'jpg')
                list_txt.append(element_path)
                number_pictures += 1
                #print(element[0:1])
                file = join(folder, element)
                Tag_txt_temp=[0]*11
                with open(file) as f:
                    OBJECT_LIST = list(nonblank_lines(f))
                    if OBJECT_LIST == []:
                        class_null+=1
                        Tag_txt_temp[10]=1
                    else:
                        for number in OBJECT_LIST:
                            if number[1]==' ':
                                class_number[int(number[0])]+=1
                                number_tags+=1
                                Tag_txt_temp[int(number[0])]+=1
                            else:
                                class_number[int(10+number[1])]+=1
                                number_tags += 1
                                Tag_txt_temp[int(10+number[1])] += 1
                Tag_txt.append(Tag_txt_temp)
            else:
                null_files+=1

    return null_files, class_number, class_null, number_pictures, number_tags, list_txt, Tag_txt

def Balanced_Random_Extractor(list_txt,Tag_txt,VALIDATION_TAGS):

    Limit=11
    Validation=0
    Validation_box=[0]*11
    Validation_list=[]
    while Validation < Limit:

        i = random.randrange(len(list_txt))  # get random index
        value_index=-1
        check=0

        for value in Tag_txt[i]:
            if check==0:
                value_index+=1

                if value!=0:
                    if Validation_box[value_index]<VALIDATION_TAGS[value_index]:
                        check=1
        if check==1:
            Validation=0
            for element in range(11):
                Validation_box[element]=Validation_box[element]+Tag_txt[i][element]
                if Validation_box[element]>=VALIDATION_TAGS[element]:
                    Validation+=1

            list_txt[i], list_txt[-1] = list_txt[-1], list_txt[i]  # swap with the last element
            Tag_txt[i], Tag_txt[-1] = Tag_txt[-1], Tag_txt[i]  # swap with the last element
            Validation_list.append(list_txt.pop())
            Tag_txt.pop()



    # print(VALIDATION_TAGS)
    # print(Validation_box)

    return list_txt,Tag_txt, Validation_box, Validation_list



## MAIN FUNCTION:
with open(CLASS_LIST_PATH) as f:
    CLASS_LIST = list(nonblank_lines(f))

null_files, class_number, class_null, number_pictures, number_tags, list_txt, Tag_txt = counter(CLASS_LIST,folder )

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


Images_dictionary = dict(zip(list_txt, Tag_txt))


TRAINING_TAGS=[0]*11
VALIDATION_TAGS=[0]*11
TESTING_TAGS=[0]*11

N=0
for i in class_number_2:
    TRAINING_TAGS[N]=round(class_number_2[N]*Training_ratio)
    TESTING_TAGS[N] = round(class_number_2[N]*Testing_ratio)
    VALIDATION_TAGS[N] = round(class_number_2[N]*Validation_ratio)
    N+=1

list_txt,Tag_txt, Validation_box, Validation_list =  Balanced_Random_Extractor(list_txt,Tag_txt,VALIDATION_TAGS)

list_txt,Tag_txt, Testing_box, Testing_list =  Balanced_Random_Extractor(list_txt,Tag_txt,TESTING_TAGS)

Training_list=list_txt
Training_box=[0]*11
for i in range(len(Tag_txt)):
    n=0
    for j in Tag_txt[i]:
        Training_box[n]+=j
        n+=1

# print(Validation_box)
# print(VALIDATION_TAGS)
#
# print(Testing_box)
# print(TESTING_TAGS)
#
# print(Training_box)
# print(TRAINING_TAGS)


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