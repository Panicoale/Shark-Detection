import os
import random
from os.path import isfile, join
from shutil import copyfile


#### INPUT - OUTPUT - OTHER ALGORITHM DATA
folder='/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/test_video/'
folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'
folder ='/home/alessandro/Shark/00_DATABASE'
folder ='/home/alessandro/Shark/00_DATABASE_FILTERED_AUGMENTED'
#folder ='/home/alessandro/Shark/00_DATABASE_FILTERED'

output_folder='/home/alessandro/Shark/lists'
# get all elements of the folder


# load class list
CLASS_LIST_PATH='/home/alessandro/Shark/SCRIPT/OPEN-LABELLING/OpenLabeling/main/class_list_shark.txt'

images_dir=os.listdir(folder)
list_txt = []
for element in images_dir:
     if element[-4:] =='.txt':
            if element[0:1] != '.':
                list_txt.append(element)

check=0
n=0
while check==0:
    i = random.randrange(len(list_txt))
    if 'aug.' in list_txt[i]:
        n+=1
    else:
        check=1
elementTXT=list_txt[i]
elementJPG=elementTXT[:-3]+'jpg'
elementXML=elementTXT[:-3]+'xml'
AUGelementTXT=elementTXT[:-4]+'_aug.txt'
AUGelementJPG=AUGelementTXT[:-3]+'jpg'
AUGelementXML=AUGelementTXT[:-3]+'xml'
lista=[elementTXT,elementJPG,elementXML,AUGelementTXT,AUGelementJPG,AUGelementXML]

for element in lista:
    print(element)
    input=join(folder,element)
    output=join(output_folder,element)
    copyfile(input,output)









