import os
from shutil import copyfile

input_path='/home/alessandro/Shark/00-DATABASE'
output_path='/home/alessandro/Shark/DATABASE_FILTERED'


# for element in os.listdir(input_path):
#     if element[-3:]=='jpg':
#         if element[0:1] != '.':
#             copyfile(os.path.join(input_path,element),os.path.join(output_path,element))


# create the function for YOLO

for element in os.listdir(output_path):
    if element[-3:]=='xml':
        if element[0:1] != '.':
            copyfile(os.path.join(input_path,element),os.path.join(output_path,element))




# AUGMENT THE DATABASE



