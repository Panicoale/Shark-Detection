import os
#from shutil import copyfile



#input_path='/home/alessandro/Shark/OPEN-LABELLING/completed/video'
#input_path='/home/alessandro/Shark/OPEN-LABELLING/completed'
#input_path='/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/TBEmoded/'
# path='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj/'
path='/home/alessandro/Shark/00_DATABASE_FILTERED'
path='/home/alessandro/Shark'

# for r, d, f in os.walk(path):
#     for file in f:
#         if '.jpg' in file:
#             if file[0:1] != '.':
#                 filepath = os.path.join(r, file)
#
#                 FILE_XML=filepath[:-3]+'xml'
#                 if os.path.isfile(FILE_XML) == 0:
#                     os.remove(filepath)



for r, d, f in os.walk(path):
    for file in f:
        if '_aug.' in file:
            filepath = os.path.join(r, file)
            os.remove(filepath)