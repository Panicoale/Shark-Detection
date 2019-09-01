import os
from shutil import copyfile

from os import listdir
from os.path import isfile, join


#input_path='/home/alessandro/Shark/OPEN-LABELLING/completed/video'
input_path='/home/alessandro/Shark/darknet-master/build/darknet/x64/data'
output_path='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'

onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]

n=0
for f in listdir(input_path):
    #print(f)
    f_path = join(input_path, f)
    if isfile(f_path)==1:
        #print(f_path[0:3])
        if f[0:3] == 'obj':
            new_f=f[3:]
           # print(new_f)
            dst=join(output_path, new_f)
            #print(f_path)
            #print(dst)
            #os.rename(f_path, dst)
            n+=1
print(n)
