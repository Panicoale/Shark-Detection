import os

path='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'


files = []
text='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj/'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.jpg' in file:

            files.append(text+file)

for f in files:
    print(f)

with open('train2.txt', 'w') as f:
    for item in files:
        f.write("%s\n" % item)