import os
from shutil import copyfile
import ntpath

ntpath.basename("a/b/c")

path = '/home/alessandro/Shark/tf/workspace_shark/COMMON_FILES'
TEST = []
destination = []

TEST.append('Training.txt')
destination.append('train')

TEST.append('Testing.txt')
destination.append('test')

TEST.append('Validation.txt')
destination.append('validation')


def jpg_filecheck(xml_file):
    jpg_file = xml_file[-3:]
    if os.path.isfile(jpg_file) == 1:
        return 1
    else:
        return 0


for idx, val in enumerate(TEST):
    print('PROCESSING: ' + val + '\n')
    folder = destination[idx]
    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)

    for file in files:
        if file[-3:] == 'xml':
            file_path = os.path.join(folder_path,file)
            integrity = jpg_filecheck(file_path)
            if integrity == 0:
                os.remove(file_path)
                print('File Removed: ' + file)
