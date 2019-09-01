import os
from shutil import copyfile
import ntpath
ntpath.basename("a/b/c")


path='/home/alessandro/Shark/tf/workspace_shark/COMMON_FILES'




TEST='Testing.txt'
destination='test'

TEST='Validation.txt'
destination='validation'

TEST='Training.txt'
destination='train'


element=os.path.join(path,TEST)

def extract(file):
    lineList = [line.rstrip('\n') for line in open(file)]
    return lineList

def move_files(lineList,path,destination):
    destination_folder=os.path.join(path,destination)
    n=0
    for element in lineList:
        head, image_name = ntpath.split(element)
        xml_name=image_name[:-3]+'xml'
        xml_element=element[:-3]+'xml'

        #element2=os.path.join(destination_folder,image_name)
        if os.path.exists(xml_element):
            image_destination = os.path.join(destination_folder, image_name)
            xml_destination = os.path.join(destination_folder, xml_name)
            copyfile(element,image_destination)
            copyfile(xml_element, xml_destination)

linelist=extract(element)
move_files(linelist,path,destination)
