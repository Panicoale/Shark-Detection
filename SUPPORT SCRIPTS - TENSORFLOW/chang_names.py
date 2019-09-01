import os
import ntpath
ntpath.basename("a/b/c")
import xml.etree.ElementTree as ET
from lxml import etree

path='/home/alessandro/Shark/tf/WS_shark/COM'

TEST = []
destination = []

TEST.append('Training.txt')
destination.append('train')

TEST.append('Testing.txt')
destination.append('test')

TEST.append('Validation.txt')
destination.append('validation')

def jpg_filecheck(xml_file):
    jpg_file = xml_file[:-3]+'jpg'

    if os.path.exists(jpg_file) == 1:
        return 1
    else:
        return 0


def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)



def change_filename_xml(folder_path,xml_path):
    #print(xml_path)
    xml_p=os.path.join(folder_path,xml_path)
    print(xml_p)
    in_file = open(xml_p, 'r')
    tree = ET.parse(in_file)
    root = tree.getroot()
    annotation = tree.getroot()
    filename=annotation.find('filename')
    filename.text=xml_path[:-3]+'jpg'
    xml_str = ET.tostring(annotation)
    write_xml(xml_str, str(xml_p))




n=0
for idx, val in enumerate(TEST):
    print('PROCESSING: ' + val + '\n')
    folder = destination[idx]
    folder_path = os.path.join(path, folder)
    os.chdir(folder_path)


    files = os.listdir(folder_path)

    for file in files:


        if file[-3:] == 'xml':
            file_path = os.path.join(folder_path,file)

            integrity = jpg_filecheck(file_path)
            if integrity == 1:
                n += 1
                file_img=file[:-3]+'jpg'
                file_xml=file
                new_name=str(n)

                os.rename(file_img, new_name+'.jpg')
                os.rename(file_xml, new_name + '.xml')
                change_filename_xml(folder_path,new_name+'.xml')