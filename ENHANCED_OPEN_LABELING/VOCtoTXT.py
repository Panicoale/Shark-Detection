import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import baker

def nonblank_lines(f):
    for l in f:
        line = l.rstrip()
        if line:
            yield line

CLASS_LIST_PATH = '/home/alessandro/Shark/OPEN-LABELLING/OpenLabeling/main/class_list.txt'
CLASS_LIST_PATH = '/home/alessandro/Shark/SCRIPT/OPEN-LABELLING/OpenLabeling/main/class_list_shark.txt'

with open(CLASS_LIST_PATH) as f:
    classes = list(nonblank_lines(f))

xml_folder = "/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/PASCAL_VOC/Aerial Dolphin Footage San Felipe_  Baja, Mexico_mp4"
txt_folder = "/Users/panicoale/PycharmProjects/OpenLabeling/OpenLabeling/main/output/YOLO_darknet/Aerial Dolphin Footage San Felipe_  Baja, Mexico_mp4"

xml_folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'
txt_folder='/home/alessandro/Shark/darknet-master/build/darknet/x64/data/obj'

xml_folder='/home/alessandro/Shark/00_DATABASE_FILTERED_AUGMENTED'
txt_folder='/home/alessandro/Shark/00_DATABASE_FILTERED_AUGMENTED'

def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(xml_path, txt_path):
    in_file = open(xml_path, 'r')
    out_file = open(txt_path, 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        if obj.find('difficult') == None:
            difficult = 0
        else:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    in_file.close()
    out_file.close()


@baker.command
def create_txt(xml_folder, txt_folder):
    if xml_folder[-1] != '/':
        xml_folder += '/'
    if txt_folder[-1] != '/':
        txt_folder += '/'
    print('Start!')
    wd = getcwd()
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
        print('Create a new folder as txt_folder: {}'.format(txt_folder))
    print('Read xml files in xml_folder: {}'.format(xml_folder))
    print('Store txt files in txt_folder'.format(txt_folder))
    count = 0
    # for image_id in os.listdir(xml_folder):
    #     if image_id[-7:]=='aug.xml':
    #         if image_id[0:1] != '.':
    #             # write list file summary
    #             count += 1
    #             if (count % 500000) == 0:
    #                 print('Already loaded {} .txt files!!'.format(count))
    #             image_id = image_id.replace('.xml', '')
    #             # Convert xml to txt and save in txt_folder
    #             convert_annotation(xml_folder + image_id + '.xml', txt_folder + image_id + '.txt')
    # #print('FINALLY loaded {} .txt files!!'.format(count))
    # #print('Darknet-format txt files are located at: {}'.format(txt_folder))

    for image_id in os.listdir(xml_folder):
        if image_id[-3:]=='xml':
            if image_id[0:1] != '.':
                # write list file summary
                count += 1
                if (count % 500000) == 0:
                    print('Already loaded {} .txt files!!'.format(count))
                image_id = image_id.replace('.xml', '')
                # Convert xml to txt and save in txt_folder
                convert_annotation(xml_folder + image_id + '.xml', txt_folder + image_id + '.txt')
    print('FINALLY loaded {} .txt files!!'.format(count))
    print('Darknet-format txt files are located at: {}'.format(txt_folder))






    #### Delete empty txt files in txt_folder
    # for txt_file in os.listdir(txt_folder):
    #     content = open(txt_folder + txt_file).read()
    #     if content == '':
    #         os.remove(txt_folder + txt_file)





if __name__ == '__main__':
    baker.run()
    create_txt(xml_folder, txt_folder)



# os.system("cat 2007_train.txt 2007_val.txt 2012_train.txt 2012_val.txt > train.txt")
# os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt 2012_train.txt 2012_val.txt > train.all.txt")
