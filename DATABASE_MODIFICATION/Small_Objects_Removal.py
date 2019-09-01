import os
import ntpath
ntpath.basename("a/b/c")
import xml.etree.ElementTree as ET
import math
from lxml import etree
import imageio

path='/home/alessandro/Shark/tf/WS_shark/COM'

input_folder='/home/alessandro/Shark/00-DATABASE'
output_folder='/home/alessandro/Shark/DATABASE_FILTERED'
output_folder_name='DATABASE_FILTERED'
input_folder='/home/alessandro/Shark/DATABASE_FILTERED'

# META-PARAMETERS
ALTITUDE=25
Angle=80
Threshold=0.5
MIN_THRESHOLD=0.1


#output_folder=

name=['shark','whale','dolphin','turtle','ray','swimmer','surfer','boat','rubbish','buoy']
length=[3,12,2.5,0.5,0.5,0.5,1.5,0.01,0.01,0.01]
width=[1,3,0.5,0.5,0.5,0.5,1,0.01,0.01,0.01]
AR=[]
area=[]
L=len(name)
for index in range(L):
    area.append(length[index]*width[index])
    AR.append(length[index]/width[index]*Threshold)
Dictionary=[name,length,width,area,AR]




# shark_size=[6,1]
# whale_size=[15,2]
# dolphin_size=[3,1]
# turtle_size=[0.5,0.5]
# ray_size=[0.5,0.5]
# swimmer_size=[0.5,0.5]
# surfer_size=[2,0.5]
# boat_size=[0,0]
# rubbish_size=[0,0]
# buoy_size=[0,0]



os.chdir(input_folder)


def extract_size_xml(tree):
    annotation = tree.getroot()
    filename = annotation.find('size')
    size=[int(filename[0].text),int(filename[1].text)]
    return size

def write_xml(xml_str, xml_path):
    # remove blank text before prettifying the xml
    parser = etree.XMLParser(remove_blank_text=True)
    root = etree.fromstring(xml_str, parser)
    # prettify
    xml_str = etree.tostring(root, pretty_print=True)
    # save to file
    with open(xml_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

def get_xml_object_data(obj,Dictionary):
    class_name = obj.find('name').text
    class_index = Dictionary[0][:].index(class_name)
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    xmax = int(bndbox.find('xmax').text)
    ymin = int(bndbox.find('ymin').text)
    ymax = int(bndbox.find('ymax').text)
    return [class_name, class_index, xmin, ymin, xmax, ymax]

def append_bb(annotation, object):


    # xmin, ymin, xmax, ymax = bbs_aug.bounding_boxes[i]

    obj = ET.SubElement(annotation, 'object')
    ET.SubElement(obj, 'name').text = object[0]
    ET.SubElement(obj, 'pose').text = 'Unspecified'
    ET.SubElement(obj, 'truncated').text = '0'
    ET.SubElement(obj, 'difficult').text = '0'

    bbox = ET.SubElement(obj, 'bndbox')
    ET.SubElement(bbox, 'xmin').text = str(object[2])
    ET.SubElement(bbox, 'ymin').text = str(object[3])
    ET.SubElement(bbox, 'xmax').text = str(object[4])
    ET.SubElement(bbox, 'ymax').text = str(object[5])

    return annotation


def create_PASCAL_VOC_xml(xml_path, abs_path, folder_name, image_name, img_height, img_width, depth,object_data):
    # By: Jatin Kumar Mandav
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = folder_name
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = abs_path
    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = img_width
    ET.SubElement(size, 'height').text = img_height
    ET.SubElement(size, 'depth').text = depth
    ET.SubElement(annotation, 'segmented').text = '0'

    for obj in object_data:
        append_bb(annotation, obj)

    xml_str = ET.tostring(annotation)
    write_xml(xml_str, xml_path)


def object_check(tree,Dictionary,Expected_Area,Expected_Area_AR, file,counter,Min_Area,size):
    annotation = tree.getroot()
    width=size[0]
    height=size[1]

    object_data = []
    object_data_support = []

    for obj in annotation.findall('object'):
        #print(obj.text)
        class_name, class_index, xmin, ymin, xmax, ymax = get_xml_object_data(obj, Dictionary)

        if xmin >= width:
            error = True
            xmin = width - 2

        if xmax >= width:
            error = True
            xmax = width - 1

        if ymin >= height:
            error = True
            ymin = height - 2

        if ymax >= height:
            error = True
            ymax = height - 1

        Length=xmax-xmin
        Width=ymax-ymin
        OBJ_AREA=Length*Width
        OBJ_AR=max(Length/Width,Width/Length)

        if OBJ_AREA < Expected_Area[class_index]/OBJ_AR*Dictionary[4][class_index]:
            if OBJ_AREA < Min_Area:
                counter[class_index]+=1
                print(class_name)
                print(file + '\n')
            else:
                object_data.append([class_name, class_index, xmin, ymin, xmax, ymax])
        else:
            object_data.append([class_name, class_index, xmin, ymin, xmax, ymax])

    return tree, counter, object_data



### MAIN FUNCTION

counter = []
for i in range(10):
    counter.append(0)

for file in os.listdir(input_folder):
    if file[-3:]=='xml':
        if file[0:1] != '.':
            in_file = open(file, 'r')
            tree = ET.parse(in_file)
            size=extract_size_xml(tree)
            Min_Area=round(size[0]*size[1]*(MIN_THRESHOLD**2))
            #print(Min_Area)

            #IFOV COMPUTATION

            IFOV=[Angle/(size[0]),Angle/(size[1])]
            PIX0=ALTITUDE*(math.tan(math.radians((size[0]/4+1)*IFOV[0]))-math.tan(math.radians(size[0]/4*IFOV[0])))
            PIX1=ALTITUDE*(math.tan(math.radians((size[1]/4+1)*IFOV[1]))-math.tan(math.radians(size[1]/4*IFOV[1])))
            #print(size)
            #print(IFOV)
            #print(PIX0)
            #print(PIX1)
            PIXEL_AREA=PIX1*PIX0


            Expected_Area=[]
            Expected_Area_AR = []
            for i in range(10):
                Expected_Area.append(round(Dictionary[3][i]/PIXEL_AREA*Threshold))
                Expected_Area_AR.append(round(Dictionary[1][i]**2 / PIXEL_AREA * Threshold*0.8))

            #print(Dictionary[3])
            #print(Expected_Area)
            tree_output,counter, object_data = object_check(tree, Dictionary, Expected_Area,Expected_Area_AR, file,counter,Min_Area,size)

            #CREATE THE XML FILE
            # xml_path=os.path.join(output_folder,file)
            # create_PASCAL_VOC_xml(xml_path, output_folder, output_folder_name, file[-3:]+'jpg', str(size[1]), str(size[0]), str(3), object_data)

print(counter)
            # IFOV COMPUTATION








    #
    # filename = annotation.find('filename')
    # filename.text = xml_path[:-3] + 'jpg'
    # xml_str = ET.tostring(annotation)
    # write_xml(xml_str, str(xml_p))







#images = imageio.imread(image_path)


