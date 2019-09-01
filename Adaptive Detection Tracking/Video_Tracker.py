import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util

from skimage.measure import compare_ssim
import skimage
import argparse
import imutils
import cv2
import numpy as np

import cv2
import time
import ntpath
ntpath.basename("a/b/c")




#### INITIAL SETUP

CASE='VIDEO'  ## VIDEO, CAMERA, IMAGE, IMAGES

if CASE=='VIDEO':
    input_path='/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/VIDEO_TEST/Drone_Shark_footage.mp4'   # IDENTIFY THE VIDEO TO BE PROCESSED
    input_path='/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/VIDEO_TEST/SURFER_SHARK.mp4'
    input_path='/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/VIDEO_TEST/D2.mp4'
    OUTPUT_DIR='/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/VIDEO_TEST'

### save the json file
    TRACKER_DIR = os.path.join(OUTPUT_DIR, 'tracker')

    #tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    trackerType='KCF'
    trackerType='MEDIANFLOW'
    trackerType='CSRT'

elif CASE=='IMAGE':
    input_path=''   # IDENTIFY THE IMAGE TO BE PROCESSED
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)
    
elif CASE=='IMAGES':
    input_path=''   ## IDENTIFY THE FOLDER IN WHICH THE IMAGES ARE LOCATED
    # Size, in inches, of the output images.
    IMAGE_SIZE = (12, 8)

##faster rcnn resne101
MODEL_PATH = '/home/mueavi-pc-01/Desktop/tf/WS_shark/FASTER_RCNN_RESNET101_COCO/INF_GRAPHS/FASTER_RCNN_RESNET101_400000'

## RESNET50
# MODEL_PATH = 'FASTER_RCNN_RESNET50.pb'


##faster SSD inception
#MODEL_PATH = '/home/mueavi-pc-01/Desktop/tf/WS_shark/SSD_INCEPTION/trained-inference-graphs/output_inference_graph_v2.pb'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_PATH + '/frozen_inference_graph.pb'


# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = '/home/mueavi-pc-01/Desktop/tf/WS_shark/COM/annotation/label_map.pbtxt'

NUM_CLASSES=10

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


###TRACKER PART

tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
if tracker_type == 'BOOSTING':
    tracker = cv2.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv2.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv2.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv2.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv2.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv2.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv2.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv2.TrackerCSRT_create()

trackerTypes = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
def createTrackerByName(trackerType):
    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]: 
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
        print('Available trackers are:')
        for t in trackerTypes:
            print(t)
        
    return tracker
        






# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)



# 
# cap = cv2.VideoCapture(0)
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# while(True):
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # FOR READING IN GRAY
#     out.write(frame)
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[1], image.shape[2])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.int64)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# def tracker_function:
#     # get list of objects associated to that frame
#     #object_list = img_objects[:]
#     # remove the objects in that frame that are already in the `.json` file
#     json_file_path = '{}.json'.format(os.path.join(TRACKER_DIR, video_name))
#     file_exists, json_file_data = get_json_file_data(json_file_path)
#     if file_exists:
#         object_list = remove_already_tracked_objects(object_list, img_path,
#                                                     json_file_data)

#         object_list = remove_new_objects(object_list, img_path,
#                                                     json_file_data)
#     print(object_list)

#     if len(object_list) > 0:
#         # get list of frames following this image
#         frames=1
#         next_frame_path_list = get_next_frame_path_list_limit(video_name, img_path,frames)
#         # initial frame
#         init_frame = img.copy()
#         label_tracker = LabelTracker('CSRT', init_frame,
#                                     next_frame_path_list)  # TODO: replace 'KCF' by 'CSRT'
#         control = 0

#         for obj in object_list:
#             control = control + 1
#             class_index = obj[0]
#             color = class_rgb[class_index].tolist()
#             label_tracker.start_tracker(json_file_data, json_file_path, img_path,
#                                         obj, color, annotation_formats)
#             if control > 40:
#                 break
                   
 
        


CASE='VIDEO'
if CASE =='VIDEO':
    FPS=0
    cap = cv2.VideoCapture(input_path)
    head, Video_name = ntpath.split(input_path)
    output_path=os.path.join(head,Video_name[:-4]+'_output.avi')
    try:
        #vidwrite = cv2.VideoWriter(['testvideo', cv2.CV_FOURCC('M','J','P','G'), 25, 
               #(640,480),True])
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        out = cv2.VideoWriter(output_path,fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))
        im_width = int(cap.get(3))
        im_height = int(cap.get(4))
        im_width_box = round(im_width*0.035)
        im_height_box = round(im_height*0.035)
        # print(im_height_box)
        # print(im_width_box)
        # input('test')
        n_frames=0
        start=time.time()
        original_interval=100
        interval=original_interval
        n_frames2=0
        FPS_TRACK=0
        FPS_DETECT=0
        FPS=0
        boxes=[]
        TRACK=0
        DETECT=0
        IOU=1.0
        

        while(cap.isOpened()):

            init_lap=time.time()
            #print(n_frames)
            ret, image_np = cap.read()
            if n_frames==0:
                grayA = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                #grayA = image_np
                
            #ok = tracker.init(frame, bbox)

            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            grayB = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            #grayB =image_np
            #(score, diff) = compare_ssim(grayA, grayB, full=True,multichannel=True)
            (score, diff) = compare_ssim(grayA, grayB, full=True)

            diff2=skimage.measure.block_reduce(diff, (im_width_box,im_height_box), np.max)
            MIN=[]
            MEAN2=[]
            for i in diff2:
                MIN.append(min(i))
            minimum_cell=min(MIN)
            
            for i in diff:
                MEAN2.append(np.mean(i))

            Mean=np.mean(MEAN2)
            #Mean=1
            #print(Mean)
            #print(score)
            #input('wait')
            # print(minimum_cell)
            # print(minimum)
            # print(MIN)
            # print(diff2)
            # print(diff)
            # print(len(diff))
            # print(len(diff[0]))
            
            #input('test')
            #print(score)
            #print(diff)

            if n_frames2%interval == 0 or Mean<0.6 or minimum_cell <0.45 or IOU <0.5:
                TRACK=0
                DETECT=1
                n_frames2=1
                print(score)
                print(minimum_cell)
                if IOU < 0.5:
                    IOU=1.0
                
                #APPLY THE DETECTION
                output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
                #print(output_dict['detection_boxes'])
                #print(len(output_dict['detection_boxes']))
                #input('test')

                Class=output_dict['detection_classes']
                Boxes = np.squeeze(output_dict['detection_boxes'])
                Scores = np.squeeze(output_dict['detection_scores'])
                              
                final_box = []
                y=0
                for box in Boxes:
                    ymin, xmin, ymax, xmax = box
                    final_box.append([xmin * im_width, ymin * im_height, xmax * im_width, ymax * im_height])
                    #final_box.append([xmin * im_width, ymin * im_height,xmax * im_width, ymax * im_height, 1,1,Class[y]])
                    y+=1
                Boxes=final_box
                BBox=[]
                for element in Boxes:
                    if element[0]+element[1]+element[2]+element[3]>1:
                        BBox.append(element)
                #print(BBox)
                #print(tuple(BBox))
                final_bbox=[]
                for element in BBox:
                    temp=[element[0],element[1],element[2]-element[0],element[3]-element[1]]
                    #print(temp)
                    element2=tuple(temp)
                    #print(element2)
                    final_bbox.append(element2)
                

                multiTracker = cv2.MultiTracker_create()
 
                    # Initialize MultiTracker 

                for bbox in final_bbox:
                    multiTracker.add(createTrackerByName(trackerType), image_np, bbox)    
                #ok=tracker.init(image_np, final_bbox)


                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)
                
                grayA = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
                

            else:
                test_box=0
            
                if DETECT==1:
                    TRACK=1
                    DETECT=0
                else:
                    test_box=1
                    old_boxes=boxes

                n_frames2+=1
                #print(n_frames2)
                ok, boxes = multiTracker.update(image_np)
                
                if test_box==1:
                    
                    IOU=[]
                    for i, newbox in enumerate(boxes):
                        
                        newbox2=old_boxes[i]
                        x1 = int(newbox[0])
                        y1 =int(newbox[1])
                        x2 = int(newbox[0] + newbox[2])
                        y2 = int(newbox[1] + newbox[3])
                        X1 = int(newbox2[0])
                        Y1 =int(newbox2[1])
                        X2 = int(newbox2[0] + newbox2[2])
                        Y2 = int(newbox2[1] + newbox2[3])
                        OVERLAP=(min(x2,X2)-max(x1,X1))*(min(y2,Y2)-max(y1,Y1))
                        A1=(x2-x1)*(y2-y1)
                        A2=(X2-X1)*(Y2-Y1)
                        UNION=A1+A2-OVERLAP
                        if min(A1,A2) < 400:
                            IOU.append(OVERLAP/UNION+0.1)
                        else:
                            IOU.append(OVERLAP/UNION)
                    if IOU==[]:
                        IOU=1.0
                        interval=35
                    else:
                        IOU=min(IOU)
                        interval=original_interval
                    
                    #print(IOU)
                #ok, bbox = tracker.update(image_np)
                out_boxes=[]
                for i, newbox in enumerate(boxes):
                    p=[]
                    p.append(int(newbox[1]) / im_height)
                    p.append(int(newbox[0])/im_width)
                    
                    
                   #p.append((int(newbox[2])) / im_width)
                    
                    p.append((int(newbox[1]) + int(newbox[3])) / im_height)
                    p.append((int(newbox[0]) + int(newbox[2])) / im_width)
                    #p.append((int(newbox[3])) / im_height)
                    output_dict['detection_boxes'][i]=p
                    
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image_np,
                    output_dict['detection_boxes'],
                    output_dict['detection_classes'],
                    output_dict['detection_scores'],
                    category_index,
                    instance_masks=output_dict.get('detection_masks'),
                    use_normalized_coordinates=True,
                    line_thickness=8)



                # for i, newbox in enumerate(boxes):
                #     p1 = (int(newbox[0]), int(newbox[1]))
                #     p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
                #     cv2.rectangle(image_np, p1, p2, (255,0,0), 2, 1)

                # if ok:
                #     # Tracking success
                #     p1 = (int(bbox[0]), int(bbox[1]))
                #     p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                #     cv2.rectangle(image_np, p1, p2, (255,0,0), 2, 1)


                #dets_arr, labels_arr = model.test(raw_frame)
                
            #elif n_frames%interval != 0:
                #APPLY THE TRACKING
                #dets_arr, labels_arr = np.array([]), np.array([])
                

            grayA = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)


            # Actual detection.
            #output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            # Visualization of the results of a detection.
            end_lap=time.time()
            LAP_FPS=1/(end_lap-init_lap)
            FPS=round(n_frames/(end_lap-start),2)


            if TRACK==1:
                FPS_TRACK=round(LAP_FPS, 2)
            else:
                FPS_DETECT=round(LAP_FPS, 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image_np, 'GLOBAL SCORE: '+str(round(Mean,2)),(10,im_height-170), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(image_np, 'LOCAL SCORE: '+str(round(minimum_cell,2)),(10,im_height-130), font, 1,(0,0,255),2,cv2.LINE_AA)
            
            cv2.putText(image_np, 'TRACK FPS: '+str(FPS_TRACK),(10,im_height-90), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(image_np, 'DETECT FPS: '+str(FPS_DETECT),(10,im_height-50), font, 1,(0,0,255),2,cv2.LINE_AA)
            cv2.putText(image_np, 'FPS: '+str(FPS),(10,im_height-10), font, 1,(0,0,255),2,cv2.LINE_AA)

            out.write(image_np)
            #cv2.imshow('object detection', cv2.resize(image_np, (640,480)))
            



            cv2.imshow('object detection', image_np)
            
            n_frames+=1




            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                end=time.time()
                seconds=end - start
                FPS= n_frames/seconds
                break
        end=time.time()
        seconds=end - start
        FPS= n_frames/seconds
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print('THERE IS AN ERROR')
        print(e)
        cap.release()
        out.release()
    print('n_frames: ' + str(n_frames))
    print('FPS: ' + str(FPS))
    # video_fps = capture.get(cv2.CAP_PROP_FPS)   ### CHECK THIS METHOD


if CASE =='CAMERA':
    cap = cv2.VideoCapture(0)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
        while True: 
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
            ret, image_np = cap.read()
            
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=8)

            out.write(image_np)
            cv2.imshow('object detection', cv2.resize(image_np, (640,640)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
    except Exception as e:
        print(e)
        cap.release()

    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)

if CASE=='IMAGES':
    for image_path in input_path:
        image = Image.open(image_path)
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        plt.figure(figsize=IMAGE_SIZE)
        plt.imshow(image_np)


if CASE=='IMAGE':
    image = Image.open(input_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
     # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)
    

    


