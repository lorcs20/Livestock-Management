import cv2
import numpy as np
import time

#cap = cv2.VideoCapture('rtmp://192.168.137.1/live/drone') # Live drone feed
#cap = cv2.VideoCapture(0) # Webcam feed
#cap = cv2.VideoCapture('Data/spacedCows.jpg') # jpg file
cap = cv2.VideoCapture('Data/grazing-cows.mp4') # .mp4 file# .mp4 file


frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
size = (frame_width, frame_height)
result = cv2.VideoWriter('resultingVideo.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, size)

## FPS Variables
# used to record the time when we processed last frame
prev_frame_time = 0
# used to record the time at which we processed current frame
new_frame_time = 0

# Detection Variables
# defining width height of input image
whT = 416
# confidence threshold
confidence_threshold = 0.2
# non maximum suppression threshold
nms_threshold = 0.4

# Defining classes for detection
# file containing all classes the yolo model was trained on
classesFile = 'Object-names/obj.names'
# array to store all classes in classesFile
classNames = []
# opening the names file and extracting the classes
# pythons open() function and the 'rt' mode opens and reads a text file
with open(classesFile, 'rt') as f:
    # splitting text file by line and storing in classNames
    classNames = f.read().split('\n')

# yolo model configuration file path
yolo_config = 'YOLOv4-models/COWyolov4-tiny-custom.cfg'
# yolo model trained weights file path
yolo_weights = 'YOLOv4-models/COWyolov4-tiny-custom_best.weights'

# using opencv's deep neural network (dnn) module to create the darknet network
net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)
# setting OpenCV as the backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# using CPU as my pc does not have GPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

##function for finding objects in img
def findObjects(outputs,img):
    # getting height, width and channels of the image
    hT, wT, cT, = img.shape
    # whenever a good detection is found, the values will be stored in these lists
    # list to contain (bounding box) values for cx, cy, w, h
    bbox = []
    # list to contain all class ids
    classIds = []
    # list to contain confidence values
    confs = []
    rectangles = []

    # we have 2 output layers
    for output in outputs:
        # for loop on detections in output (outputs contain the array of values for each detection)
        for det in output:
            # removing the first 5 (cx, cy, w, h, conf) elements in numpy arrays to access the class id probabilities
            scores = det[5:]
            # define classID as the class with the largest probability that its being detected
            classId = np.argmax(scores)
            # confidence is the class with the max probability
            confidence = scores[classId]
            # filtering the classes
            # if confidence is greater than the threshold save the detection
            if confidence > confidence_threshold:
                #save the w and h values (3rd and 4th element in the numpy arrays), multiply by actual w and h of the image to get (int)pixel values
                w,h = int(det[2]*wT) , int(det[3]*hT)
                # save the cx and cy values (1st and 2nd element in the numpy arrays), divide w and h by 2 and subtract from cx,cy to get centre point of image, (int) pixel values
                cx,cy = int((det[0]*wT) - w/2), int((det[1]*hT) - h/2)
                # add the good detections to the bbox list
                bbox.append([cx,cy,w,h])
                # add the classID of the detection to the classIDs list
                classIds.append(classId)
                # add the confidence scores to the confs list
                confs.append(float(confidence))

    print(len(bbox))
    ## which of the bounding boxes we want to keep
    indices = cv2.dnn.NMSBoxes(bbox,confs,confidence_threshold,nms_threshold)

    for i in indices:
        # take first element, i.e. remove square brackets
        i = i[0]
        box = bbox[i]
        cx,cy,w,h = box[0],box[1],box[2],box[3]
        # drawing rectangles for highlighting detections in the image
        rectangle = cv2.rectangle(img,(cx,cy),(cx+w,cy+h),(255,0,255),2)
        # storing rectangles drawn in an array for counting
        rectangles.append(rectangle)
        #num_of_rectangles = len(rectangles)
        # writing text: image window, output: class name and confidence score *100, position (above box), font, font-scale, color, thickness
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',(cx+50,cy-10),cv2.FONT_HERSHEY_COMPLEX,0.4,(255,0,255),1)
    cv2.putText(img, 'Livestock Count: ' + str(len(rectangles)), (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)


#while loop that gets the frames of our cam
while True:
    success, img = cap.read()
    if success == True:
        # Displaying Detections

        # the dnn only supports blob format so we convert the input image to blob
        blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
        net.setInput(blob)

        ## getting names of all layers in the network
        layerNames = net.getLayerNames()
        #print(layerNames)
        ## printing the index of the yolo layers. getUnconnectedOutLayers retrieves indexes of all layers with unconnected output
       # print(net.getUnconnectedOutLayers())
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        # print the yolo layer names, note: yolo-tiny has only 2 yolo layers, while normal yolo has 3
        #print(outputNames)

        outputs = net.forward(outputNames)
        # the output layers are lists of numpy arrays (matrices)
        #print(outputs[0].shape) ## (507, 6) matrices of 507 rows and 6 columns, the 6 values is num of classes + 5.. the 5 extra elements are cx, cy, w, h, conf
        #print(outputs[1].shape)  ## (2028, 6) 507 & 2028 are the bounding boxes
        findObjects(outputs,img)

        # Calculating the fps

        # time the frame finished processing
        new_frame_time = time.time()
        # fps is number of frame processed in given time
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # fps into integer
        fps = int(fps)
        # fps to string for putText()
        fps = str(fps)
        # configuring putText function: window, output, pos, font, font-scale, color, thickness
        cv2.putText(img, 'FPS: '+fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 0), 2)

        
        result.write(img)
        cv2.imshow('webcam', img)
        #delay cam for 1 milisec, and e to exit
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

    else:
        break
cap.release()
result.release()

# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")