import cv2
import time
import numpy as np
from vidgear.gears import CamGear

# switcher used to get the corresponding youtube video url from the given input name
def video_url(argument):
    switcher = {   
        "adam":     lambda : "https://www.youtube.com/watch?v=V4LxfzjRYQY",
        "alyssa":   lambda : "https://www.youtube.com/watch?v=vSMC6lLfgLI",
        "bruce":    lambda : "https://www.youtube.com/watch?v=-pwvctnQUYM",
        "denise":   lambda : "https://www.youtube.com/watch?v=6sb0Ii0EkUY",
        "george":   lambda : "https://www.youtube.com/watch?v=0t1-Jy3UNRY",
        "gwyneth":  lambda : "https://www.youtube.com/watch?v=Eog5RGbqgKQ",
        "hugh":     lambda : "https://www.youtube.com/watch?v=vJdLROysHHs",  
        "jason":    lambda : "https://www.youtube.com/watch?v=ehDVAfH9038",
        "jennifer": lambda : "https://www.youtube.com/watch?v=xt1bCqZaD0k",
        "lindsay":  lambda : "https://www.youtube.com/watch?v=F6g85lp2wJc", 
        "mark":     lambda : "https://www.youtube.com/watch?v=dmwO8tGHvdo", 
        "robert":   lambda : "https://www.youtube.com/watch?v=w5cu7y6xyMw", 
        "will":     lambda : "https://www.youtube.com/watch?v=YsfYyWc_BfE", 
    }
    return switcher.get(argument, lambda :  argument)()


def video_classifier(face_cascade, net, stream, classes):
 
    arr_classes = []
    
    # number of consecutive frames used to compute the class of the detected face
    n_consecutive_frames = 20
    idframe = 0
    # loop over all the video frames
    while True:
        
        # read the next frame
        frame = stream.read()
        
        if frame is None:   
            print("End of the video")
            break
        
        idframe += 1
        
        # resize the frame 
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
            
        # compute the grayscale image --> for the cascade classifier
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # detect the faces with cascade classifier
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # loop over all the detected faces
        for (x,y,w,h) in faces:
            
            # wh = int((w + h) / 2)
            # frame = cv2.rectangle(frame,(x,y),(x+wh,y+wh),(0,255,0),2)
            # roi_color = frame[y:y+wh, x:x+wh]
            
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            roi_color = frame[y:y+h, x:x+w]
            
            blob = cv2.dnn.blobFromImage(roi_color, 1, (64, 64))
            net.setInput(blob)
            start = time.time()
            preds = net.forward()
            end = time.time()
            print()
            print("[INFO] classification took {:.5} seconds".format(end - start))
            
            # find the index of the class with higher probabilitiy
            idx = np.argsort(preds[0])[::-1][0]


            if idframe % n_consecutive_frames:
                arr_classes.append(classes[idx])
                text_processing_info = "Processing"
                cv2.putText(frame, text_processing_info, (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               
                
            else:
                # print(arr_classes)
                if len(arr_classes) > 0:
                    predicted = max(set(arr_classes), key = arr_classes.count)
                    arr_classes = []

                    cv2.putText(frame, predicted, (x, y+1),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    text_processing_info = "Press any key to continue"
                    cv2.putText(frame, text_processing_info, (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
               

                    # display the frame with the classes  
                    cv2.imshow('videoframe',frame)   
            
                    # stop the video till one key is pressed, to display the current label
                    cv2.waitKey(0)
          
        cv2.imshow('videoframe',frame) 
        key = cv2.waitKey(10) 
        # quit if q is pressed, or pause if p is pressed
        if key == ord('q'): 
            break
        elif key == ord('p'): 
            cv2.waitKey(0)

    cv2.destroyAllWindows() 
