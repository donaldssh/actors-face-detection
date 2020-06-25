import cv2
import time
import numpy as np
import argparse
from vidgear.gears import CamGear


def video_url(argument):
    switcher = {   
        "adam" : lambda : "https://www.youtube.com/watch?v=eghK5yMpNuc",
        "alyssa" : lambda : "https://www.youtube.com/watch?v=dmMbfKnT38k",
        "bruce" : lambda : "https://www.youtube.com/watch?v=-pwvctnQUYM"
    }
    return switcher.get(argument, lambda :  bruce)()


def main():
    
    parser = argparse.ArgumentParser(description='Actor video classifier')
    parser.add_argument('--input', type=str, help='Path to video')  
    args = parser.parse_args()
    
    url_video = video_url(args.input)
    
    classes = ["Adam Sandler", "Alyssa Milano", "Bruce Willis", "Denise Richards",
            "George Clooney", "Gwyneth Paltrow", "Hugh Jackman", "Jason Statham",
            "Jennifer Love Hewitt", "Lindsay Lohan", "Mark Ruffalo", 
            "Robert Downey Jr", "Will Smith"]


    face_cascade = cv2.CascadeClassifier('../data/face_cascade_cv2/haarcascade_frontalface_default.xml')
    net = cv2.dnn.readNetFromONNX("../data/trained_net/CNNNet_13.onnx")
    


    stream = CamGear(source=url_video, y_tube=True).start() # YouTube Video URL as input

    count = 0
    while True:
        frame_normal = stream.read()
        
        if frame_normal is None:   #se sono finiti i frame del video
            print("End of the video")
            break
        
        frame = cv2.resize(frame_normal, None, fx=0.5, fy=0.5)
        

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blobs = []
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            blob = cv2.dnn.blobFromImage(roi_color, 1, (64, 64), (104, 117, 123))
            net.setInput(blob)
            start = time.time()
            preds = net.forward()
            end = time.time()
            print()
            print("[INFO] classification took {:.5} seconds".format(end - start))
            # sort the indexes of the probabilities in descending order (higher
            # probabilitiy first) and grab the top-5 predictions
            idxs = np.argsort(preds[0])[::-1][:5]
            # loop over the top-5 predictions and display them
            for (i, idx) in enumerate(idxs):
                # draw the top prediction on the input image
                if i == 0:
                    text = "Label: {}, {:.2f}%".format(classes[idx],
                        preds[0][idx] * 100)
                    cv2.putText(frame, text, (5, 25),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
                # display the predicted label + associated probability to the
                # console	
                print("[INFO] {}. label: {}, probability: {:.5}".format(i + 1,
                    classes[idx], preds[0][idx]))
                            
        cv2.imshow('videoframe',frame)    
        
        key = cv2.waitKey(10) 
        
        if key == ord('q'): 
            break
        elif key == ord('p'): 
            cv2.waitKey(0)

    cv2.destroyAllWindows() 


if __name__ == "__main__":
    main()
    
    
