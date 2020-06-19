import cv2
import time
import numpy as np

classes = ["Adam Sandler", "Alyssa Milano", "Bruce Willis", "Denise Richards",
           "George Clooney", "Gwyneth Paltrow", "Hugh Jackman", "Jason Statham",
           "Jennifer Love Hewitt", "Lindsay Lohan", "Mark Ruffalo", 
           "Robert Downey Jr", "Will Smith"]

face_cascade = cv2.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
net = cv2.dnn.readNetFromONNX("../../trained-nets/CNNNet_13.onnx");


cap = cv2.VideoCapture("video.mp4")
count = 0
while cap.isOpened():
    _,frame = cap.read()

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
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()  
