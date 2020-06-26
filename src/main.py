import cv2
import numpy as np
import argparse
from vidgear.gears import CamGear

# load my functions
from video_utils import video_classifier, video_url


def main():
    
    # extract the video url
    parser = argparse.ArgumentParser(description='Actor video classifier')
    parser.add_argument('--input', type=str, help='Path to video')  
    args = parser.parse_args()
    
    # set the video url
    url = video_url(args.input)
    
    # load the classes used in our CNN
    classes = [ "Adam Sandler", "Alyssa Milano", "Bruce Willis", "Denise Richards",
                "George Clooney", "Gwyneth Paltrow", "Hugh Jackman", "Jason Statham",
                "Jennifer Love Hewitt", "Lindsay Lohan", "Mark Ruffalo", 
                "Robert Downey Jr", "Will Smith" ]


    # load the trained CascadeClassifier for face detection
    face_cascade = cv2.CascadeClassifier('../data/face_cascade_cv2/haarcascade_frontalface_default.xml')
    
    # load the CNN trained with matlab
    net = cv2.dnn.readNetFromONNX("../data/trained_net/CNNNet_13.onnx")
    
    # load the youtube video
    stream = CamGear(source=url, y_tube=True).start() # YouTube Video URL as input

    # execute the detection and classification of the video
    video_classifier(face_cascade, net, stream, classes)
    
    # close the stream
    stream.stop()

if __name__ == "__main__":
    main()
