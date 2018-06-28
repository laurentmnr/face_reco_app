from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import lfw
import os
import sys
from tensorflow.python.ops import data_flow_ops
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import cv2

def main(args):
    curr_dir=os.getcwd()
    path=os.path.join(curr_dir,args.output_dir,args.person)
    if not os.path.exists(path):
        print(path)
        os.mkdir(path)
    else:
        print("Error: Name already in database")
        return

    cam = cv2.VideoCapture(0)
    cam.set(3,args.image_size);
    cam.set(4,args.image_size);
    cv2.namedWindow("test")

    img_counter = 0
    curr_dir=os.getcwd()
    while img_counter<args.n_pictures:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed

            img_name = os.path.join(curr_dir,args.output_dir,args.person,args.person+str(img_counter)+".png")
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1


    cam.release()

    cv2.destroyAllWindows()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('output_dir', type=str,
        help='Path to the output directory')
    parser.add_argument('--n_pictures', type=int,
        help='Number of pictures to take', default=1)
    parser.add_argument('--person', type=str,
        help='Name of the person',default="person")
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=250)

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
