# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
import time

import cv2

import face2 as face


def add_overlays(frame, faces):
    res=[]
    if faces is not None:
        for face in faces:
            score=face.score

            face_bb = face.bounding_box.astype(int)

            if face.score is not None:
                if score>1:
                    cv2.rectangle(frame,
                                  (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                                  (0, 0,255), 2)
                    cv2.putText(frame, 'Not OK: '+str(score), (face_bb[0], face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255),
                                thickness=2, lineType=2)
                    res.append((False, face_bb[0], face_bb[1], face_bb[2]-face_bb[0], face_bb[3]-face_bb[1]))

                else:

                    cv2.rectangle(frame,
                                  (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                                  (0,255, 0), 2)
                    cv2.putText(frame, 'OK: '+str(score), (face_bb[0], face_bb[3]),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0),
                                thickness=2, lineType=2)
                    res.append((True, face_bb[0], face_bb[1], face_bb[2]-face_bb[0], face_bb[3]-face_bb[1]))

    return res




def main(args):
    frame_count = 0
    frame_interval = 3
    count_id = 0
    count_smile = 0
    count_no_smile = 0
    count_tot_smile = 0
    count_tot_no_smile = 0
    smilePath = "src/haarcascade_smile.xml"
    smileCascade = cv2.CascadeClassifier(smilePath)

    video_capture = cv2.VideoCapture(0)
    face_verification = face.Verification(args.person)
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (frame_count % frame_interval) == 0:
            faces = face_verification.verify(frame)


        run_time = time.time()-start_time
        if run_time>3 and run_time<6:
            cv2.putText(frame, "Don't smile", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2, lineType=2)

        if run_time>6 and run_time<9:
            cv2.putText(frame, "Smile", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                        thickness=2, lineType=2)

        if run_time>9:
            break



        faces_coor = add_overlays(frame, faces)


        for (verif_id, x, y, w, h) in faces_coor:
            if verif_id:
                count_id+=1
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]


                smile = smileCascade.detectMultiScale(
                    roi_gray,
                    scaleFactor= 1.16,
                    minNeighbors=35,
                    minSize=(25, 25),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if run_time>3 and run_time<6:
                    count_tot_no_smile+=1
                    if len(smile)==0:
                        count_no_smile+=1

                if run_time>6 and run_time<9:
                    count_tot_smile+=1
                    if len(smile)>=1:
                        count_smile+=1



                for (x2, y2, w2, h2) in smile:
                    cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
                    cv2.putText(frame,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
                break


        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        frame_count += 1
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()



    print(count_id/frame_count)
    if count_id/frame_count>=0.7:
        print("Idenity IS TRUE")
    else:
        print("Identity is FALSE")

    print(count_id/frame_count)
    if count_no_smile/count_tot_no_smile>=0.6 and count_smile/count_tot_smile>=0.6 and count_id/frame_count>=0.9:
        print("Idenity has been fully verified with smile detection")





def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    parser.add_argument('--person', help='Name of the person')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
