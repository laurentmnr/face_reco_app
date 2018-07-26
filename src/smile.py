# -*- coding=utf-8 -*-
import cv2


facePath = "src/lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(facePath)

smilePath = "src/haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smilePath)


def main():
    video_capture = cv2.VideoCapture(0)


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()



        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor= 1.1,
            minNeighbors=8,
            minSize=(55, 55),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]


            smile = smileCascade.detectMultiScale(
                roi_gray,
                scaleFactor= 1.16,
                minNeighbors=35,
                minSize=(25, 25),
                flags=cv2.CASCADE_SCALE_IMAGE
            )


            for (x2, y2, w2, h2) in smile:
                cv2.rectangle(roi_color, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
                cv2.putText(frame,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break

        cv2.namedWindow("Video", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Video",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    main()
