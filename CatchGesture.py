import cv2
import Preprocessing as pre

size = 0.5  # set font size
font = cv2.FONT_HERSHEY_SIMPLEX  # set font
cnt = 1
width, height = 300, 300  # catching window size
x0, y0 = 300, 100  # window left bottom position
cap = cv2.VideoCapture(0)  # open camera

if __name__ == "__main__":
    # loop
    while (1):
        ret, frame = cap.read()  # load content from camera, return bool value and the image
        frame = cv2.flip(frame, 2)  # mirror
        # process the image for better recognition
        gesture,res,ret,fourier_result = pre.binaryMask(frame, x0, y0, width, height)
        cv2.imshow("gesture", gesture)
        cv2.imshow("res", res)
        cv2.imshow("ret", ret)
        key = cv2.waitKey(1) & 0xFF  # adjust the window
        # press 'a''d''w''s'to move the window to left/right/up/down, press 'q' to quit
        if key == ord('s'):
            y0 += 5
        elif key == ord('w'):
            y0 -= 5
        elif key == ord('d'):
            x0 += 5
        elif key == ord('a'):
            x0 -= 5
        elif key == ord('q'):
            break
        elif key == ord('z'):
            path = './' + 'images' + '/'
            name = str(cnt)
            cv2.imwrite(path + name+'.png',res)
            cnt += 1


        cv2.imshow('frame', frame)  # show the content loaded from camera
    cap.release()
    cv2.destroyAllWindows()  # close all windows
