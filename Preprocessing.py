import cv2
import numpy as np
import FourierDescriptor as fd
import time


def binaryMask(frame, x0, y0, width, height):
	cv2.rectangle(frame,(x0,y0),(x0+width, y0+height),(0,0,255)) # write a red frame of the window
	gesture = frame[y0:y0+height, x0:x0+width] # obtain the frame in the window
	res = skinMask(gesture) # skin detection
	kernel = np.ones((3,3), np.uint8)  # convolution kernel
	erosion = cv2.erode(res, kernel)
	dilation = cv2.dilate(erosion, kernel)
	ret, fourier_result = fd.FourierDescriptor(dilation)
	return gesture, dilation, ret, fourier_result


# based on RGB
# def skinMask(gesture):
# 	rgb = cv2.cvtColor(gesture, cv2.COLOR_BGR2RGB) # transform into RGB space
# 	(R,G,B) = cv2.split(rgb) # acquire value of every pixel. convert data from 2 dim into 3 dim(RGB)
# 	skin = np.zeros(R.shape, dtype = np.uint8) # mask
# 	(x,y) = R.shape # pixel range
# 	for i in range(0, x):
# 		for j in range(0, y):
# 			# if the value is not in the judging range, assign 255, black
# 			if (abs(R[i][j] - G[i][j]) > 15) and (R[i][j] > G[i][j]) and (R[i][j] > B[i][j]):
# 				if (R[i][j] > 95) and (G[i][j] > 40) and (B[i][j] > 20) \
# 						and (max(R[i][j],G[i][j],B[i][j]) - min(R[i][j],G[i][j],B[i][j]) > 15):
# 					skin[i][j] = 255
# 				elif (R[i][j] > 220) and (G[i][j] > 210) and (B[i][j] > 170):
# 					skin[i][j] = 255
# 	res = cv2.bitwise_and(gesture,gesture, mask = skin) # get rid of black area
# 	return res

# based on HSV
# def skinMask(gesture):
# 	low = np.array([0, 48, 50]) # low boundary
# 	high = np.array([20, 255, 255]) # up boundary
# 	hsv = cv2.cvtColor(gesture, cv2.COLOR_BGR2HSV) # transform into HSV space
# 	mask = cv2.inRange(hsv,low,high) # mask
# 	res = cv2.bitwise_and(gesture,gesture, mask = mask) # get rid of black area
# 	return res

# based on ellipse
# def skinMask(gesture):
# 	skinCrCbHist = np.zeros((256,256), dtype= np.uint8)
# 	cv2.ellipse(skinCrCbHist, (113,155),(23,25), 43, 0, 360, (255,255,255), -1) # draw an ellipse model
# 	YCrCb = cv2.cvtColor(gesture, cv2.COLOR_BGR2YCR_CB) # transform into YCrCb space
# 	(y,Cr,Cb) = cv2.split(YCrCb) # split Y,Cr,Cb value
# 	skin = np.zeros(Cr.shape, dtype = np.uint8) #mask
# 	(x,y) = Cr.shape
# 	for i in range(0, x):
# 		for j in range(0, y):
# 			if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0: #if not inside the ellipse
# 				skin[i][j] = 255
# 	res = cv2.bitwise_and(gesture,gesture, mask = skin)
# 	return res

# based on Cr value in YCrCb space and Otsu algorithm
def skinMask(gesture):
	YCrCb = cv2.cvtColor(gesture, cv2.COLOR_BGR2YCR_CB) # transform into YCrCb space
	(y,cr,cb) = cv2.split(YCrCb) # split Y,Cr,Cb value
	cr1 = cv2.GaussianBlur(cr, (5,5), 0)
	_, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Ostu process
	res = cv2.bitwise_and(gesture,gesture, mask = skin)
	return res
