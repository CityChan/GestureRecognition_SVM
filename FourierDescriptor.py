import cv2
import numpy as np

MIN_DESCRIPTOR = 32  #2 descriptors are already enough

def FourierDescriptor(res):
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    contour = find_contours(Laplacian) #extract contours coordinate
    contour_array = contour[0][:, 0, :]
    ret_np = np.ones(dst.shape, np.uint8)   #create black gackground
    ret = cv2.drawContours(ret_np, contour[0], -1, (255, 255, 255), 1)  # draw white contour
    contours_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contours_complex.real = contour_array[:, 0]  # horizontal axis = real part
    contours_complex.imag = contour_array[:, 1]  # vertical coordinates = imaginary part
    fourier_result = np.fft.fft(contours_complex)  # Fourier Transform
    descirptor_in_use = truncate_descriptor(fourier_result)
    return ret, descirptor_in_use


def find_contours(Laplacian):
    h = cv2.findContours(Laplacian,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # finding contours
    contour = h[1]
    contour = sorted(contour, key = cv2.contourArea, reverse=True) # sort any contours according to area they surround
    return contour


def truncate_descriptor(fourierresult):
    descriptors_in_use = np.fft.fftshift(fourierresult)
    # take middle descriptor
    center_index = int(len(descriptors_in_use)/2)
    low, high = center_index - int(MIN_DESCRIPTOR/2),center_index+int(MIN_DESCRIPTOR/2)
    descriptors_in_use = descriptors_in_use[low:high]
    descriptors_in_use = np.fft.ifftshift(descriptors_in_use)
    return descriptors_in_use


def reconstruct(img, descirptor_in_use):
    contour_reconstruct = np.fft.ifft(descirptor_in_use)
    contour_reconstruct = np.array([contour_reconstruct.real,
                                    contour_reconstruct.imag])
    contour_reconstruct = np.transpose(contour_reconstruct)
    contour_reconstruct = np.expand_dims(contour_reconstruct, axis=1)
    if contour_reconstruct.min() < 0:
        contour_reconstruct -= contour_reconstruct.min()
    contour_reconstruct *= img.shape[0] / contour_reconstruct.max()
    contour_reconstruct = contour_reconstruct.astype(np.int32, copy=False)

    black_np = np.ones(img.shape, np.uint8)  # create black background
    black = cv2.drawContours(black_np, contour_reconstruct, -1, (255, 255, 255), 1)  # draw white contours
    cv2.imshow("contour_reconstruct", black)
    return black

