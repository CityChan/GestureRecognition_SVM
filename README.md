# Gesture Recognition based on SVM
##  [Course Porject in EE381 Digital Video](https://drive.google.com/file/d/12gz5XkhfFTu24mR2Aay16ELDhS2TAk_B/view?usp=sharing)
I present a gestures recognition system based on OpenCV. The system contains data collection, feature extraction, data argument and support vector machine (SVM) learning. Data collection function allows us to build our database through camera in computer. Feature extraction function contains denoising, skin detection, binarization, morphological processing and contour extraction. I try to use several methods to detect skin such as RGB, HSV color space, ellipse model and Otsu method then compare the results. Data argument function allow us to enrich our database with adjusting our original pictures in random direction instead of repeating using camera. Finally, I transform pictures in database into numerical values and use SVM to train our model. For convenience, I build a GUI via PyQt5 library to test my system.

## Description
**CatchGesture.py**: Using **OpenCV2** to capture gesture images with PC's camera and save images in assigned folder. Keys's', 'w', 'd', 'a' are used to adjust the camera window.<br><br>
**GUI.py**: Using **QtWidgets** to develop a graphical user interface for experiments conveniently.<br><br>
**Argument.py**: add more images by filp and rotationwith different angles from original images. These arguments remain the same labels.<br><br>
**Preprocessing.py**: preprocessing the collected images including computing segmentation, erosion, dilation and edge detection.<br><br>
**FourierDescriptor.py**: Fourier Descriptor for edge detection.<br><br>
**ImageToData.py**: convert processed images into vectors.<br><br>
**Training.py:** training the SVM model with data.<br><br>
