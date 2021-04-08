# Gesture Recognition based on SVM
## Course Porject in EE381 Digital Video
I present a gestures recognition system based on OpenCV. The system contains data collection, feature extraction, data argument and support vector machine (SVM) learning. Data collection function allows us to build our database through camera in computer. Feature extraction function contains denoising, skin detection, binarization, morphological processing and contour extraction. I try to use several methods to detect skin such as RGB, HSV color space, ellipse model and Otsu method then compare the results. Data argument function allow us to enrich our database with adjusting our original pictures in random direction instead of repeating using camera. Finally, I transform pictures in database into numerical values and use SVM to train our model. For convenience, I build a GUI via PyQt5 library to test my system.
