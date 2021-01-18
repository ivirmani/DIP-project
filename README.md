# DIP-project
## Emotion Recognition With OpenCV in Python

INFORMATION SCIENCE AND ENGINEERING


DIGITAL IMAGE PROCESSING REPORT ON:
     EMOTION RECOGNITION
USING FACIAL EXPRESSION ANALYSIS

BY:
Ishaan Virmani 1MS18IS039
Abhinav Koul 1MS18IS006
Sai Sumanth 1MS18IS022
Hrushika Chitloor 1MS18IS025

  
							
Abstract
Emotion recognition is a natural capability in human beings. However, if we are to ever create a humanoid “robot” that can interact and emote with its human companions, the difficult task of emotion recognition will have to be solved. The ability for a computer to recognize human emotion has many highly valuable real-world applications. Consider the domain of therapy robots which are designed to provide care and comfort for infirm and disabled individuals. These machines could leverage information on a patient’s current and evolving state of mind, in order to tailor personalized strategies for patient care and interaction. For example, when a patient is upset or unhappy, a more effective strategy may be to take a moment to recognize the emotion and offer sympathy. Even outside of the realm of robotics, working with computers that can sense and respond to emotional state can go a long way to improve the quality of human-computer interaction (HCI). By designing HCI to be more like human-human interaction, we can create more natural, fulfilled, and productive working relationships with our machines. In this research, we explain how to recognize emotions through digital images using an Android application, and we will identify seven types of emotions (neutral- happy-sad- surprised- afraid- angry- disgusted). We designed this work based on a popular library called OpenCv, and the Fisherfaces algorithm that consists of (PCA) principal component analysis algorithm and (LDA) the linear discriminant analysis algorithm. 




Introduction
 The identification of facial expressions plays a key role in identifying patterns and image processing, and identifying facial expressions through three main stages: face detection, extraction features and classification.  

Fear, surprise, sadness, happiness, anger and disgust are six basic emotions that are universally accepted. These emotions can be classified as negative and positive emotions. Fear, anger, disgust and sadness are negative emotions and the majority of people do not like them whereas happiness is a positive emotion and everybody wishes to enjoy it. Anger is the most dangerous emotion and at some point in this emotion a person can hurt others purposefully. In this research, emotion detection systems using facial expressions have been implemented. Firstly, video frames are captured using a built in webcam. Then face extraction and cropping is carried out from these video frames and a training and test database is prepared. Then a low dimensional face space is constructed of a training database using Principal Component Analysis (PCA) and emotions are detected using Linear Discriminant Analysis (LDA) between various feature points of the test image to the train images. 

Literature survey
The process of emotion recognition involves the processing images and detecting the face then extracting the facial feature. 

Facial Expression Recognition consists of three main steps:
1. In the first step, face image is acquired and detects the face region from the images and pre-processed the input image to obtain images that have a normalized size or intensity. 
2. Next is expression features are extracted from the observed facial image or image sequence. 
3. Then extracted features are given to the classifier and the classifier provides the recognized expression as output.

A. Face Detection and Pre-processing 
The face detection is the process of extracting the face region from the background. It means to determine the position of the face in the image. This step is required because images have different scales. Input images having a complex background and variety of lightning conditions can be also quite confusing in tracking. Face expression recognition tends to fail if the test image has a different lighting condition than that of the training images. For that facial point can be detected inaccurately for that pre-processing step is required. 


B. Feature Extraction And Classification 
Selecting a set of feature points which represent the important characteristics of the human face. After the face has been located in the image, it can be analysed in terms of facial features. The features measure the certain parts of the face such as eyebrows or mouth corners. Various methods exist which can extract features for expression based on motion of the feature such as Active Appearance Model which is a statistical model of shape and gray scale information. The Features describe the change in face texture when particular action is performed such as wrinkles, bulges, forefront, regions surrounding the mouth and eyes. Image filters are used, applied to either the whole-face or specific regions in a face image to extract a feature vector. Principal Component Analysis, Local Binary Pattern (LBP),Fisher’s Linear Discriminator based approaches are the main categories of the approaches available.


After the set of features are extracted from the face region are used in the classification stage. The set of features are used to describe the facial expression. Classification requires supervised training, so the training set should consist of labelled data. Once the classifier is trained, it can recognize input images by assigning them a particular class label. The most commonly used facial expressions classification is done both in terms of Action Units, proposed in Facial Action Coding System (FACS) and in terms of six universal emotions: happy, sad, anger, surprise, disgust and fear.

Techniques used for emotion recognition 
Principal Components Analysis (PCA): It is a way of identifying patterns in data, and expressing the data in such a way as to highlight their similarities and differences. The facial expression recognition using eigen faces in which PCA is used to extract features from input images. 
First of all they create a training dataset to compare results. Once inputted face image is pre-processed and compared with training dataset which are already computed but based on the idea, they divided the training set into six basic classes according to universal expression(Happy, Surprise, Disgust, sad, Angry, Fear).

Local Binary Pattern:  LBP based feature extraction method is used owing to its excellent light invariance property and low computational complexity.The neighbourhood values are threshold by the centre value and the result is treated as a binary number. If the canter pixels value is greater than the neighbour’s value write 1, otherwise 0. In this way, it encodes the neighbourhood information very efficiently.

Active Appearance Model (AAM) is a statistical approach for shape and texture modelling and feature extraction. It has been extensively used in computer vision applications. AAM generates statistical appearance models by combining a model of shape variation with a texture variation. So the AAM creates the shape, texture combination model of training facial image sequence “Textures” are pixel intensities of the target image. 

Facial Action Coding System (FACS): was developed by Paul Ekman and Wallace Friesen in 1976 is a system for measuring facial expression. FACS is based on the analysis of the relations between muscle contraction and changes in the face appearance. The Face can be divided into Upper Face and Lower Face Action units. Action Units are changes in the face caused by one muscle or a combination of muscles. There are 46 AUs that represent changes in facial expression and 12 AUs connected with eye gaze direction and head orientation. 

Haar Classifier based method is chosen for face detection owing to its high detection accuracy and real time performance. Consists of black and white connected rectangles in which the value of the feature is the difference of sum of pixel values in black and white regions. The computational speed of the feature calculation is increased with the use of Integral image



System Design


IMAGE ACQUISITION
↓
IMAGE PREPROCESSING
↓
FEATURE EXTRACTION
↓
CLASSIFICATION
↓
DECISION MAKING


Hardware requirements:
Video Quality           : 720p 
Frame Rate               : 30fps
Flicker Resolution    : 60Hz
Display Resolution   : 4K
Refresh Rate             : 60Hz
Graphics Card          : GeForce GTX 1650
Processor	        : Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz 2.59 GHz
Installed RAM	        : 8.00 GB (7.84 GB usable)
System type	        : 64-bit operating system, x64-based processor
Software requirements:
Python
opencv-python
opencv-contrib-python

Dataset information with web links
Database consists of 1000 face images that are stored in the computer. Images obtained from the Karolinska Directed Emotional Faces (KDEF) were used to train various Fisherface models. We train these photos using OpenCv library for detecting faces, eyes and mouth using a cascade classifier. 
Fisherface recognizer requires every training data to have the same pixel count. This raises a problem because the dataset from KDEF(link here) does not have uniform size and thus produces error during training.
To address this problem, emotion_data_prep.py are created. Both of them use face detection algorithms from face_detection.py to detect faces in photos. Then, the picture would be normalized to uniform size (350px x 350px) and saved in grayscale to speed up the training process.
KDEF
├── data
    ├── raw_emotion
        ├── afraid
        ├── angry
        ├── disgusted
        ├── happy
        ├── neutral
        ├── sad
        ├── surprised
Before running emotion_data_prep.py, ensure that your file structure is as pictured above. You would need to extract the KDEF images and put them in the respective directories that resemble the appropriate emotion.



Methodology
To make our model run in real time, we leverage the Viola-Jones Object Detection framework (VJODF) described in, as implemented in OpenCV, for the task of face detection and tracking. The algorithm is robust, and fast enough to run in real time.

Algorithm in detail
Face Detection
First part of my system is a module for face detection and landmark localization in the image. Algorithm for face detection is based on work by Viola and Jones. In this approach the image is represented by a set of Haar-like features. Possible types of features are two-, three- and four rectangular features. 
Feature value is calculated by subtracting the sum of the pixels covered by white rectangle from the sum of pixels under the gray rectangle. Two rectangular features detect contrast between two vertically or horizontally adjacent regions. Three rectangular features detect contrasted regions placed between two similar regions and four rectangular features detect similar regions placed diagonally. 

The method is widely used in area of face detection. However, it can be trained to detect any object. What is more, this algorithm is quick and efficient and could be used in real time applications. In proposed system, the algorithm is applied for face, eyes and mouth localization with use of already trained classifiers from OpenCV library. 

Feature Extraction:
	Gray scale conversion: Modern descriptor-based image recognition systems often operate on grayscale images, with little being said of the mechanism used to convert from color-to-grayscale. 
The main reason why grayscale representations are often used for extracting descriptors instead of operating on color images directly is that grayscale simplifies the algorithm and reduces computational requirements.
Histogram equalization: Histogram equalization is a contrast enhancement technique in a spatial domain in image processing using histogram of image. Histogram equalization usually increases the global contrast of the processing image. This method is useful for the images which are bright or dark.
Gaussian blur: In image processing, a Gaussian blur (also known as Gaussian smoothing) is the result of blurring an image by a Gaussian function . It is a widely used effect in graphics software, typically to reduce image noise and reduce detail. 
The visual effect of this blurring technique is a smooth blur resembling that of viewing the image through a translucent screen, distinctly different from the bokeh effect produced by an out-of-focus lens or the shadow of an object under usual illumination. Mathematically, applying a Gaussian blur to an image is the same as convolving the image with a Gaussian function. 
	OpenCv eye pair classifier: Use OpenCV eye pair classifier to locate the eyes. Calculate the midpoint between the eyes and use this data to vertically and horizontally align the face within the image. 
OpenCv eye/mouth classifier: Use OpenCV eye pair and mouth classifiers. Use this location data to obtain a tightly cropped facial image that will align with the template images used to build the Fisherface models. 

Emotion Recognition :

The last stage of our system is devoted to facial expressions recognition. This task requires classifier training with a set of images with particular emotions displayed.
Principle component analysis (PCA): The main purposes of a principal component analysis are the analysis of data to identify patterns and finding patterns to reduce the dimensions of the dataset with minimal loss of information. Here, our desired outcome of the principal component analysis is to project a feature space (our dataset consisting of nn dd-dimensional samples) onto a smaller subspace that represents our data “well”. A possible application would be a pattern classification task, where we want to reduce the computational costs and the error of parameter estimation by reducing the number of dimensions of our feature space by extracting a subspace that describes our data “best”. 

 Linear discriminate analysis (LDA):
Linear Discriminate Analysis (LDA) is used to solve dimensionality reduction for data with higher attributes and pre-processing step for pattern-classification and machine learning applications. It used for feature extraction and linear transformation that maximize the separation between multiple classes.

Fisherfaces algorithm:
Fisherfaces were first described by Belhumeur, Hespanha, and Kriegman in their paper “Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.” The authors note that with respect to a set of images, variation within classes lies in a linear subspace of the image space indicating that the classes are convex, and thus linearly separable. In the Fisherfaces method, the task of classification is simplified using of Fisher’s Linear Discriminate (FLD) which attains a greater between-class scatter than PCA. In order to obtain tightly clustered well separated classes, LDA maximizes the ratio of the determinant of between-class to within-class scatter. 
The Fisherfaces technique takes a pattern classification approach considering each pixel in an image as a coordinate in the high-dimensional image space. The algorithm begins by creating a matrix wherein each column vector (consisting of pixel intensities) represents an image. A corresponding class vector containing class labels is also created. The image matrix is projected into (nc)-dimensional subspace (where n is the number of images and c is the number of classes). The between-class and within-class scatter of the projection is calculated and LDA is applied. For our purposes here, we levered functionality available within the libraries of OpenCV to implement LDA using the Fisherface methodology.
This method requires that all images, both in the training and testing set, be equal in size. Method performance is the highest when all images are full frontal head shots with major features aligned.  This technique does not work on an image directly rather it converts images into grayscale vector matrices and works with the vector form. Ultimately, each image is represented by a weight vector which indicates the percentage of each Fisherface it contains. It is this weight vector representing unique image attributes that is used in a nearest neighbor search of the training set to predict the identity of an unknown face.


Implementation:
Haar Cascade
These models are provided by OpenCV and allows the program to detect human faces. After some manual and automated testings, I decided to use the first alternate version. If for some reason you want to change the way this program detect human faces, open face_detection.py, search the following line: faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml') 
and change the model path to the desired one.
 
Emotion Classifier
These models are created with train_emotion_classifier.py. Each model is trained with dataset from KDEF. There are 2 versions: normal and decent. The normal version is trained with all the data from KDEF, while the decent version is trained with modified data from KDEF.
Modified here means deleting obviously misleading emotions. For example, there were a picture labelled sad that shows the person smiling while having tears around the eyes. It is very unusual for people to smile while crying, but this one person does it. To achieve better result, the said picture is removed from dataset. Another example, a person shows no real emotion in a picture labelled angry. That particular picture is then re-labelled as neutral.
To switch versions, open facifier.py and search the following line: fisher_face_emotion.read('models/emotion_classifier_model.xml') and change the model path to the desired one.
Windows/Linux
It turns out that a model trained using Windows can only work in Windows and that also applies to Linux. A new Windows-friendly model has been added to the model directory.
For anyone using Windows, go to line 69-73 in
src/facifier.py.
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model.xml')

Change them into:
fisher_face_emotion = cv2.face.FisherFaceRecognizer_create()
fisher_face_emotion.read('models/emotion_classifier_model_windows.xml')

The application should work properly in Windows with the new models.


Accuracy:

Results and Conclusion:
Our results suggest that the Fisherface model can be successful in recognizing human emotions in facial images.
The main objective of this work is to design, implement and evaluate the system of knowledge of emotions through the analysis of facial expressions using different techniques. My work has made facial tracking techniques work better under varying illumination and pose. The main contributions of this research are as follows. Firstly, using Viola and Jones algorithm that is based on the famous OpenCV library for facial tracking and detection in difficult conditions. Secondly, applying histogram equalization and Gaussian blur for per-processing facial images. Thirdly, extract features from facial images such as eyes and mouth. Finally, classify emotions using a Fisherface system that is based on PCA and LDA algorithms. 


Future enhancements:

Despite the promising results, the presented approaches in this thesis are limited to acted facial expressions. In practice, spontaneous and subtle facial expressions can more reveal the real emotional state of human beings. The proposed methods may suffer from the subtle changes and irregular motion variation of facial expression in spontaneous behavior.


References:
http://www.scholarpedia.org/article/Fisherfaces
http://www.pitt.edu/~emotion/ck-spread.htm
https://www.kdef.se/download-2/register.html
Viola and M.J.Jones (2001) Rapid Object Detection using a Boosted Cascade of Simple Features Computer Vision and Pattern . Recognition. 
OpenCv. http://opencv.org.
S Sukanya Sagarika ;Pallavi Maben, Laser Face Recognition and Facial Expression Identification using PCA, 2014 IEEE 
S L Happy;Anjith George;Aurobinda Routray, “A Real Time Facial Expression Classification System Using Local Binary Patterns.,” 2012 IEEE
Kwang-Eun Ko;Kwee-Bo Sim, “Development of a Facial Emotion recognition Method based on combining AAM with DBN,” IEEE 2010
Ms.Aswathy.R “A Literature review on Facial Expression Recognition Techniques,” IOSR Journal of Computer Engineering (IOSR-JCE) 2013



