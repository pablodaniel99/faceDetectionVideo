# Face Detection Video

## Table of contents
1. [Embedded Faces](#embedded)  
    1. [First Step](#embedfirst)
    2. [Second Step](#embedsecond)
    3. [Third Step](#embedthird)


2. [Recognice Faces](#recognize)
    1. [First Step](#recognizefirst)
    2. [First Step](#recognizesecond)
    3. [Third Step](#recognizethird)
    4. [Fourth Step](#recognizefourth)


# Embedded Faces <a name="embedded"></a>

#### The first script that I'm going to explain is the first one to be executed, which is "embeddedFaces". The first thing wee need is to have and import this libraries: <a name="embedfirst"></a>

```python
import sys
import cv2 
import face_recognition
import pickle
```

#### As I explained [on other project](https://github.com/pablodaniel99/detectFaces/blob/main/README.md#maybe-you-have-some-problems-importing-cv2-it-is-highly-recommended-to-use-virtual-studio-code-why-because-usign-this-platform-will-make-the-installation-of-this-package-way-more-easy-than-using-conda-o-pip-which-not-only-could-provoke-an-error-but-may-also-last-2-hours), you maybe have some problems installing "face_recognition". If any problem is created, the next step is to enter a name and id: <a name="embedsecond"></a>

```python
# Name and id will be enter in order to identificate the person on the camera
name=input("Enter name: ")
ref_id=input("Enter id: ")
```

#### I have entered the name of ELon Musk with ID 001: 




#### Then we will open the pickle file (or create it if it isn't created): 

```python
# Open the pickle file, in order to get the name and id by first, getting the name
try:
	f=open("ref_name.pkl","rb")
	ref_dictt=pickle.load(f)
	f.close()
except:
	ref_dictt={}
ref_dictt[ref_id]=name
```
#### Do the same with the other pickle file:

```python
# Then, write a pickled representation of obj to the open file

f=open("ref_name.pkl","wb")
pickle.dump(ref_dictt,f)
f.close()


# The embed file (where the captured faces are) is opened and loaded 
try:
	f=open("ref_embed.pkl","rb")
	embed_dictt=pickle.load(f)
	f.close()
except:
	embed_dictt={}
```

#### After that, on the core of this script, we will reord 5 photos on the face that is currently on the camera. I highly recommend due to the algorithm that you have natural light for the photos coming from both sides, to be the more be as calm as possible. After that, take the photos pressing the key 's' five times and press to finish the process the key 'q'. The code you need is the one below: <a name="embedthird"></a>

```python

# For 5 times in a row 5 images will be take in order to capture the faces needed to detect the face in the other script

for i in range(5):
	key = cv2. waitKey(1)
	webcam = cv2.VideoCapture(0)
	while True:
	     
		check, frame = webcam.read()

		# Open the camera with this size:

		cv2.imshow("Capturing", frame)
		small_frame = cv2.resize(frame, (0, 0), fx=0.35, fy=0.35)
		rgb_small_frame = small_frame[:, :, ::-1]

		# Waitkey allows as to close the camera after 5 miliseconds

		key = cv2.waitKey(5)

		# Now lets define what happens when the key 's' is press

		if key == ord('s') : 
			face_locations = face_recognition.face_locations(rgb_small_frame)
			if face_locations != []:
				
				# Take the frame when the key is pressed. after that, in 1 milisecond, the camera window is destroyed

				face_encoding = face_recognition.face_encodings(frame)[0]
				if ref_id in embed_dictt:
					embed_dictt[ref_id]+=[face_encoding]
				else:
					embed_dictt[ref_id]=[face_encoding]
				webcam.release()
				cv2.waitKey(1)
				cv2.destroyAllWindows()     
				break
		
		# When the key 'q' is pressed, the camera has to end and the loop 'for' is done

		elif key == ord('q'):
			print("Turning off camera.")
			webcam.release()
			print("Camera and program off.")
			cv2.destroyAllWindows()
			break
```

#### The image i have selected is this one, from a New York Time post: 
<p align="center" width="100%">
    <img width="30%" src="https://user-images.githubusercontent.com/116290888/199306421-6b83beb5-55de-4696-841a-4651f6aa0105.PNG"> 
</p>


#### Now, all the work related with the first script is done, let's move forward to the second and las one.

# Recognice Faces <a name="recognize"></a>
#### The second script and last one script is the one executed in order to detect the faces recorded on the last script, for that mission, this libraries are necessary:

```python
import face_recognition
import cv2
import numpy as np
import glob
import time
import csv
import pickle
```

# Recognice Faces <a name="recognizefirst"></a>
