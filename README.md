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

3. [Final Result](#final)


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

<p align="center" width="200%">
    <img width="30%" src="https://user-images.githubusercontent.com/116290888/199313545-526cc451-0f3f-4308-8fad-1f3db2ed4bf0.PNG"> 
</p>

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
#### The image i have selected is this one, from a newspaper called Independent: 
<p align="center" width="200%">
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

### Recognice Faces <a name="recognizefirst"></a>

```python
# After playing embeddings script, the pickle files must be open in order to get all the faces and names

f=open("ref_name.pkl","rb")
ref_dictt=pickle.load(f)         
f.close()

f=open("ref_embed.pkl","rb")
embed_dictt=pickle.load(f)     
f.close()

known_face_encodings = []  # encodingd of all faces
known_face_names = []	   # ref_id of all faces
```

#### Afther that, we will create the object to record the faces on camera and the sum of variables needed to process the information:  <a name="recognizesecond"></a>

````python

# Then, start the record of the image
video_capture = cv2.VideoCapture(0)

# Initialize some variables to locate the face encodings and names
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
````

#### Afther that we create the core of the using a while loop that will execute forever until we eliminate or quite the face/s on camera that can be recognice by the algotithm. <a name="recognizethird"></a>

````python
while True  :

	# Grab a single frame of video
	ret, frame = video_capture.read()

	# Resize frame of video to 1/4 size for faster face recognition processing
	small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

	# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
	rgb_small_frame = small_frame[:, :, ::-1]

	# Only process every other frame of video to save time
	if process_this_frame:
		# Find all the faces and face encodings in the current frame of video
		face_locations = face_recognition.face_locations(rgb_small_frame)
		face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

		face_names = []
		for face_encoding in face_encodings:
			# See if the face is a match for the known face(s)
			matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
			name = "Unknown"

			# Or instead, use the known face with the smallest distance to the new face
			face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
			best_match_index = np.argmin(face_distances)
			if matches[best_match_index]:
				name = known_face_names[best_match_index]
			face_names.append(name)
			
	# Turn the variable the cotnrary of his current value
	process_this_frame = not process_this_frame
````


#### And here the results are display. Notacie that other color, label or whatever feature on the rectangle that is ont he recogniced face can be changed. I did choose red because is a very striking color but you can use whatever you want. Also you can change the size not onyl of the rectangle but the size of the window displayed. <a name="recognizefourth"></a>

````python
# Display the results
	for (top, right, bottom, left), name in zip(face_locations, face_names):
		
		# Scale back up face locations since the frame we detected in was scaled to 1/4 size
		top *= 4
		right *= 4
		bottom *= 4
		left *= 4

		# Create a rectangle with the red color
		cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

		# Draw a label with a name (the name stored before) below the face
		cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
		font = cv2.FONT_HERSHEY_DUPLEX
		cv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	font = cv2.FONT_HERSHEY_DUPLEX

	# Display the resulting imagecv2.putText(frame, ref_dictt[name], (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
	cv2.imshow('Video', frame)

	# Hit 'q' on the keyboard to stop and the execution
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
		
# Destroy the process if nothing is found
video_capture.release()
cv2.destroyAllWindows()
````

# Final Result <a name="final"></a>

#### First, you have to put your image/s in fornt of the camera. A window with the video the camera is capturing will be deployed and the face in this case, as you can see, is detected WITH OTHER COMPLETLY IMAGE on the screen of my own phone:

<p align="center" width="200%">
    <img width="30%" src="https://user-images.githubusercontent.com/116290888/199321536-4cc5cf67-576c-4a53-9d01-010ddd2a22a1.PNG"> 
</p>



#### Image of Elon 1: https://www.independentespanol.com/tecnologia/twitter-elon-musk-compra-acuerdo-b2119250.html

#### Image of Elon 2: https://es.wikipedia.org/wiki/Elon_Musk#/media/Archivo:Elon_Musk_Royal_Society_(crop1).jpg
