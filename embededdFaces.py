import sys
import cv2 
import face_recognition
import pickle

# Name and id will be enter in order to identificate the person on the camera
name=input("Enter name: ")
ref_id=input("Enter id: ")

# Open the pickle file, in order to get the name and id by first, getting the name
try:
	f=open("ref_name.pkl","rb")
	ref_dictt=pickle.load(f)
	f.close()
except:
	ref_dictt={}
ref_dictt[ref_id]=name


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

# After 5 iterations or after the key q is pressed, the pickle file is open in order to get the face in the embed file

f=open("ref_embed.pkl","wb")
pickle.dump(embed_dictt,f)
f.close()