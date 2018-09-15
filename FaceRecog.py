import face_recognition
import cv2
import os
import glob
import pickle
import numpy as np
import sys

from imutils.video import WebcamVideoStream

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)


def SetLabel(frame, label, point):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    box_padding = 5

    text = cv2.getTextSize(label, fontface, scale, thickness)
    cv2.rectangle(frame, (point[0] - box_padding, point[1] + box_padding),
                  (point[0] + text[0][0] + box_padding, point[1] - text[0][1] - box_padding), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, point,
                fontface, scale, (255, 255, 255), 1)

def ReadWriteFace():
    global process_this_frame, face_locations, face_encodings, face_names

    # Grab a single frame of video
    frame = video_capture.read()

    #Flip frame
    frame = cv2.flip(frame, 1)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:,:,::-1]
    
    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"
            if True in matches:
                face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)

                name_index = np.argmin(face_distance)
                name = known_face_names[name_index]
                
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display basic information
    text_height = 22
    top_padding = text_height*5
    if not start_game:
        SetLabel(frame, "Press <Space> to register player/s in camera",
                 (0, text_height))
        SetLabel(frame, "Press <Enter> to start game", (0, text_height*2))
        SetLabel(frame, "Press 'r' to restart game", (0, text_height*3))
        SetLabel(frame, "Press 'q' to quit game", (0, text_height*4))
    else:
        SetLabel(frame, "Game Started", (0, text_height))
        SetLabel(frame, "Press 'r' to restart game", (0, text_height*2))
        SetLabel(frame, "Press 'q' to quit game", (0, text_height*3))
        top_padding = text_height*4

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        SetLabel(frame, name, (left, bottom))

    # Show current registered player
    for index, player_name in enumerate(np.unique(player_names)):
        SetLabel(frame, 'Player ' + str(index + 1) + ': ' +
                 player_name, (0, top_padding + text_height * (index + 1)))

    # Display the resulting image
    cv2.imshow('Video', frame)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
player_encodings = []
player_names = []
process_this_frame = True
start_game = False

video_capture = WebcamVideoStream(src=0).start()

# Load face encodings
with open(os.getcwd() + '/classmate/dataset_faces.dat', 'rb') as f:
    all_face_encodings = pickle.load(f)

# Grab the list of names and the list of encodings
known_face_names = list(all_face_encodings.keys())
known_face_encodings = np.array(list(all_face_encodings.values()))

while True:
    ReadWriteFace()

    key = cv2.waitKey(1) & 0xFF
    # Hit <Space> to capture player image
    if (not start_game) and key == ord(' '):
        # Load pictures and learn how to recognize them.
        # Update player information if player already exits
        for index, player_name in enumerate(player_names):
            if player_name in face_names:
                player_names.pop(index)
                player_encodings.pop(index)

        player_names.extend(face_names)
        player_encodings.extend(face_encodings)

    # Hit <Enter> on the keyboard to confirm player and start play
    elif key == ord('\r') and len(player_names) != 0:
        # Replace known faces to prevent confusion
        known_face_names.clear()
        for index in range(len(player_names)):
            known_face_names.append("Player " + str(index+1))
        known_face_encodings = player_encodings
        start_game = True
    # Hit 'r' to restart
    elif key == ord('r'):
        os.execl(sys.executable, sys.executable, *sys.argv)
    # Hit 'q' to quit
    elif key == ord('q'):
        break

# Release handle to the webcam
video_capture.stop()
cv2.destroyAllWindows()
