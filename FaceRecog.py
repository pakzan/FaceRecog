import face_recognition
import cv2
import os
import glob
import pickle
import numpy as np
import math
import sys
import time


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


def DispResult(frame, face_locations, face_names):
    global player_info
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Calculate distance and angle from camera
        focal_length = 600
        avg_face_height = 18
        img_height = bottom - top
        img_angle = (right + left) / 2 - frame.shape[1] / 2

        distance = focal_length * avg_face_height / img_height
        angle = math.degrees(math.atan(img_angle / focal_length))

        # Draw a label with a name below the face
        SetLabel(frame, name + ', ' + str(int(distance)) +
                 ', ' + str(int(angle)), (left, bottom))

        #set player name and location if all players are found
        if start_game and len(face_names) == len(player_names) and 'Unknown' not in face_names:
            player_info[name] = [distance, angle]
        else:
            player_info = {}


def DispInfo(frame):
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

    # Show current registered player
    for index, player_name in enumerate(player_names):
        SetLabel(frame, 'Player ' + str(index + 1) + ': ' +
                 player_name, (0, top_padding + text_height * (index + 1)))


def ProcessFrame(rgb_frame):
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    # Using GPU
    #face_locations = face_recognition.batch_face_locations(rgb_frame, 5)
    #face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, 5)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"

        # See if the face is a match for the known face(s)
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)
        if len(face_distances) != 0 and min(face_distances) < 0.5:
            name_index = np.argmin(face_distances)
            name = known_face_names[name_index]

        face_names.append(name)
    return face_locations, face_encodings, face_names


def ReadWriteFace():
    global process_this_frame, face_locations, face_encodings, face_names, outFrame

    # Grab a single frame of video
    ret, frame = video_capture.read()
    #Flip frame
    frame = cv2.flip(frame, 1)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        face_locations, face_encodings, face_names = ProcessFrame(rgb_frame)
    process_this_frame = not process_this_frame

    # Display basic information and registered player
    DispInfo(frame)
    # Display name, distance and angle
    DispResult(frame, face_locations, face_names)

    # Display the resulting image
    cv2.imshow('Casino', frame)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
player_encodings = []
player_names = []
process_this_frame = True
start_game = False
#additional feature
player_info = {}
video_capture = cv2.VideoCapture(0)


def main():
    global start_game, known_face_names, known_face_encodings, process_this_frame
    missing_time = round(time.time())
    # Load face encodings
    with open(os.getcwd() + '/classmate/dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys())
    known_face_encodings = np.array(list(all_face_encodings.values()))

    #set window size
    cv2.namedWindow("Casino", cv2.WINDOW_AUTOSIZE)
    while True:
        ReadWriteFace()

        #if found player for more than 2 seconds, return player_info to caller
        if round(time.time()) - missing_time > 2:
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()
            return player_info
        elif len(player_info) == 0:
            missing_time = round(time.time())

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
            #make sure the next frame will be executed
            process_this_frame = True
            start_game = True
        # Hit 'r' to restart
        elif key == ord('r'):
            os.execl(sys.executable, sys.executable, *sys.argv)
        # Hit 'q' or cross button to quit
        elif key == ord('q') or cv2.getWindowProperty('Casino', cv2.WINDOW_AUTOSIZE) < 0:
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()