import face_recognition
import cv2
import os
import glob
import pickle
import numpy as np
import math
import sys
import time
from PIL import Image
# for voice input
import speech_recognition as sr
import threading
import queue


def HasKeywords(texts, keywords):
    if texts != []:
        textList = [text['transcript']
                    for text in texts['alternative']]
        print(textList)
        start = False
        for text in textList:
            for keyword in keywords:
                if keyword in text:
                    return True
    return False


def Voice2Text(qCommand):
    while not start_game:
        # check that recognizer and microphone arguments are appropriate type
        # if not isinstance(recognizer, sr.Recognizer):
        #     raise TypeError("`recognizer` must be `Recognizer` instance")
        # if not isinstance(microphone, sr.Microphone):
        #     raise TypeError("`microphone` must be `Microphone` instance")

        # adjust the recognizer sensitivity to ambient noise and record audio
        # from the microphone
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                # show the user the transcription
                audio = recognizer.listen(source, 2, 1)
                texts = recognizer.recognize_google(audio, show_all=True)
                qCommand.put(texts)
            except (sr.RequestError, sr.UnknownValueError, sr.WaitTimeoutError) as e:
                print("Voice Error")
            

def IsWatching(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = cv2.CascadeClassifier('haarcascade_eye.xml')
    detected = faces.detectMultiScale(frame, 1.3, 5)

    return len(detected) >= 1
 

def UpdateStareList(frame, rect, face_encoding, name, starer_idxes):
    (top, right, bottom, left) = rect
    # if is registering player and the perosn is watching camera
    if (not start_game) and IsWatching(frame[top:bottom, left:right]):
        # if the person is already in staring list
        # find the person by comparing distance of face encoding
        face_distances = face_recognition.face_distance(
            starer_encodings, face_encoding)
        if len(face_distances) != 0 and min(face_distances) < 0.5:
            index = np.argmin(face_distances)
            starer_idxes.remove(index)

            # add person to player list if:
            # the person not in player list AND stared camera for more than 1 seconds
            face_distances = face_recognition.face_distance(
                player_encodings, face_encoding)
            if (len(face_distances) == 0 or min(face_distances) > 0.5) and time.time() - stare_time[index] > 1:
                player_names.append(name)
                player_encodings.append(face_encoding)
        else:
            # if the person not in staring list
            # add face encondings and stare time to lists
            starer_encodings.append(face_encoding)
            stare_time.append(time.time())
    return starer_idxes

def SetLabel(frame, label, point):
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    box_padding = 5

    text = cv2.getTextSize(label, fontface, scale, thickness)
    cv2.rectangle(frame, (point[0] - box_padding, point[1] + box_padding),
                  (point[0] + text[0][0] + box_padding, point[1] - text[0][1] - box_padding), (100, 100, 100), cv2.FILLED)
    cv2.putText(frame, label, point,
                fontface, scale, (255, 255, 255), 1)


def DispResult(frame):
    global player_info, player_encodings, player_names, stare_time

    # starer_idxes is used to check which starer index is checked
    # the idxes will be removed after checked starer_encodings
    starer_idxes = list(range(len(starer_encodings)))

    for (top, right, bottom, left), name, face_encoding in zip(face_locations, face_names, face_encodings):
        
        starer_idxes = UpdateStareList(frame, (top, right, bottom, left),
                        face_encoding, name, starer_idxes)
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (100, 100, 100), 2)

        # Calculate distance and angle from camera
        focal_length = 600
        avg_face_height = 18
        img_height = bottom - top
        img_angle = (right + left) / 2 - frame.shape[1] / 2

        distance = focal_length * avg_face_height / img_height
        angle = math.atan(img_angle / focal_length)
        x = int(distance * math.cos(angle))
        y = int(-distance * math.sin(angle))

        # Draw a label with a name below the face
        SetLabel(frame, name + ', ' + str(x) +
                 ', ' + str(y), (left, bottom))

        #set player name and location if all players are found
        if start_game and len(face_names) == len(player_names) and 'Unknown' not in face_names:
            player_info[name] = [distance, angle]
        else:
            player_info = {}

    # remove starers that stop staring
    # remove list in reverse order to prevent messing up the indexes
    for index in sorted(starer_idxes, reverse=True):
        del starer_encodings[index]


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
    if HAS_GPU:
        # Use GPU
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, 10)
    else:
        # Use CPU
        face_locations = face_recognition.face_locations(rgb_frame)
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
    global prev_frame_time, face_locations, face_encodings, face_names

    # Grab a single frame of video
    ret = False
    while not ret:
        ret, frame = video_capture.read()

    #Flip frame
    frame = cv2.flip(frame, 1)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Process frame every 0.2 seconds
    if time.time() - prev_frame_time > 0.2:
        face_locations, face_encodings, face_names = ProcessFrame(rgb_frame)
        prev_frame_time = time.time()

    # Display basic information and registered player
    DispInfo(frame)
    # Display name, distance and angle
    DispResult(frame)

    # Display the resulting image
    cv2.imshow('Casino', frame)

    #Send frame as jpg to queue(for Flask)
    ret, jpeg = cv2.imencode('.jpg', frame)
    return jpeg

def ProcessFrame(rgb_frame):
    # Find all the faces and face encodings in the current frame of video
    if HAS_GPU:
        # Use GPU
        face_locations = face_recognition.face_locations(
            rgb_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, 10)
    else:
        # Use CPU
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)

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

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
known_face_encodings = []
known_face_names = []
player_encodings = []
player_names = []
prev_frame_time = time.time()
start_game = False

#additional feature
starer_encodings = []
stare_time = []
player_info = {}
video_capture = cv2.VideoCapture(0)
HAS_GPU = False

#voice command
recognizer = sr.Recognizer()
microphone = sr.Microphone(device_index=2)

def main(qFrame):
    # voice command
    qCommand = queue.Queue()
    thread = threading.Thread(target=Voice2Text,
                              name=Voice2Text, args=(qCommand,))
    thread.start()
    # end of voice command

    global start_game, known_face_names, known_face_encodings, prev_frame_time
    missing_time = time.time()
    # Load face encodings
    with open(os.getcwd() + '/classmate/dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys())
    known_face_encodings = np.array(list(all_face_encodings.values()))

    #set window size
    cv2.namedWindow("Casino", cv2.WINDOW_AUTOSIZE)
    while True:
        # store return value to queue (For Flask)
        qFrame.put(ReadWriteFace())

        #if found player for more than 2 seconds, return player_info to caller
        if start_game and time.time() - missing_time > 2:
            # Release handle to the webcam
            video_capture.release()
            cv2.destroyAllWindows()

            # Set Buffering image if video streaming ended
            ret, bufImg = cv2.imencode('.jpg', cv2.imread(
                'casino.jpg', cv2.IMREAD_UNCHANGED))
            qFrame.put(bufImg)

            print(player_info)
            #store player info into a file to be retrive later
            with open('player_info.pkl', 'wb') as f:
                pickle.dump(player_info, f, pickle.HIGHEST_PROTOCOL)
                
            return player_info
        elif len(player_info) == 0:
            missing_time = time.time()

        # get voice command
        try:
            command = qCommand.get(False)
        except queue.Empty:
            command = []
            
        key = cv2.waitKey(1) & 0xFF
        # Hit <Space> to capture player image
        # or stare at the camera for 2 seconds(as shown in DispResult and IsWatching functions)
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
        # Voice command with 'start' / 'stop' / 'game' will do the same thing
        elif HasKeywords(command, ['start', 'stop', 'game']) or key == ord('\r'):
            # Replace known faces to prevent confusion
            known_face_names.clear()
            for index in range(len(player_names)):
                known_face_names.append("Player " + str(index+1))
            known_face_encodings = player_encodings
            #make sure the next frame will be executed
            prev_frame_time = 0
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


# Parse Jpeg Frame to Flask
import threading
import queue
def getFrameOrInfo():
    qFrame = queue.Queue()
    thread = threading.Thread(target=main,
                              name=main, args=(qFrame,))
    thread.start()
    
    while thread.isAlive():
        try:
            # Yield jpeg frame if queue is not empty
            jpegFrame = qFrame.get(False)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpegFrame.tobytes() + b'\r\n')
        except queue.Empty:
            pass

if __name__ == "__main__":
    qFrame = queue.Queue()
    thread = threading.Thread(target=main,
                              name=main, args=(qFrame,))
    thread.start()
    
