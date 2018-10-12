# FaceRecog
Recognizes classmates and displays their names.

This program requires face_recognition and cv2 library to be installed.

# Run instruction:
1. Paste the faces you want to recognize into classmate folder. The images have to be in JPG format with file name NAME.jpg.
2. Run the file PreSavedFace.py to encode and store the faces.
3. Change the HAS_GPU in FaceRecog.py to true to use GPU.
3. Run the file main.py to recognize the faces.
4. Press <SPACE> to register current faces or new faces as players.
5. Press <ENTER> to start the game.
6. Press 'r' to restart or 'q' to close the program.

# Additional Information:
The name, location and angle of the players to the camera will be calculated and displayed on the screen with the format: name, location(cm), angle(degree).
Location and angle is calculated by using simple trigonometry and optical knowledge.
The values will return to caller(main.py) after all players has been detected for more than 2 seconds.
