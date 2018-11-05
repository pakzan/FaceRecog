import face_recognition
import os
import glob
import pickle

# Create arrays of known face encodings and their names
all_face_encodings = {}
path = os.getcwd() + '/faces'
for filename in glob.glob(os.path.join(path, '*.jpg')):
    # Load pictures and learn how to recognize them.
    name = os.path.splitext(os.path.basename(filename))[0]
    image = face_recognition.load_image_file(filename)
    all_face_encodings[name] = face_recognition.face_encodings(image)[0]

# Save them into a file
with open(path + '/dataset_faces.dat', 'wb') as f:
    pickle.dump(all_face_encodings, f)
