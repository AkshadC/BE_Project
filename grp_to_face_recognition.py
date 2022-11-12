import cv2
import face_recognition
from os import listdir
from os.path import isfile, join
from itertools import compress


class Recognition:
    def __init__(self, folder_path, files_in_folder=None, present=None, names=None, encodings=None):
        if encodings is None:
            encodings = []
        if names is None:
            names = []
        self.present = present
        self.present = []
        self.names = []
        self.encodings = []
        self.folder_path = folder_path
        self.files_in_folder = files_in_folder

    def get_file_path(self):

        self.files_in_folder = listdir(self.folder_path)

    def extract_images_from_image(self, grp_image):
        image = face_recognition.load_image_file(grp_image)
        face_locations = face_recognition.face_locations(image)
        for faces in face_locations:
            top, right, bottom, left = faces
            face_image = image[top:bottom, left:right]
            self.recognize_image_from_all(face_image)

    def recognize_image_from_all(self, image_detected):

        img = image_detected
        single_image_enc = face_recognition.face_encodings(img)[0]
        results = face_recognition.compare_faces(self.encodings, single_image_enc)
        if len(results) != 0:
            index = list(compress(range(len(results)), results))
            self.present.append(self.names[index[0]])

    def find_all_names_and_encodings(self):

        name = ""
        for i in self.files_in_folder:
            path = join(self.folder_path, i)
            image = cv2.imread(path)
            self.encodings.append(face_recognition.face_encodings(image)[0])
            for x in i:
                if x == ".":
                    break
                name += x
            self.names.append(name)
            name = ""


def main():
    reco = Recognition("people/")
    reco.get_file_path()
    reco.find_all_names_and_encodings()
    reco.extract_images_from_image("IMG_1741.JPG")
    print("THE NAMES FOUND ARE : ", end=" ")
    for i in reco.present:
        print(i, end=" ")


if __name__ == "__main__":
    main()
