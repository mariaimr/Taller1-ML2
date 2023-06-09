import os

import cv2
import numpy as np

from picture.DownloadPicture import download_pictures


class Picture:
    face_pictures = []
    my_picture = 0
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.resources_path = os.path.join(path, "../resources")
        self.pictures_path = os.path.join(self.resources_path, "face_pictures")
        self.my_picture_filename = "MariaMartinez.jpg"
        self.new_size = (256, 256)

    def load_pictures(self):
        download_pictures()
        face_pictures = []
        for file in os.listdir(self.pictures_path):
            file_path = os.path.join(self.pictures_path, file)
            if os.path.isfile(file_path) and file.lower().endswith((".jpg", ".jpeg", ".png")):
                picture = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if picture is not None:
                    gray_picture = cv2.resize(picture, self.new_size)
                    face_pictures.append(gray_picture)
        Picture.face_pictures = face_pictures
        return face_pictures

    def edit_my_picture(self):
        my_picture_file = os.path.join(self.pictures_path, self.my_picture_filename)
        picture = cv2.imread(my_picture_file, cv2.IMREAD_GRAYSCALE)
        Picture.my_picture = cv2.resize(picture, self.new_size)
        return cv2.resize(picture, self.new_size)

    def get_my_picture(self):
        my_picture = self.edit_my_picture()
        cv2.imwrite(os.path.join(self.resources_path, self.my_picture_filename), my_picture)
        return os.path.join(self.resources_path, self.my_picture_filename)

    def calculate_average(self):
        return np.mean(Picture.face_pictures, axis=0)

    def get_picture_average(self):
        picture_name = "PictureAverage.jpg"
        cv2.imwrite(os.path.join(self.resources_path, picture_name), self.calculate_average())
        return os.path.join(self.resources_path, picture_name)

    def calculate_distance_my_picture_to_avg(self):
        """
        :return: MSE measures the average difference in pixel values between two images.
        The lower the MSE value, the greater the similarity between the two images.
        """
        #  revisar la distancia euclidiana y distancia Frobenius
        return np.mean((Picture.my_picture - self.calculate_average()) ** 2)
