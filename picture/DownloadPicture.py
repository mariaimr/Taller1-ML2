import os
import re

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def download_pictures():
    path = os.path.dirname(os.path.abspath(__file__))
    path_images = os.path.join(path, "resources/face_pictures")

    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()
    drive = GoogleDrive(gauth)

    folder_id = '1f1aZ4i1lYsRaW9ID76iHfGztKdmAsg21'

    query = f"'{folder_id}' in parents and trashed = false"
    file_list = drive.ListFile({'q': query}).GetList()

    if not os.path.exists(path_images):
        os.mkdir(path_images)

    for file in file_list:
        if 'image' in file['mimeType']:  # Download just pictures
            file.GetContentFile(os.path.join(path_images, re.sub('[^a-zA-Z0-9 \n\.]', '', file['title'])))
