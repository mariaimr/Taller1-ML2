>### If you want to run the dimensional reduction API, rename main_dim_red.py to main.py, for the clustering API you can keep the main file without changing its name

## Requirements
 > pip install -r requirements.txt

## How to Run?
> uvicorn main:app --reload

## Create client_secrets.json file with your credentials
This file is a **mandatory** requirement to allow the application to connect to Google Drive and download the images
> https://cloud.google.com/bigquery/docs/authentication/end-user-installed?hl=es-419#client-credentials 

## Open FastAPI to run endpoints
> http://127.0.0.1:8000/docs

