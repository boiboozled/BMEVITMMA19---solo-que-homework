import os
import requests
def download_file_to_folder(link, folder_name):
    try:
        # Ensure the folder exists, or create it if it doesn't.
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # Extract the file name from the link.
        file_name = link.split("/")[-1]

        # Define the complete path for the downloaded file.
        file_path = os.path.join(folder_name, file_name)

        # Send a request to the link and download the file.
        response = requests.get(link)
        if response.status_code == 200:
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded '{file_name}' to '{folder_name}' successfully.")
        else:
            print(f"Failed to download '{file_name}'. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

import tarfile

def untar_gz_file(tar_gz_file, output_folder):
    try:
        with tarfile.open(tar_gz_file, 'r:gz') as tar:
            tar.extractall(path=output_folder)
        print(f"Extracted '{tar_gz_file}' to '{output_folder}' successfully.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Download the data.
download_file_to_folder("https://snap.stanford.edu/data/facebook.tar.gz", "./data")

# Extract the data.
untar_gz_file("./data/facebook.tar.gz", "./data")