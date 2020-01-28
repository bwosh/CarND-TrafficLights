import os
import shutil

from utils.download import download

annotations_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

def download_annotations(output_folder):
    file_name = os.path.basename(annotations_url)
    target_path = os.path.join(output_folder, file_name)
    temp_path = target_path+"_"

    if os.path.isfile(target_path):
        print(f"{target_path} already downloaded.")
    else:
        print(f"Downloading annotations (241MB) to \"{target_path}\"...")
        download(annotations_url, temp_path)
        shutil.move(temp_path, target_path)

def unzip_annotations(output_folder):
    print("Unzipping annotations...")

def extract_class_annotations(output_folder, class_name):
    print("Extracting annotations...")

def download_missing_images(output_folder, annotations):
    print("Downloading images...")

def generate_segmentation_images(output_folder, annotations):
    print("Creating segmentation images...")

def download_data(output_folder, class_name):
    download_annotations(output_folder)
    unzip_annotations(output_folder)
    annotations = extract_class_annotations(output_folder, class_name)
    download_missing_images(output_folder, annotations)
    generate_segmentation_images(output_folder, annotations)



download_data('../data/coco','traffic lights')