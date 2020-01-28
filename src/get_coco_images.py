import os
import shutil
import zipfile

from utils.download import download

annotations_url='http://images.cocodataset.org/annotations/annotations_trainval2017.zip'

def download_annotations(output_folder):
    print("Downloading annotations...")
    file_name = os.path.basename(annotations_url)
    target_path = os.path.join(output_folder, file_name)
    temp_path = target_path+"_"

    if os.path.isfile(target_path):
        print(f"[*] {target_path} already downloaded.")
    else:
        print(f"Downloading 241MB to \"{target_path}\"...")
        download(annotations_url, temp_path)
        shutil.move(temp_path, target_path)

def unzip_annotations(output_folder):
    print("Unzipping annotations...")
    file_name = os.path.basename(annotations_url)
    zip_path = os.path.join(output_folder, file_name)
    ann_folder = os.path.join(output_folder, 'annotations')

    if os.path.isdir(ann_folder) and len(os.listdir(ann_folder))>=6:
        print("[*] Already unzipped.")
        return
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_folder)

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