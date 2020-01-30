import argparse
import cv2
import json
import numpy as np
import os
import shutil
import zipfile

from utils.download import download, show_progress

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

def get_annotations_from_json(path, class_name):
    with open(path,"r") as file:
        data = json.load(file)
    categoryId = None
    for category in data['categories']:
        if category['name']==class_name:
            categoryId = category['id']

    if categoryId is None:
        raise Exception(f"ERROR: Category {class_name} not found!")

    imageData = {}

    skipped_image_ids = []

    for annotations in data['annotations']:
        if annotations['category_id'] == categoryId:
            segmentation = annotations['segmentation']
            image_id = annotations['image_id']

            if any(['counts' in s for s in segmentation]):
                skipped_image_ids.append(image_id)
                continue

            if image_id in imageData:
                imageData[image_id]['annotations'].append(segmentation)
            else:
                imageData[image_id] = {'annotations':[segmentation]}

    for image in data['images']:
        if image['id'] in imageData and not image['id'] in skipped_image_ids:
            imageData[image['id']]['url'] = image['coco_url']
            imageData[image['id']]['file_name'] = image['file_name']
            imageData[image['id']]['height'] = image['height']
            imageData[image['id']]['width'] = image['width']

    return imageData

def extract_class_annotations(output_folder, class_name):
    print("Extracting annotations...")

    ann_folder = os.path.join(output_folder, 'annotations')
    train_path = os.path.join(ann_folder,'instances_train2017.json')
    val_path = os.path.join(ann_folder,'instances_val2017.json')

    train_annotations = get_annotations_from_json(train_path, class_name)
    val_annotations = get_annotations_from_json(val_path, class_name)

    print(f"[*] Images found: TRAIN-{len(train_annotations)} VAL-{len(val_annotations)}")
    return train_annotations, val_annotations

def get_missing_files_from_annotations(output_folder, ann_set):
    missing = []
    for imageId in ann_set:
        imageData = ann_set[imageId]
        imageName = imageData['file_name']
        imageUrl = imageData['url']

        target_location = os.path.join(output_folder, imageName)

        if not os.path.isfile(target_location):
            missing.append((target_location, imageUrl))

    return missing

def download_missing_images(output_folder, annotations):
    print("Downloading images...")
    train, val = annotations

    missing_train = get_missing_files_from_annotations(output_folder, train)
    missing_val = get_missing_files_from_annotations(output_folder, val)

    missing = missing_train + missing_val
    count = 0
    all = len(missing)

    for file_name, url in missing: 
        show_progress(count,1, all)
        try:
            temp_path = file_name+"_"
            if not os.path.isfile(file_name):
                download(url, temp_path)
                shutil.move(temp_path, file_name)       
        except:
            print("Issue with ", file_name)
        count+=1
        show_progress(count,1, all)

def apply_segc(img_orig, h, w, s, color_val):
    img = np.zeros((h,w), dtype='uint8')
    img = img.reshape(-1)
    steps = np.array(s[1:]).reshape(-1,2)
    offset=s[0]
    for on,off in steps:
        img[offset:offset+on] = color_val
        offset += on+off
    
    img = img.reshape(w,h).transpose(1,0)
    return np.maximum(img_orig,np.repeat(np.expand_dims(img,axis=2),3,axis=2))

def generate_segmentation_image(target_path, height, width, segmentation_data):
    target_path = ".".join(target_path.split('.')[:-1])+"_seg.png"
    img = np.zeros((height, width,3), dtype='uint8')
    color = 255
    for seg_idx, seg in enumerate(segmentation_data):
        if 'counts' in seg:
            segc = seg['counts']
            img = apply_segc(img, height, width, segc, color )
        else:
            for s in seg:
                s = np.array(s).reshape(1,-1,2).astype(int)
                cv2.fillPoly(img,s, (color,color,color))

        color -= 1
        if color<=1:
            raise Exception("Too many instances to distinguish in 255 values of gray.")
    cv2.imwrite(target_path, img)

def generate_segmentation_images(output_folder, annotations):
    print("Creating segmentation images...")
    train, val = annotations
    all = {**train, **val}
    for imageId in all:
        image = all[imageId]
        file_name = image['file_name']
        height = image['height']
        width = image['width']
        segmentation_data = image['annotations']
        generate_segmentation_image(os.path.join(output_folder, file_name), height, width, segmentation_data)

def download_data(output_folder, class_name):
    download_annotations(output_folder)
    unzip_annotations(output_folder)
    annotations = extract_class_annotations(output_folder, class_name)
    download_missing_images(output_folder, annotations)
    generate_segmentation_images(output_folder, annotations)

# TODO: add args
#download_data('../data/coco','traffic light')