import albumentations as a
import cv2
import numpy as np
import os
import torch

from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset

from get_coco_images import extract_class_annotations
from utils.image import gaussian_radius

from shape import ShapeEncoder

class SingleClassDataset(Dataset):
    def __init__(self, annotations, images_path, width, height, output_shape, augment=False):
        self.annotations = annotations

        self.annotation_keys = list(self.annotations.keys())
        self.max_objs = max([len(annotations[k]['annotations']) for k in annotations])
        self.images_path = images_path
        self.width = width
        self.height = height
        self.output_shape = output_shape
        self.bbox_mode = False
        self.seg_mode = True
        self.augment = augment

        self.aug_set=a.Compose([
            a.HorizontalFlip(p=0.5),
            a.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.25, rotate_limit=30,p=0.5),
            a.RandomBrightness(0.3,p=0.5),
            a.RandomContrast(0.3,p=0.5),
            a.GaussNoise(p=0.5),
            a.Blur(blur_limit=5,p=0.2),
        ],p=0.95)

        self.shape_encoder = ShapeEncoder()

    def __len__(self):
        return len(self.annotations)

    def augment_image(self, img, mask):
        data = self.aug_set(image=img, mask=mask)
        return data['image'], data['mask']

    def get_unchanged(self, index):
        annotation = self.annotations[self.annotation_keys[index]]
        img_path = annotation['file_name']
        img_path = os.path.join(self.images_path, img_path)

        # Input
        img = cv2.imread(img_path)
        
        # Output
        mask = cv2.imread(img_path.replace(".jpg","_seg.png"))

        # To proper resolutions
        img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask,(self.output_shape[0], self.output_shape[1]), interpolation=cv2.INTER_NEAREST)

        return img, mask

    def get_augmented(self, index):
        img, mask = self.get_unchanged(index)
        img, mask = self.augment_image(img, mask)
        return img, mask

    def get_shape_embedding(self, mask, x1, y1, x2, y2):

        boxed = mask[y1:y2+1, x1:x2+1].copy()
        boxed = boxed.astype('uint8')
        resized = cv2.resize(boxed,(32, 32))

        encoded = self.shape_encoder.encode_bvae(resized)

        return encoded

    def to_heatmap_widthandheight_se(self, mask):
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        ind = np.zeros((self.max_objs), dtype=np.int64)

        instance_ids = [i for i in set(list(mask.ravel())) if i != 0]

        heatmap = np.zeros((self.output_shape[1], self.output_shape[0]), dtype=float)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        se = np.zeros((self.output_shape[1], self.output_shape[0], 16), dtype=np.float32)
        bboxes = []
        for idx,instance_id in enumerate(instance_ids):

            y_coords, x_coords = np.where(mask[:,:,0]==instance_id)
            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)
            width = xmax - xmin
            height = ymax - ymin

            cx = xmin + width//2
            cy = ymin + height//2

            ind[idx] = cy*self.output_shape[1]+cx
            reg_mask[idx] = 1

            # Heatmap
            radius = gaussian_radius((width, height))
            temp = np.zeros((self.output_shape[1], self.output_shape[0]), dtype=float)
            temp[cy,cx] = 1
            se[cy,cx] = self.get_shape_embedding(mask[:,:,0]==instance_id, xmin, ymin, xmax, ymax)
            temp = gaussian_filter(temp, radius)
            temp = temp/np.max(temp)
            heatmap = np.maximum(heatmap, temp)
        
            # Width & Height
            wh[idx] = 1. * width, 1. * height

        return heatmap, wh, reg_mask, ind, se

    def __getitem__(self,index):
        if self.augment:
            img, mask = self.get_augmented(index)
        else:
            img, mask = self.get_unchanged(index)

        center_heatmap, widthandheight, reg_mask, ind, se = self.to_heatmap_widthandheight_se(mask)

        # To proper tensors
        img = torch.tensor(img.transpose(2,0,1), dtype=torch.float)/255
        center_heatmap = torch.tensor(np.expand_dims(center_heatmap,axis=0), dtype=torch.float)
        shape_embeddings = torch.tensor(se.transpose(2,0,1), dtype=torch.float)

        return img, center_heatmap, widthandheight, reg_mask, ind, shape_embeddings