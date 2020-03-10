import cv2
import numpy as np

def save_image(path, array):
    print(f"Saving {path}: {array.shape}")
    img = array.copy()
    if img.shape[2]==3:
        img = np.clip(img * 255,0,255).astype('uint8')
        cv2.imwrite(path, img)
    else:
        img = np.mean(img, axis=2)
        img = np.clip(img * 255,0,255).astype('uint8')
        cv2.imwrite(path, img)

def save_batch_images(idx, input, masks, heatmaps, wh):
    print("IDX",idx)
    for n in range(input.shape[0]):
        filename_base=f"./temp/preview_batch{idx}_n{n}_"
        save_image(filename_base+"input.png", input[n])
        save_image(filename_base+"masks.png", masks[n])
        save_image(filename_base+"heatmaps.png", heatmaps[n])
        save_image(filename_base+"wh.png", wh[n])