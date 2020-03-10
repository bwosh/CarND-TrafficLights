import cv2
import numpy as np
from keras.layers import MaxPool2D, Input
from keras import backend as K
from keras.layers.merge import multiply

def save_image(path, array,max_div=False, verbose=True):
    if verbose:
        print(f"Saving {path}: {array.shape}")
    img = array.copy()
    if img.shape[2]==3:
        img = np.clip(img * 255,0,255).astype('uint8')
        cv2.imwrite(path, img)
    else:
        img = np.mean(img, axis=2)
        eps=1e-4
        img = img/(np.max(img)+eps)
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

def nms(hm):
    hmax = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(hm)
    keep = K.cast(K.equal(hm,hmax), 'float32')
    return multiply([keep,hmax])

def topk(heat, k=13):
    hm = heat.reshape(-1)
    max_ind = np.argsort(hm)[-k:][::-1]
    print(type(max_ind))
    max_val = hm[max_ind]

    return max_val, max_ind

def to_bboxes(hm, wh, threshold=0.3):
    hm_input = K.constant(np.expand_dims(hm, axis=0), dtype='float32')
    heat = nms(hm_input)
    heat = K.eval(heat)[0]
    scores, inds = topk(heat)

    bboxes = []
    for i in range(len(inds)):
        if scores[i]<threshold:
            continue

        yy = inds[i] % 128
        xx = inds[i] // 128

        w = wh[xx][yy][0]
        h = wh[xx][yy][1]
        x1 = int(xx-w/2)
        x2 = int(xx+w/2)
        y1 = int(yy-h/2)
        y2 = int(yy+h/2)

        bboxes.append((x1,y1,x2,y2))
    return bboxes

def calc_map(hm, wh, gt_hm, gt_wh):
    
    pred_bboxes = to_bboxes(hm, wh)
    gt_bboxes = to_bboxes(gt_hm, gt_wh)

    # TODO
    return 0.0