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

def nms(hm, kernel=3):
    hmax = MaxPool2D(pool_size=(kernel,kernel), strides=(1,1), padding='same')(hm)
    keep = K.cast(K.equal(hm,hmax), 'float32')
    return multiply([keep,hmax])

def topk(heat, k=13):
    hm = heat.reshape(-1)
    max_ind = np.argsort(hm)[-k:][::-1]
    max_val = hm[max_ind]

    return max_val, max_ind

def to_range(val):
    if val<0:
        return 0
    if val>127:
        return 127

    return int(val)

def to_bboxes(hm, wh, do_nms=True, nms_kernel=3):
    if do_nms:
        hm_input = K.constant(np.expand_dims(hm, axis=0), dtype='float32')
        heat = nms(hm_input, kernel = nms_kernel)
        heat = K.eval(heat)[0]
    else:
        heat = hm

    scores, inds = topk(heat)
    #Shapes (13,) (13,) (128, 128, 2) (128, 128, 1)
    # inds: 4613,6197,4895, 6002, 4612
    bboxes = []
    for i in range(len(inds)):
        xx = inds[i] % 128
        yy = inds[i] // 128

        w = wh[yy][xx][0]
        h = wh[yy][xx][1]

        x1 = to_range(xx-w/2)
        x2 = to_range(xx+w/2)
        y1 = to_range(yy-h/2)
        y2 = to_range(yy+h/2)

        if (x2-x1)>0 and (y2-y1)>0:
            bboxes.append((x1,y1,x2,y2, scores[i]))
    return bboxes


def ap_iou_bb(a,b):
    # TODO
    return 0.0

def calc_ap_iou(hm, wh, gt_hm, gt_wh):
    
    pred_bboxes = to_bboxes(hm, wh)
    gt_bboxes = to_bboxes(gt_hm, gt_wh)

    return ap_iou_bb(pred_bboxes, gt_bboxes)


def preview_bbox(path, input, hm, wh, gt_hm, gt_wh, threshold=0.4, nms_kernel=7):
    img = np.clip(input *255,0,255).astype('uint8')

    pred_bboxes = to_bboxes(hm, wh, do_nms=False, nms_kernel=nms_kernel)
    gt_bboxes = to_bboxes(gt_hm, gt_wh, do_nms=False)

    for x1,y1,x2,y2, score in pred_bboxes:
        x1,y1,x2,y2 = 4*x1,4*y1,4*x2,4*y2 
        if score>threshold:
            img = cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
            print(score)

    for x1,y1,x2,y2,_ in gt_bboxes:
        x1,y1,x2,y2 = 4*x1,4*y1,4*x2,4*y2 
        img = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)

        cv2.imwrite(path, img)
