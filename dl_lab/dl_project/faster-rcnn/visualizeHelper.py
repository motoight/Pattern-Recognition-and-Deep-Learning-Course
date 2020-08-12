import matplotlib.pyplot as plt
import numpy as np
import cv2

def image_convert(image):
    image = image.clone().cpu().numpy()
    image = image.transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    return image

def plot_img(data,idx):
    out = data.__getitem__(idx)
    image = image_convert(out[0])
    image = np.ascontiguousarray(image)
    bb = out[1]['boxes'].numpy()
    for i in bb:
        cv2.rectangle(image, (i[0],i[1]), (i[2],i[3]), (0,255,0), thickness=2)
    plt.figure(figsize=(10,10))
    plt.imshow(image)

def vis_boxes(img, bboxes,scores):
    image = image_convert(img)
    for box,score in zip(bboxes,scores):
        cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (0,255,0), thickness=2)
        text = 'wheat '+str(score)
        cv2.putText(image, text, (box[0],box[1]), font = cv2.FONT_HERSHEY_SIMPLEX,
                   fontScale = 1, color = (255,255,255), thickness = 1)
    plt.figure(figsize=(10,10))
    plt.imshow(image)