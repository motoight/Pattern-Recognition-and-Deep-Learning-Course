# import cv2
# from skimage import  io
#
# path = '0009_2m_-15P_-10V_-5H_face.jpg'
# img = io.imread(path)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# io.imsave('nconvert.jpg',img)

import dlib
import cv2
from skimage import io
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import traceback

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# print('------ Load file shape_predictor_68_face_landmarks.dat -------\n ')


# img = dlib.load_rgb_image('../input/face22/0009_2m_-15P_-10V_-5H.jpg')


# def corp_face_and_eye(path):
#     img = io.imread(path)
#
#     print(detector(img))
#
#     rect = detector(img)[0]
#     sp = predictor(img, rect)
#     landmarks = np.array([[p.x, p.y] for p in sp.parts()])
#
#     face = landmarks[[*range(17), *range(26, 16, -1)]]
#
#     eye_l = landmarks[[*range(36, 42, 1)]]
#     eye_r = landmarks[[*range(42, 48, 1)]]
#
#     left_eye_r = int((landmarks[39][0] - landmarks[36][0]) * 1.8)
#     right_eye_r = int((landmarks[45][0] - landmarks[42][0]) * 1.8)
#
#     left_eye_cx = (landmarks[39][0] + landmarks[36][0]) // 2 - left_eye_r // 2
#     left_eye_cy = (landmarks[39][1] + landmarks[36][1]) // 2 - left_eye_r // 2
#
#     right_eye_cx = (landmarks[45][0] + landmarks[42][0]) // 2 - right_eye_r // 2
#     right_eye_cy = (landmarks[45][1] + landmarks[42][1]) // 2 - right_eye_r // 2
#
#     x, y, w, h = cv2.boundingRect(face)
#     w, h = max(w, h), max(w, h)
#     face_r = max(w, h)
#     face_x = landmarks[31][0] - face_r // 2
#     face_y = landmarks[31][1] - face_r // 2
#
#     def adjustface(x, y, w, h):
#         x = max(0, x - 20)
#         y = max(0, y - 60)
#         w = h + 20
#         h = h + 60
#         return x, y, w, h
#
#     #     x,y,w,h = adjustRect(x,y,w,h)
#
#     # c_img = img.copy()
#     #     for point in eyes:
#     #         c_img = cv2.circle(c_img,tuple(point),5,(0,255,0),15)
#
#     # fig = plt.figure(figsize=(15, 15))
#     # ax1 = fig.add_subplot(121)
#     #     cv2.rectangle(c_img,(left_eye_cx,left_eye_cy),(left_eye_cx+left_eye_r,left_eye_cy+left_eye_r),(255,0,0),10)
#     r_eye = img[right_eye_cy:right_eye_cy + right_eye_r, right_eye_cx:right_eye_cx + right_eye_r]
#     #     face1 = c_img[y:y+h,x:x+w]
#     #     ax1.imshow(face1)
#
#     l_eye = img[left_eye_cy:left_eye_cy + left_eye_r, left_eye_cx:left_eye_cx + left_eye_r]
#     #     ax2 = fig.add_subplot(122)
#
#     face = img[face_y:face_y + face_r, face_x:face_x + face_r]
#     #     ax2.imshow(face)
#     return [face, l_eye, r_eye]


root = 'columbia_gaze/CAVE'
dst_root = ['gaze/face','gaze/l_eye','gaze/r_eye']
suffix = ['_face.jpg','_le.jpg','_re.jpg']

# finished_dir = ['0009','0010','0011','0012','0013','0014','0015','0016']

# dir_exist = []
failed_path = []
cnt = 0

#将目录按序排列 0001 0002 0003 ...
dir_list = sorted(os.listdir(root))
for item in dir_list:
    if len(item) == 4:
        path = os.path.join(root,item)
        imgs = os.listdir(path)
        for img in imgs:
            if img.endswith('.jpg'):
                cnt+=1
                filename = img.split('.')[0] + suffix[0]
                filepath = os.path.join(dst_root[0],item,filename)

                if not os.path.exists(filepath):
                    print(filepath)
                    failed_path.append(os.path.join(path,img))
                    # results = corp_face_and_eye(filepath)
        #             print(results)
        #             for i in range(3):
        #                 filename = img.split('.')[0] + suffix[i]
        #
        #                 temp_dir = os.path.join(dst_root[i],item)
        #                 if not os.path.exists(temp_dir):
        #                     os.makedirs(temp_dir)
        #                     # dir_exist.append(temp_dir)
        #
        #                 dst_path = os.path.join(dst_root[i],item,filename)
        #                 cv2.imwrite(dst_path,results[i])
        #         except Exception as e:
        #             print('Write: {} failed!'.format(filepath))
        #             traceback.print_exc()
        #             failed_path.append(filepath)

print('total img in CAVE :{}'.format(cnt))
try :
    with open('gaze/failed_path.txt','wb') as ff:
        pickle.dump(failed_path,ff)
    ff.close()
    print('------- dump failed_path: {} items  success ------\n'.format(len(failed_path)))
except:
    print('------- dump failed_path failed ------\n')