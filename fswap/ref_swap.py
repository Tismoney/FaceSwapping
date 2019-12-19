from imutils import face_utils
import dlib
import numpy as np
import cv2
from matplotlib import pyplot as plt

from .base import BaseFaceSwapping
from .utils import *

class RefFaceSwapping(BaseFaceSwapping):
    
    @staticmethod
    def swap_faces(src_img, src_pt, dst_img, dst_pt, tr):
        
        h, w, _ = dst_img.shape
        convexhull = cv2.convexHull(dst_pt)

        face_mask = np.zeros((h, w), np.uint8)
        head_mask = cv2.fillConvexPoly(face_mask, convexhull, 255)
        face_mask = cv2.bitwise_not(head_mask)

        old_head = cv2.bitwise_and(dst_img, dst_img, mask=face_mask)
        new_face = np.zeros_like(dst_img)
        i = 0
        for idxs in tr:
            i += 1
            idxs = list(idxs)
            
            # Source Face
            src_tr = src_pt[idxs]
            (x, y, w, h) = cv2.boundingRect(src_tr)
            src_tr = src_tr - np.array([x, y], np.int32)
            src_cropped_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(src_cropped_mask, src_tr, 255)
            src_cropped_tr = src_img[y: y + h, x: x + w]
            # src_cropped_tr = cv2.bitwise_and(src_cropped_tr, src_cropped_tr, mask=src_cropped_mask)

            # Destination Face
            dst_tr = dst_pt[idxs]
            (x, y, w, h) = cv2.boundingRect(dst_tr)
            dst_tr = dst_tr - np.array([x, y], np.int32)
            dst_cropped_mask = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(dst_cropped_mask, dst_tr, 255)
            dst_cropped_tr = dst_img[y: y + h, x: x + w]
            dst_cropped_tr = cv2.bitwise_and(dst_cropped_tr, dst_cropped_tr, mask=dst_cropped_mask)

            # Affine
            M = cv2.getAffineTransform(np.float32(src_tr), np.float32(dst_tr))
            warped_tr = cv2.warpAffine(src_cropped_tr, M, (w, h))
            warped_tr = cv2.bitwise_and(warped_tr, warped_tr, mask=dst_cropped_mask)
            
            #Reconstract face
            new_face_rect = new_face[y: y + h, x: x + w]
            new_face_rect_gray = cv2.cvtColor(new_face_rect, cv2.COLOR_RGB2GRAY)
            _, mask_tr_designed = cv2.threshold(new_face_rect_gray, 1, 255, cv2.THRESH_BINARY_INV)
            warped_tr = cv2.bitwise_and(warped_tr, warped_tr, mask=mask_tr_designed)
            new_face_rect = cv2.add(new_face_rect, warped_tr)
            new_face[y: y + h, x: x + w] = new_face_rect
            
#             plt.figure(figsize=(16,9))
#             plt.imshow(new_face)
#             plt.axis(False)
#             plt.savefig(f'imgs/step5/face_{i}.png')
        
        result = cv2.add(old_head, new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull)
        center = (x + w // 2, y + h // 2)
        result = cv2.seamlessClone(result, dst_img, head_mask, center, cv2.NORMAL_CLONE)
        
        return result
    
    def __call__(self, img):
        keypoints = self.find_keypoints(img)
        h, w, _ = self.ref_img.shape
        new_img = img.copy()

        for points in keypoints:
            try:
                new_img = self.swap_faces(self.ref_img, self.ref_keypoints,
                                          new_img, points,
                                          self.ref_traingles_indxs)
            except:
                new_img = new_img
        return new_img
