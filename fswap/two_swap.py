from imutils import face_utils
import dlib
import numpy as np
import cv2

from .base import BaseFaceSwapping
from .utils import *

from matplotlib import pyplot as plt

class TwoFaceSwapping(BaseFaceSwapping):
    
    @staticmethod
    def swap_faces(img, src_pt, dst_pt, tr):
        
        h, w, _ = img.shape

        src_convexhull = cv2.convexHull(src_pt)
        dst_convexhull = cv2.convexHull(dst_pt)

        src_face_mask = np.zeros((h, w), np.uint8)
        dst_face_mask = np.zeros((h, w), np.uint8)
        
        src_head_mask = cv2.fillConvexPoly(src_face_mask, src_convexhull, 255)
        dst_head_mask = cv2.fillConvexPoly(dst_face_mask, dst_convexhull, 255)

        src_face_mask = cv2.bitwise_not(src_head_mask)
        dst_face_mask = cv2.bitwise_not(dst_head_mask)

        heads_mask = src_head_mask + dst_head_mask
        faces_mask = cv2.bitwise_not(heads_mask)

        # old_heads = cv2.bitwise_and(img, img, mask=faces_mask)
        # old_faces = cv2.bitwise_and(img, img, mask=heads_mask)
        
        src_new_face = np.zeros_like(img)
        dst_new_face = np.zeros_like(img)
        
        for idxs in tr:
            
            idxs = list(idxs)
            
            # Source Face
            src_tr = src_pt[idxs]
            (src_x, src_y, src_w, src_h) = cv2.boundingRect(src_tr)
            src_tr = src_tr - np.array([src_x, src_y], np.int32)
            src_cropped_mask = np.zeros((src_h, src_w), np.uint8)
            cv2.fillConvexPoly(src_cropped_mask, src_tr, 255)
            src_cropped = img[src_y: src_y + src_h, src_x: src_x + src_w]
            src_cropped_tr = cv2.bitwise_and(src_cropped, src_cropped, mask=src_cropped_mask)

            # Destination Face
            dst_tr = dst_pt[idxs]
            (dst_x, dst_y, dst_w, dst_h) = cv2.boundingRect(dst_tr)
            dst_tr = dst_tr - np.array([dst_x, dst_y], np.int32)
            dst_cropped_mask = np.zeros((dst_h, dst_w), np.uint8)
            cv2.fillConvexPoly(dst_cropped_mask, dst_tr, 255)
            dst_cropped = img[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w]
            dst_cropped_tr = cv2.bitwise_and(dst_cropped, dst_cropped, mask=dst_cropped_mask)

            # Affine
            src_M = cv2.getAffineTransform(np.float32(src_tr), np.float32(dst_tr))
            dst_M = cv2.getAffineTransform(np.float32(dst_tr), np.float32(src_tr))
            
            dst_warped_tr = cv2.warpAffine(src_cropped, src_M, (dst_w, dst_h))
            src_warped_tr = cv2.warpAffine(dst_cropped, dst_M, (src_w, src_h))

            src_warped_tr = cv2.bitwise_and(src_warped_tr, src_warped_tr, mask=src_cropped_mask)
            dst_warped_tr = cv2.bitwise_and(dst_warped_tr, dst_warped_tr, mask=dst_cropped_mask)
            
            #Reconstract face
            dst_new_face_rect = dst_new_face[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w]
            dst_new_face_rect_gray = cv2.cvtColor(dst_new_face_rect, cv2.COLOR_RGB2GRAY)
            _, dst_mask_tr_designed = cv2.threshold(dst_new_face_rect_gray, 1, 255, cv2.THRESH_BINARY_INV)
            dst_warped_tr = cv2.bitwise_and(dst_warped_tr, dst_warped_tr, mask=dst_mask_tr_designed)
            dst_new_face_rect = cv2.add(dst_new_face_rect, dst_warped_tr)
            dst_new_face[dst_y: dst_y + dst_h, dst_x: dst_x + dst_w] = dst_new_face_rect

            src_new_face_rect = src_new_face[src_y: src_y + src_h, src_x: src_x + src_w]
            src_new_face_rect_gray = cv2.cvtColor(src_new_face_rect, cv2.COLOR_RGB2GRAY)
            _, src_mask_tr_designed = cv2.threshold(src_new_face_rect_gray, 1, 255, cv2.THRESH_BINARY_INV)
            src_warped_tr = cv2.bitwise_and(src_warped_tr, src_warped_tr, mask=src_mask_tr_designed)
            src_new_face_rect = cv2.add(src_new_face_rect, src_warped_tr)
            src_new_face[src_y: src_y + src_h, src_x: src_x + src_w] = src_new_face_rect
            
        src_result = cv2.bitwise_and(img, img, mask=src_face_mask)
        src_result = cv2.add(src_result, src_new_face)
        (x, y, w, h) = cv2.boundingRect(src_convexhull)
        center = (x + w // 2, y + h // 2)
        src_result = cv2.seamlessClone(src_result, img, src_head_mask, center, cv2.NORMAL_CLONE)


        dst_result = cv2.bitwise_and(img, img, mask=dst_face_mask)
        dst_result = cv2.add(dst_result, dst_new_face)
        (x, y, w, h) = cv2.boundingRect(dst_convexhull)
        center = (x + w // 2, y + h // 2)
        dst_result = cv2.seamlessClone(dst_result, img, dst_head_mask, center, cv2.NORMAL_CLONE)

        src_face = cv2.bitwise_and(src_result, src_result, mask=src_head_mask)
        src_head = cv2.bitwise_and(dst_result, dst_result, mask=src_face_mask)

        result = cv2.add(src_face, src_head)

        return result
    
    def __call__(self, img):
        keypoints = self.find_keypoints(img)
        h, w, _ = self.ref_img.shape
        
        new_img = img.copy()
        if len(keypoints) >= 2:
            new_img = self.swap_faces(new_img, keypoints[0] , keypoints[1],
                                      self.ref_traingles_indxs)
        return new_img
