from imutils import face_utils
import dlib
from collections import OrderedDict
import numpy as np
import cv2


FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])

def swap_pts_fmt(pts):
    if isinstance(pts, list):
        return np.array(pts, np.int32)
    elif isinstance(pts, np.ndarray):
        if len(pts.shape) == 2:
            return [(int(x), int(y)) for x, y in pts]
        elif len(pts.shape) == 3:
            return [[(int(x), int(y)) for x, y in pts_] for pts_ in pts]


def indexing_trtiangles(triangles, points):
    pts1 = triangles[:, 0:2]
    pts2 = triangles[:, 2:4]
    pts3 = triangles[:, 4:6]

    ind_pts1 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts1]
    ind_pts2 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts2]
    ind_pts3 = [np.where((pt == points).all(axis=1))[0][0] for pt in pts3]

    triangles_indxs = [(i, j, k) for i, j, k, tr in zip(ind_pts1, ind_pts2, ind_pts3, triangles)]
    return triangles_indxs         


class FaceSwapping():
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat',
                 reference_path='reference.png'):
        
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.init_reference(reference_path)
    
    @staticmethod
    def load_img(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def init_reference(self, reference_path):
        self.ref_img = self.load_img(reference_path)
        self.ref_keypoints = self.find_keypoints(self.ref_img)
        assert len(self.ref_keypoints) > 0
        self.ref_keypoints = self.ref_keypoints[0]
        convexhull = cv2.convexHull(self.ref_keypoints)
        h, w, _ = self.ref_img.shape
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, convexhull, 255)
        self.ref_face = cv2.bitwise_and(self.ref_img, self.ref_img, mask=mask)
        
        rect = cv2.boundingRect(convexhull)
        points = swap_pts_fmt(self.ref_keypoints)
        #Make traingulations
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)
        # Extract traingulations indexes
        self.ref_traingles_indxs = indexing_trtiangles(triangles, self.ref_keypoints)
        
    def find_keypoints(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rects = self.detector(img, 1)
        keypoints = []
        for rect in rects:
            shape = self.predictor(img, rect)
            shape = face_utils.shape_to_np(shape)
            keypoints.append(shape)
        return swap_pts_fmt(keypoints)
    
    
    @staticmethod
    def swap_faces(src_img, src_pt, dst_img, dst_pt, tr):
        
        h, w, _ = dst_img.shape
        convexhull = cv2.convexHull(dst_pt)

        face_mask = np.zeros((h, w), np.uint8)
        head_mask = cv2.fillConvexPoly(face_mask, convexhull, 255)
        face_mask = cv2.bitwise_not(head_mask)

        old_head = cv2.bitwise_and(dst_img, dst_img, mask=face_mask)
        new_face = np.zeros_like(dst_img)

        for idxs in tr:
            
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
            
        
        result = cv2.add(old_head, new_face)

        (x, y, w, h) = cv2.boundingRect(convexhull)
        center = (x + w // 2, y + h // 2)
        result = cv2.seamlessClone(result, dst_img, head_mask, center, cv2.MIXED_CLONE)
        
        return result
    
    def __call__(self, img):
        keypoints = self.find_keypoints(img)
        h, w, _ = self.ref_img.shape
        new_img = img.copy()

        for points in keypoints:
            new_img = self.swap_faces(self.ref_img, self.ref_keypoints,
                                      new_img, points,
                                      self.ref_traingles_indxs)
        return new_img
