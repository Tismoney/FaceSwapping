from imutils import face_utils
import dlib
import numpy as np
import cv2

from .utils import *

class BaseFaceSwapping():
    def __init__(self, predictor_path='shape_predictor_68_face_landmarks.dat',
                 reference_path='reference.png', color_format='RGB'):
        
        self.color_format = color_format
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.init_reference(reference_path)
    
    def load_img(self, path):
        img = cv2.imread(path)
        if self.color_format == 'RGB':
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
