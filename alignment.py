# from keras import backend as K
import cv2
from scripts.utils import *
import dlib
import numpy as np
from skimage import io
from IPython import embed
from IPython.terminal.embed import InteractiveShellEmbed
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os
import glob
import sys
import time
from PIL import Image

predictor_path = 'scripts/shape_predictor_68_face_landmarks.dat'

def get_image_contours(img):

    # embed()
    image = (img * 255).astype("uint8")
    cv_image = image
    # images_path = './scripts/test.png'
    # image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # cv_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv_image = expand_image(cv_image)

    height, width = cv_image.shape[:2]
    print(height, width)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    faces = detector(cv_image, 1)

    # get the biggest image
    max_area = 0
    max_face = None
    for face in faces:
        area = (face.bottom() - face.top()) * (face.right() - face.left())
        if max_area < area:
            max_area = area
            max_face = face

    if max_face is not None:
        shape = predictor(cv_image, max_face)
        contours = np.zeros((41, 2), dtype='uint32')
        for i in range(0, 41):
            x = int(shape.part(i+27).x)
            y = int(shape.part(i+27).y)
            contours[i] = [x, y]

        return cv_image, contours, shape
    else:
        print('none face')
        return None, None, None

def make_mask(face_img, contours):

    mask = np.zeros_like(face_img)
    approx = ConvexHull(contours)
    hull_indices = approx.vertices
    approx = contours[hull_indices, :]

    approx = scale_approx(approx)

    cv2.fillConvexPoly(mask, points=approx.astype(np.int64), color=(255,255,255))
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)

    return mask

def scale_approx(approx):
    # initialize
    scale_x = 1.3
    scale_y = 1.4
    sum_x = 0
    sum_y = 0
    ave_x = 0
    ave_y = 0

    # cal sum
    for pt in approx:
        sum_x += pt[0]
        sum_y += pt[1]

    # cal ave
    ave_x = sum_x / len(approx)
    ave_y = sum_y / len(approx)

    # scale img
    for pt in approx:
        pt[0] = (pt[0] - ave_x) * scale_x + ave_x
        pt[1] = (pt[1] - ave_y) * scale_y + ave_y

    return approx

def trim_image(img, lbl):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shape = predictor(gray, rects[0])
    pts = []
    for pt in shape.parts():
        pts.append([pt.x, pt.y])

    pts = np.asarray(pts)

    right_eye = np.mean(pts[36:42, :], axis=0)
    left_eye = np.mean(pts[42:48, :], axis=0)

    eye_gap = left_eye - right_eye
    eye_center = 0.5 * (left_eye + right_eye)
    eye_dist = np.linalg.norm(eye_gap)
    theta = np.arctan2(-eye_gap[1], eye_gap[0])

    h, w = gray.shape

    trans = np.eye(3, 3)
    trans[0:2, :] = np.array([[1.0, 0.0, (w // 2 - eye_center[0])],
                              [0.0, 1.0, (h // 2 - eye_center[1])]])

    rot = np.eye(3, 3)
    scale = 0.20 * w / eye_dist
    rot[0:2, :] = cv2.getRotationMatrix2D((w // 2, h // 2), -np.degrees(theta), scale)

    affine = np.dot(trans, rot)[0:2, :]
    res_img = cv2.warpAffine(img, affine, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    res_lbl = cv2.warpAffine(lbl, affine, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


    return res_img, res_lbl

def trim_image2(img, mask, shape):

    height, width, _ = img.shape
    center_target = np.asarray([width//2, height//2])

    left_eye = np.asarray(extract_left_eye_center(shape))
    right_eye = np.asarray(extract_right_eye_center(shape))
    # print(left_eye)
    # print(right_eye)

    center_eye = (left_eye + right_eye) / 2
    # print(center_eye)
    distance = np.linalg.norm(left_eye - right_eye)
    # print(center_eye)
    # print(distance)

    x_gap = center_target[0] - center_eye[0]
    y_gap = center_target[1] - center_eye[1]

    T = np.float32([[1.0, 0.0, x_gap],[0.0, 1.0, y_gap]])
    img = cv2.warpAffine(img, T, (width, height))
    mask = cv2.warpAffine(mask, T, (width, height))

    angle = angle_between_2_points(left_eye, right_eye)
    M = cv2.getRotationMatrix2D(tuple(center_target), angle, 1)
    img = cv2.warpAffine(img, M, (width, height))
    mask = cv2.warpAffine(mask, M, (width, height))
    # print(angle)

    resize_scale = 36 / distance
    target_width  = width * resize_scale
    target_height = height * resize_scale
    img = cv2.resize(img, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (int(target_width), int(target_height)), interpolation=cv2.INTER_AREA)

    center_target_resize = np.asarray([img.shape[0]//2, img.shape[1]//2])

    left   = center_target_resize[1] - 89
    right  = center_target_resize[1] + 89
    top    = center_target_resize[0] - 112
    bottom = center_target_resize[0] + 106

    crop_det = [int(left), int(top), int(right), int(bottom)]

    if (left >= 0) and (right < img.shape[1]) and (top >= 0) and (bottom < img.shape[0]):
        cropped = crop_image(img, crop_det)
        cropped = cv2.resize(cropped, (178, 218), interpolation=cv2.INTER_AREA)
        cropped_mask = crop_image(mask, crop_det)
        cropped_mask = cv2.resize(cropped_mask, (178, 218), interpolation=cv2.INTER_AREA)
        return cropped, cropped_mask

    else:
        print('not avaiable size')
        return None, None

def get_croped_images(img_path, img_size):
    face_img, contours, shape = get_image_contours(f)
    if face_img is not None:

        mask_img = make_mask(face_img, contours)
        croped_image, croped_mask = trim_image2(face_img, mask_img, shape)
        if croped_image is not None:
            croped_image = cv2.cvtColor(croped_image, cv2.COLOR_RGB2BGR)
            croped_image = croped_image.crop((0, 20, 178, 218-20)).resize((img_size, img_size), Image.BICUBIC)

    return croped_image


def make_mask_from_image(img):
    # images_path = './scripts/test.png'
    img_size = 128

    # face_detect(f)
    face_img, contours, shape = get_image_contours(img)
    if face_img is not None:

        mask_img = make_mask(face_img, contours)
        croped_image, croped_mask = trim_image2(face_img, mask_img, shape)
        if croped_image is not None:
            # croped_mask = cv2.cvtColor(croped_image, cv2.COLOR_RGB2BGR)
            croped_image = Image.fromarray(croped_image)
            croped_mask = Image.fromarray(croped_mask)
            croped_image = croped_image.crop((0, 20, 178, 218-20)).resize((img_size, img_size), Image.BICUBIC)
            croped_mask  = croped_mask.crop((0, 20, 178, 218-20)).resize((img_size, img_size), Image.BICUBIC)

            return croped_image, croped_mask

    else:
        print('there are no face!!!')
