import numpy as np
import imutils
import cv2
import pytesseract
import re
from pathlib import Path

PATH = "data/jpg/230208190315_0.jpg"
PADDING = 0.05

def step_1(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))

    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 90))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    closed = cv2.erode(closed, None, iterations = 10)
    closed = cv2.dilate(closed, None, iterations = 10)
    return closed

def get_counturs(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def get_crop(c, image):
    max_h, max_w, _ = image.shape
    # print(image.shape)
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    x1, x2, x3, x4 = box

    xmin = min(x1[0], x2[0],x3[0], x4[0]) 
    xmax = max(x1[0], x2[0],x3[0], x4[0]) 

    ymin = min(x1[1], x2[1],x3[1], x4[1])
    ymax = max(x1[1], x2[1],x3[1], x4[1])

    xmin = max(0, round(xmin - PADDING*(xmax-xmin)))
    xmax = min(max_w, round(xmax + PADDING*(xmax-xmin)))

    ymin = max(0, ymin - round(PADDING*(ymax-ymin)))
    ymax = min(max_h, ymax + round(PADDING*(ymax-ymin)))


    crop = image[ymin: ymax, xmin:xmax] #TODO: разобраться с координатами и делать нормальный кроп
    return crop

def step_2(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (9, 9))
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1000, 20))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    cv2.imwrite(f'c.png',closed,)
    return closed

def step_3(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (1, 500))
    # blurred = cv2.blur(blurred, (20, 20))
    cv2.imwrite(f'last.png',blurred,)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    cv2.imwrite(f'last.png',closed,)
    return closed 

def run(path):
    image = cv2.imread(path)
    closed = step_1(image)
    column_conturs = get_counturs(closed)
    
    for i, conutur_culumn in enumerate(column_conturs):
        crop_column = get_crop(conutur_culumn, image)
        closed = step_2(crop_column)
        barcode_conturs = get_counturs(closed)
        for j, contur_barcode in enumerate(barcode_conturs):
            crop_barcode = get_crop(contur_barcode, crop_column)

            cv2.imwrite(f'{i}_{j}.png', crop_barcode)
            text = pytesseract.image_to_string(crop_barcode).split('\n')[0]
            text = re.sub('[^A-Za-z0-9]+', '', text)
            print(f'{i} {j} - {text}')
            closed = step_3(crop_barcode)
            barcode_only_cntr = get_counturs(closed)
            for k, barcode_only in enumerate(barcode_only_cntr):
                
                barcode_only_crop = get_crop(barcode_only, crop_barcode)
                cv2.imwrite(f'{i}_{j}_{text}.png',barcode_only_crop,)


            # rect = cv2.minAreaRect(contur_barcode)
            # box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
            # box = np.int0(box)

        #     cv2.drawContours(crop_column, [box], -1, (0, 255, 0), 3)
        # cv2.imwrite(f'{i}_{j}.png',crop_column,)
if __name__ == "__main__":
    run(PATH)