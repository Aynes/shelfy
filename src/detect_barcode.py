import numpy as np
import imutils
import cv2
import pytesseract
import re
from pathlib import Path
from pyzbar import pyzbar
import pandas as pd
from tqdm import tqdm
from loguru import logger

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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2000))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 10)
    closed = cv2.dilate(closed, None, iterations = 10)
    return closed

def get_counturs(image):
    cnts = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def get_crop(c, image, padding=PADDING):
    max_h, max_w, _ = image.shape
    rect = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
    box = np.int0(box)

    x1, x2, x3, x4 = box

    xmin = min(x1[0], x2[0],x3[0], x4[0])
    xmax = max(x1[0], x2[0],x3[0], x4[0])

    ymin = min(x1[1], x2[1],x3[1], x4[1])
    ymax = max(x1[1], x2[1],x3[1], x4[1])

    xmin = max(0, round(xmin - padding*(xmax-xmin)))
    xmax = min(max_w, round(xmax + padding*(xmax-xmin)))

    ymin = max(0, ymin - round(padding*(ymax-ymin)))
    ymax = min(max_h, ymax + round(padding*(ymax-ymin)))

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
    # cv2.imwrite(f'c.png',closed,)
    return closed

def step_3(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
    gradX = cv2.Sobel(gray, ddepth=ddepth, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, ddepth=ddepth, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
    blurred = cv2.blur(gradient, (1, 500))
    # cv2.imwrite(f'last.png',blurred,)
    (_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    # cv2.imwrite(f'last.png',closed,)
    return closed

def recognize_page(path_to_image, path_to_csv):
    logger.info(f'Current image: {path_to_image}')
    image = cv2.imread(path_to_image)


    closed = step_1(image)

    column_conturs = get_counturs(closed)
    table_data = []
    for i, conutur_culumn in tqdm(enumerate(column_conturs)):

        crop_column = get_crop(conutur_culumn, image)
        cv2.imwrite(f'column.png',crop_column ,)
        closed = step_2(crop_column)
        barcode_conturs = get_counturs(closed)
        for j, contur_barcode in tqdm(enumerate(barcode_conturs)):
            crop_barcode = get_crop(contur_barcode, crop_column)

            (_, crop_barcode) = cv2.threshold(crop_barcode, 225, 255, cv2.THRESH_BINARY)

            cv2.imwrite(f'tmp.png',crop_barcode,)

            text = pytesseract.image_to_string(crop_barcode, config="-c tessedit_char_whitelist=0123456789ToOоО --psm 6 ").split('\n')[0]
            text = re.sub('[^A-Za-z0-9]+', '', text)
            if text.startswith('7'):
                text = 'T' + text[1:]
            if text.startswith('4'):
                text = '1' + text[1:]
            text = text.replace('O', '0')
            text = text.replace('o', '0')
            text = text.replace('О', '0')
            text = text.replace('о', '0')

            if text.startswith('1') or text.startswith('T'):
                pass
            else:
                cv2.imwrite(f'errors/{Path(path_to_image).stem}_{i}_{j}.png',crop_barcode,)
                continue

            if len(text) < 4:
                cv2.imwrite(f'errors/{Path(path_to_image).stem}_{i}_{j}.png',crop_barcode,)
                continue
            # closed = step_3(crop_barcode)
            # barcode_only_cntr = get_counturs(closed)

            decoded_objects = pyzbar.decode(crop_barcode)
            if len(decoded_objects) == 1:
                line = {'text': text, 'barcode': int(decoded_objects[0].data)}
                table_data.append(line)
            else:
                cv2.imwrite(f'errors/{Path(path_to_image).stem}_{i}_{j}.png',crop_barcode,)
    df = pd.DataFrame.from_dict(table_data)
    df.to_csv(path_to_csv)

if __name__ == "__main__":
    recognize_page(PATH, 'result.csv')
