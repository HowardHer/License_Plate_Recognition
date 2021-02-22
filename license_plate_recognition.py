import cv2
import numpy as np
import math
import pytesseract
import os
import re
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ratio_check(len_top,len_left):
    if len_top < len_left: # needs to be a horizontal rectangle
        return False
    if len_top/len_left < 1.5 or len_top/len_left > 2.7:
        return False
    else:
        print('is license plate')
        return True

def is_rectangle(top_left,top_right,bottom_left,bottom_right):
    len_top = math.sqrt(math.pow(top_right[0]-top_left[0],2)+math.pow(top_right[1]-top_left[1],2))
    len_bottom = math.sqrt(math.pow(bottom_right[0]-bottom_left[0],2)+math.pow(bottom_right[1]-bottom_left[1],2))
    len_right = math.sqrt(math.pow(top_right[0]-bottom_right[0],2)+math.pow(top_right[1]-bottom_right[1],2))
    len_left = math.sqrt(math.pow(top_left[0] - bottom_left[0], 2) + math.pow(top_left[1] - bottom_left[1], 2))

    print('sides: '+ str(len_top) + ', ' + str(len_bottom) + ', ' + str(len_right) + ', ' + str(len_left) + '\n')

    if (len_top < len_bottom*0.9) | (len_bottom < len_top*0.9):
        return False
    elif (len_right < len_left*0.9) | (len_left < len_right*0.9):
        return False
    elif ratio_check(len_top,len_left) == False:
        return False
    else:
        return True

def perspective_transform(cnts,image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        font = cv2.FONT_HERSHEY_COMPLEX
        points = list()

        if len(approx) == 4:  # Select the contour with 4 corners
            NumberPlateCnt = approx  # This is our approx Number Plate Contour
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)

            n = approx.ravel()
            i = 0

            for j in n:
                if (i % 2 == 0):
                    x = n[i]
                    y = n[i + 1]

                    # String containing the co-ordinates.
                    string = str(x) + " " + str(y)
                    cv2.putText(image, string, (x, y), font, 0.5, (0, 255, 0))
                    p = Point()
                    p.x = x
                    p.y = y
                    points.append(p)

                i = i + 1

                points.sort(key=takeFirst)

            if points[0].y > points[1].y:
                tempx = points[1].x
                tempy = points[1].y
                points[1].x = points[0].x
                points[1].y = points[0].y
                points[0].x = tempx
                points[0].y = tempy

            if points[2].y > points[3].y:
                tempx = points[3].x
                tempy = points[3].y
                points[3].x = points[2].x
                points[3].y = points[2].y
                points[2].x = tempx
                points[2].y = tempy

            top_left = [points[0].x, points[0].y]
            top_right = [points[2].x, points[2].y]
            bottom_left = [points[1].x, points[1].y]
            bottom_right = [points[3].x, points[3].y]

            if is_rectangle(top_left, top_right, bottom_left, bottom_right) == True:
                print('Is rectangle\n')
            else:
                print('Is not rectangle\n')
                points.clear()
                continue
            break

    pts1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    pts2 = np.float32([[0, 0], [114, 0], [0, 48], [114, 48]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result = cv2.warpPerspective(gray, matrix, (114, 48))
    
    return result

def plate_number_formatting(ini_string):
    getVals = list([val for val in ini_string
                    if val.isupper() or val.isnumeric()])
    result = "".join(getVals)
    return result

def takeFirst(p):
    return p.x

def autoCanny(img, sigma=0.5):
    v = np.median(img)
    low_threshold = int(max(0, (1.0-sigma)*v))
    high_threshold = int(min(255, (1.0+sigma)*v))

    return cv2.Canny(img, low_threshold, high_threshold)

class Point:
    x = 0.0
    y = 0.0

def license_plate_detection(path,i):
    image = cv2.imread(path)

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    edged = autoCanny(closing)
    cv2.imwrite('./edge/canny_edge' + str(i) + '.jpg', edged)

    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

    print('Number is :', plate_number_formatting(pytesseract.image_to_string(perspective_transform(cnts, image), lang='eng')))
    cv2.imwrite('./result/result' + str(i) + '.jpg', image)

def main():
    path = '../license_photos_sample/picked_sample'
    files = os.listdir(path)
    print(files)
    i = 0

    for file in files:
        try:
            license_plate_detection(file,i)
        except UnboundLocalError:
            pass
        except cv2.error:
            pass
        i = i + 1
    

if __name__ == "__main__":
  main()