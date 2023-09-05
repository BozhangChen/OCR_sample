import cv2
from matplotlib import pyplot as plt
import time
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes, VisualFeatureTypes
import pandas as pd

# Create ComputerVisionClient
# Need modification here
EndPoint = 'https://bozhangchen.cognitiveservices.azure.com/'
Key = '816f2d27fef64c169d3f2fa9541192ab'
cv_client = ComputerVisionClient(endpoint=EndPoint,
                                 credentials=CognitiveServicesCredentials(Key))

# Preprocessing images
image_file = '/Users/kenneth/Desktop/Tolstoy/Image_1.jpg'
img = cv2.imread(image_file)
def binarize(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh, img_bw = cv2.threshold(img_gray, 210, 230, cv2.THRESH_BINARY)
    return img_bw
img_bw = binarize(img)

# Draw boundary
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (9, 9), 0)
img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (25,25))
img_dilate = cv2.dilate(img_thresh, kernal, iterations=2)

# Draw Contours
cnts = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key= lambda x: cv2.boundingRect(x)[1])
img_cnts = img_bw.copy()
imgs = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 2000 and h < 2000:
        imgs.append(img_bw[y:y+h, x:x+w])
        cv2.rectangle(img_cnts, (x, y), (x+w, y+h), cv2.COLORMAP_DEEPGREEN, 5)

# Extract value
for i in range(3):
    cv2.imwrite('temp.jpg', imgs[i])
    response1 = cv_client.read_in_stream(open('temp.jpg','rb'), language='en', raw=True)
    operationLocation1 = response1.headers['Operation-Location']
    operation_id1 = operationLocation1.split('/')[-1]
    time.sleep(5)
    result1 = cv_client.get_read_result(operation_id1)
    frame = {}
    if result1.status == OperationStatusCodes.succeeded:
        read_results = result1.analyze_result.read_results
        for analyzed_result in read_results:
            analyzed_result_lines = analyzed_result.lines
            if i == 1:
                frame[analyzed_result_lines[0].text] = ''
                line = 1
            else:
                line = 0
            while line < len(analyzed_result_lines):
                line_text = analyzed_result_lines[line].text
                if ':' in line_text:
                    line_text_sp = line_text.split(':')
                    key = line_text_sp[0]
                    value = line_text_sp[1]
                    while (line+1) < len(analyzed_result_lines) and ':' not in analyzed_result_lines[line+1].text:
                        value+=analyzed_result_lines[line+1].text
                        line+=1
                    frame[key] = value
                line+=1
    df = pd.DataFrame(data=frame, index=['value'])
    df = (df.T)
    df.to_excel('Frame {}.xlsx'.format(int(i+1)))
