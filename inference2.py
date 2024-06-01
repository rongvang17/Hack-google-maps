import easyocr
import cv2
import numpy as np
import os
import time
import torch
import json

from paddleocr import PaddleOCR


reader = easyocr.Reader(['en', 'vi'], gpu=False) # read text by use Easyocr
ocr = PaddleOCR(use_angle_cls=False, use_gpu=False, lang='vi', dilation=True, # detect box text by use PaddleOCR
                det_db_box_thresh=0.5, det_limit_side_len=2200, use_dilation=True, 
                det_east_nms_thresh=0.6, det_sast_nms_thresh=0.6, show_log=False)

# detect icon use yolov5
model = torch.hub.load("yolov5", "custom", path="last.pt", source="local")  # local repo
model.conf = 0.8
model.max_det = 1000
# model.cuda()

# agument image
def agument_image(bgr_image):

    gray_crop_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    up_gray_crop_image = cv2.pyrUp(gray_crop_image)
    blur_img = cv2.GaussianBlur(up_gray_crop_image,(3,3), 0)
    kernel = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])
    agument_img = cv2.filter2D(blur_img, -1, kernel)

    return agument_img

# return coordinates of truth-detected text
def combine_box_and_text_detection(input_image):

    mask = np.zeros_like(input_image, dtype=np.uint8)

    result = ocr.ocr(input_image, cls=False)

    if result[0] == None:
        print("no find text in image")

    else:
        result_length = sum(len(item) for item in result)
        list_box = []

        for index in range(result_length):
            boxes = [line[index] for line in result]
            left = int(boxes[0][0][0][0])
            top = int(boxes[0][0][0][1])
            right = int(boxes[0][0][2][0])
            bottom = int(boxes[0][0][2][1])
            value = (left, top, right, bottom)
            list_box.append(value)

        for (l, t, r, b) in list_box:
            cv2.rectangle(mask, (l, t), (r, b), (255, 255, 255), -1)

        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5, 1), np.uint8)
        dilate_img = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return contours


# get color classification
def classify_color(input_img, color):

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = color

    gray_image = cv2.inRange(hsv, lower_color, upper_color)

    res = cv2.bitwise_and(input_img,input_img, mask= gray_image)
    # black_mask = np.all(res == [0, 0, 0], axis=2)

    # res[black_mask] = [255, 255, 255]
    return res


# return box text, text
def get_text_information(input_image):

    list_box_text = []

    temp_image = input_image
    
    contours = combine_box_and_text_detection(input_image)
    if contours is None:
        print("images has icon but no text")
        return

    else:
        for contour in contours:
            if cv2.contourArea(contour) > 25:

                [X, Y, W, H] = cv2.boundingRect(contour)
                text_coord = (X, Y, X+W, Y+H)

                crop_image = temp_image[Y:(Y+H), X:(X+W)]

                agument_img = agument_image(crop_image)
                result = reader.readtext(agument_img, detail=1, paragraph=True)

                for _, text in result:
                    list_box_text.append((text_coord, text))

        return list_box_text

# return lat-long-text
def combine_text_box(list_box_text, img_name):

    global list_box_icon
    global exception_path
    global check_icon_used

    list_value = []

    for itt, boxtt in enumerate(list_box_text):
        x_left_text, y_top_text, x_right_text, y_bottom_text = boxtt[0]

        check_save = False
        x_min = 1e+5
        y_min = 1e+5
        x_index = 0
        y_index = 0

        for ilb, boxlb in enumerate(list_box_icon):
            x_label_center, y_label_center = (boxlb[0] + boxlb[2]) / 2, (boxlb[1] + boxlb[3]) / 2
            x_dis = (x_left_text <= x_label_center <= x_right_text)
            y_dis = (y_top_text <= y_label_center <= y_bottom_text)

            if x_dis:
                y = min(abs(y_label_center - y_top_text), abs(y_label_center - y_bottom_text))
                if y < y_min and y <= 50:
                    y_min = y
                    y_index = ilb
                    check_save = True
            if y_dis:
                x = min(abs(x_label_center - x_left_text), abs(x_label_center - x_right_text))
                if x < x_min and x <= 50:
                    x_min = x
                    x_index = ilb
                    check_save = True
        if check_save:
            if y_min < x_min:
                index_icon = y_index
            else:
                index_icon = x_index

            text_text = boxtt[1]
            x_circle, y_circle = int((list_box_icon[index_icon][0] + list_box_icon[index_icon][2]) / 2), list_box_icon[index_icon][3]
            get_value = (x_circle, y_circle, text_text)
            list_value.append(get_value)

            if check_icon_used[index_icon] == 0:
                check_icon_used[index_icon] = 1
            else:
                txt_name = img_name + '.txt'
                with open(os.path.join(exception_path, txt_name), 'a') as file:
                    file.write(f"{x_circle},{y_circle},{text_text}\n")

    return list_value


# save lat-long-text in file .txt
def save_lat_long_text(output_folder, txt_name, values):

    with open(os.path.join(output_folder, txt_name), 'a') as file:
        for value in values:
            xi, yi, text = value
            file.write(f"{xi},{yi},{text}\n")

# get box icon by yolov5
def get_box_icon(image_path):

    crop_icon_img = cv2.imread(image_path)
    results = model(image_path)
    data = results.pandas().xyxy[0].to_json(orient="records")  # JSON img1 predictions
    json_data = json.loads(data)

    if not json_data:
        return image_path, []

    list_box_yolov5 = []
    for record in json_data:
        value = (int(record['xmin']), int(record['ymin']), int(record['xmax']), int(record['ymax']))
        crop_icon_img[int(record['ymin']):int(record['ymax']), int(record['xmin']):int(record['xmax'])] = (255, 255, 255)
        list_box_yolov5.append(value)

    return crop_icon_img, list_box_yolov5


# inference all
def process_images_in_folder(images_path):

    global list_colors
    global list_box_icon # save box icon
    global check_icon_used # save icon used
    image_files = os.listdir(images_path)

    for image_file in image_files:

        start = time.time()
        image_path = os.path.join(images_path, image_file)

        if os.path.isfile(image_path) and image_path.lower().endswith(('png')):

            list_box_icon = []
            input_img = cv2.imread(image_path)
            img_name = os.path.splitext(image_file)[0]

            # remove all icon
            crop_icon_image, list_box_icon = get_box_icon(image_path)
            if not list_box_icon:
                print(f"image {image_file} has no icon")
            else:
                check_icon_used = np.zeros(len(list_box_icon), dtype=int)
                for color in list_colors:

                    list_box_text = []

                    # image for each label
                    classify_img = classify_color(crop_icon_image, color)
                    infor = get_text_information(classify_img) # return text coord, text
                    if infor:
                        list_box_text = infor
                        value = combine_text_box(list_box_text, img_name)
                        if value:
                            txt_name = img_name + '.txt'
                            save_lat_long_text(output_data, txt_name, value)

        print(f"Total time: {time.time() - start}s")

images_path = '/home/minhthanh/Desktop/test_temp_img/'
output_data = '/home/minhthanh/Desktop/lat_long_text_temp/' # save lat, long, text
exception_path = '/home/minhthanh/Desktop/lat_long_text_temp/save_exception/'

# food store
lower_yellow = np.array([15,118,211])
upper_yellow = np.array([16,255,233])

# closing store
lower_blue = np.array([107,77,231])
upper_blue = np.array([109,229,245])

# company
lower_gray = np.array([95,30,118])
upper_gray = np.array([101,91,196])

# hotel
lower_pink = np.array([168,57,231])
upper_pink = np.array([171,224,249])

# pharmacity
lower_red = np.array([1,95,215])
upper_red = np.array([3,213,237])

# musume 
lower_greenblue = np.array([92,69,143])
upper_greenblue = np.array([94,243,206])

# bank and petrol station
lower_bank = np.array([114,89,191])
upper_bank = np.array([117,135,222])

# park 
lower_park = np.array([68, 68, 128])
upper_park = np.array([69, 208, 190])

yellow = (lower_yellow, upper_yellow)
blue = (lower_blue, upper_blue)
gray = (lower_gray, upper_gray)
pink = (lower_pink, upper_pink)
red = (lower_red, upper_red)
greenblue = (lower_greenblue, upper_greenblue)
bank = (lower_bank, upper_bank)
park = (lower_park, upper_park)

list_colors = []
list_colors.append(yellow)
list_colors.append(blue)
list_colors.append(gray)
list_colors.append(pink)
list_colors.append(red)
list_colors.append(greenblue)
list_colors.append(bank)
list_colors.append(park)

# inference all
if __name__ == "__main__":
    process_images_in_folder(images_path)
