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
                det_east_nms_thresh=0.6, det_sast_nms_thresh=0.6) 

# detect icon use yolov5
model = torch.hub.load("yolov5", "custom", path="last.pt", source="local")  # local repo
model.conf = 0.8
model.max_det = 1000
model.cpu()

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


# combined detect box text
def combine_box_text(input_image):

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
        kernel = np.ones((4, 1), np.uint8)
        dilate_img = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(dilate_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        return contours


# get color classification
def color_classification(input_img, color):

    hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
    lower_color, upper_color = color

    gray_image = cv2.inRange(hsv, lower_color, upper_color)

    res = cv2.bitwise_and(input_img,input_img, mask= gray_image)
    black_mask = np.all(res == [0, 0, 0], axis=2)

    res[black_mask] = [255, 255, 255]
    return res


# return box text, text
def get_text_information(input_image):

    list_box_text = []

    temp_image = input_image
    
    contours = combine_box_text(input_image)
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


def combine_text_box(list_box_text, img_name):

    global list_box_icon
    global exception_path
    global check_used_icon

    list_value = []
    new_list_box_text = []

    for list in list_box_text:

        for lt in list:
            new_list_box_text.append(lt)

    for itt, boxtt in enumerate(new_list_box_text):
        dis_min = 1e+10
        min_index = 0
        my_check = False
        for ilb, boxlb in enumerate(list_box_icon):
            status, dis_temp = get_coord_min(boxtt[0], boxlb)
            if status:
                my_check = True
                if dis_temp < dis_min:
                    dis_min = dis_temp
                    min_index = ilb
            else:
                continue
        if my_check:
            x_left_text, y_top_text, x_right_text, y_bottom_text = boxtt[0]
            text_text = boxtt[1]
            x_left_label, y_top_label, x_right_label, y_bottom_label = list_box_icon[min_index]
            x_circle, y_circle = int((x_left_label + x_right_label) / 2), y_bottom_label
            get_value = (x_circle, y_circle, text_text)
            list_value.append(get_value)

            if check_used_icon[min_index] == 1:
                txt_name = img_name + '.txt'
                with open(os.path.join(exception_path, txt_name), 'a') as file:
                    file.write(f"{x_circle},{y_circle},{text_text}\n")
            else:
                check_used_icon[min_index] = 1

    return list_value


# return status distance, min distance
def combine_text_box2(list_box_text, img_name):

    global list_box_icon
    global exception_path
    global check_used_icon

    # list_box_icon_tensor = torch.tensor(list_box_icon, dtype=torch.float16).cuda()
    # list_box_text_tensor = torch.tensor(list_box_text, dtype=torch.float16).cuda()

    list_value = []
    x_distance = []
    y_distance = []

    for ilb, boxlb in enumerate(list_box_icon):
        x_left_label, y_top_label, x_right_label, y_bottom_label = boxlb
        x_label_center, y_label_center = (x_left_label + x_right_label) / 2, (y_top_label + y_bottom_label) / 2

        check_save = False
        x_min = 1e+5
        y_min = 1e+5
        x_index = 0
        y_index = 0

        for itt, boxtt in enumerate(list_box_text):
            x_left_text, y_top_text, x_right_text, y_bottom_text = boxtt[0]
            x_dis = (x_left_text <= x_label_center <= x_right_text)
            y_dis = (y_top_text <= y_label_center <= y_bottom_text)

            if x_dis:
                y = min(abs(y_label_center - y_top_text), abs(y_label_center - y_bottom_text))
                if y < y_min and y < 44:
                    y_min = y
                    y_index = itt
                    check_save = True
            if y_dis:
                x = min(abs(x_label_center - x_left_text), abs(x_label_center - x_right_text))
                if x < x_min and x < 44:
                    x_min = x
                    x_index = itt
                    check_save = True
        print(x_min, y_min)
        if check_save:
            if y_min < x_min:
                index_text = y_index
            else:
                index_text = x_index

            text_text = list_box_text[index_text][1]
            x_circle, y_circle = int((x_left_label + x_right_label) / 2), y_bottom_label
            get_value = (x_circle, y_circle, text_text)
            list_value.append(get_value)

            if check_used_icon[ilb] == 1:
                txt_name = img_name + '.txt'
                with open(os.path.join(exception_path, txt_name), 'a') as file:
                    file.write(f"{x_circle},{y_circle},{text_text}\n")
            else:
                check_used_icon[ilb] = 1

    return list_value

def combine_text_box3(list_box_text, img_name):

    global list_box_icon
    global exception_path
    global check_used_icon

    list_value = []
    x_distance = []
    y_distance = []

    list_box_icon_tensor = torch.tensor(list_box_icon, dtype=torch.float32).cuda() # upload to gpu
    list_box_text_tensor = torch.tensor([box[0] for box in list_box_text], dtype=torch.float32).cuda() # upload to gpu

    for ilb, boxlb in enumerate(list_box_icon_tensor):
        x_label_center, y_label_center = (boxlb[0] + boxlb[2]) / 2, (boxlb[1] + boxlb[3]) / 2

        check_save = False
        x_min = 1e+5
        y_min = 1e+5
        x_index = 0
        y_index = 0

        for itt, boxtt in enumerate(list_box_text_tensor):
            x_left_text, y_top_text, x_right_text, y_bottom_text = boxtt
            x_dis = (x_left_text <= x_label_center <= x_right_text)
            y_dis = (y_top_text <= y_label_center <= y_bottom_text)

            if x_dis:
                y = min(abs(y_label_center - y_top_text), abs(y_label_center - y_bottom_text))
                if y < y_min and y < 45:
                    y_min = y
                    y_index = itt
                    check_save = True
            if y_dis:
                x = min(abs(x_label_center - x_left_text), abs(x_label_center - x_right_text))
                if x < x_min and x < 45:
                    x_min = x
                    x_index = itt
                    check_save = True

        if check_save:
            if y_min < x_min:
                index_text = y_index
            else:
                index_text = x_index

            text_text = list_box_text[index_text][1]
            x_circle, y_circle = int((boxlb[0] + boxlb[2]) / 2), boxlb[3]
            get_value = (x_circle, y_circle, text_text)
            list_value.append(get_value)

            if check_used_icon[ilb] == 1:
                txt_name = img_name + '.txt'
                with open(os.path.join(exception_path, txt_name), 'a') as file:
                    file.write(f"{x_circle},{y_circle},{text_text}\n")
            else:
                check_used_icon[ilb] = 1

    return list_value

def save_xy_text(output_folder, txt_name, values):

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

    global list_color
    global list_box_icon
    global check_used_icon
    image_files = os.listdir(images_path)

    for image_file in image_files:

        start = time.time()
        image_path = os.path.join(images_path, image_file)

        if os.path.isfile(image_path) and image_path.lower().endswith(('png')):

            input_img = cv2.imread(image_path)
            img_name = os.path.splitext(image_file)[0]

            list_box_icon = []

            # remove all icon
            crop_icon_image, list_box_icon = get_box_icon(image_path)
            cv2.imwrite('crop.png', crop_icon_image)
            if not list_box_icon:
                print(f"image {image_file} has no icon")
            else:  
                check_used_icon = np.zeros(len(list_box_icon)).astype(int)
                for color in list_color:

                    list_box_text = []

                    # image for each label
                    classification_img = color_classification(crop_icon_image, color)
                    infor = get_text_information(classification_img) # return text coord, text
                    if infor:
                        list_box_text = infor
                        value = combine_text_box3(list_box_text, img_name)
                        txt_name = img_name + '.txt'
                        save_xy_text(output_data, txt_name, value)
    
        print(f"Total time: {time.time() - start}s")

# images_path = '/home/minhthanh/Desktop/advanced_images/save_images/'
# labels_path = '/home/minhthanh/Desktop/advanced_images/save_labels/'
# output_data = '/home/minhthanh/Desktop/advanced_images/lat_long-text/'

images_path = '/home/minhthanh/Desktop/test_temp_img/'
labels_path = '/home/minhthanh/Desktop/test_temp_label/'
output_data = '/home/minhthanh/Desktop/lat_long_text_temp/' # save lat, long, text
exception_path = '/home/minhthanh/Desktop/lat_long_text_temp/save_exception/'

list_color = []
# food store
lower_yellow = np.array([15,118,211])
upper_yellow = np.array([16,255,233])

# closing store
lower_blue = np.array([107,77,231])
upper_blue = np.array([109,229,245])

# company
lower_gray = np.array([95,17,118])
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

list_color.append(yellow)
list_color.append(blue)
list_color.append(gray)
list_color.append(pink)
list_color.append(red)
list_color.append(greenblue)
list_color.append(bank)

# inference all
if __name__ == "__main__":

    process_images_in_folder(images_path)

