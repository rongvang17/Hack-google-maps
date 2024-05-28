import easyocr
import cv2
import numpy as np
import os
import time
import torch

from paddleocr import PaddleOCR


start = time.time()

reader = easyocr.Reader(['en', 'vi'], gpu=False) # read text by Easyocr
ocr = PaddleOCR(use_angle_cls=False, use_gpu=False, lang='vi', dilation=True,  # get box text by Paddleocr
                det_db_box_thresh=0.5, det_limit_side_len=2200, use_dilation=True, 
                det_east_nms_thresh=0.6, det_sast_nms_thresh=0.6) 


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


# remove icon
def remove_icon(input_image, label_path):

    height, width, chanel = input_image.shape
    crop_icon_img = input_image
    temp_list_box_icon = []

    with open(label_path, 'r') as file:
        lines = file.readlines()

    for line in lines:
        class_id, x_center, y_center, box_width, box_height = map(float, line.split())

        left = int((x_center - box_width / 2) * width)
        top = int((y_center - box_height / 2) * height)
        right = int((x_center + box_width / 2) * width)
        bottom = int((y_center + box_height / 2) * height)
        coord = (left, top, right, bottom)

        temp_list_box_icon.append(coord)
        crop_icon_img[top:bottom, left:right] = (255, 255, 255)

    return crop_icon_img, temp_list_box_icon


# combined detect box
def combine_box(input_image):

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
    
    contours = combine_box(input_image)
    if contours is None:
        print("images has icon but no text")
        return

    else:

        for contour in contours:

            if cv2.contourArea(contour) > 20:

                [X, Y, W, H] = cv2.boundingRect(contour)
                text_coord = (X, Y, X+W, Y+H)

                crop_image = temp_image[Y:(Y+H), X:(X+W)]

                agument_img = agument_image(crop_image)
                result = reader.readtext(agument_img, detail=1, paragraph=True)

                for _, text in result:
                    list_box_text.append((text_coord, text))

        return list_box_text


# return status distance, min distance
def get_coord_min(boxtt, boxlb, max_distance):

    boxtt = torch.tensor(boxtt)
    boxlb = torch.tensor(boxlb)
    max_distance = torch.tensor(max_distance)

    x_left_label, y_top_label, x_right_label, y_bottom_label = boxlb
    x_left_text, y_top_text, x_right_text, y_bottom_text = boxtt

    

    x_label_center, y_label_center = (x_left_label + x_right_label) / 2, (y_top_label + y_bottom_label) / 2

    x_distance, y_distance = torch.abs(x_label_center - torch.tensor([x_left_text, x_right_text])), torch.abs(y_label_center - torch.tensor([y_top_text, y_bottom_text]))

    x_min, y_min = torch.min(x_distance), torch.min(y_distance)
    check_distance = (x_min <= max_distance) and (y_min <= max_distance)
    check_place = (x_left_text <= x_label_center <= x_right_text) or (y_top_text <= y_label_center <= y_bottom_text)

    if check_distance and check_place:

        return check_distance, (x_min.item() + y_min.item())
    
    return False, 0


def combine_text_box(list_box_text):

    list_value = []
    new_list_box_text = []
    max_distance = 65

    global list_box_icon

    for list in list_box_text:

        for lt in list:
            new_list_box_text.append(lt)

    for itt, boxtt in enumerate(new_list_box_text):
        dis_min = 1e+10
        min_index = 0
        my_check = False
        for ilb, boxlb in enumerate(list_box_icon):
            status, dis_temp = get_coord_min(boxtt[0], boxlb, max_distance)
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
            # list_box_icon.pop(min_index)

    return list_value

def save_xy_text(output_folder, txt_name, values):

    with open(os.path.join(output_folder, txt_name), 'a') as file:

        for value in values:

            xi, yi, text = value
            file.write(f"{xi},{yi},{text}\n")

        
# inference all
def process_images_in_folder(images_path, labels_path):

    global list_color
    global list_box_icon
    global cnt
    image_files = os.listdir(images_path)

    for image_file in image_files:

        if cnt > 0:
            image_path = os.path.join(images_path, image_file)

            if os.path.isfile(image_path) and image_path.lower().endswith(('png')):

                input_img = cv2.imread(image_path)
                img_name = os.path.splitext(image_file)[0]
                label_path = os.path.join(labels_path, img_name + '.txt')

                if os.path.exists(label_path):

                    list_box_icon = []

                    # remove all icon
                    crop_icon_image, box_icon = remove_icon(input_img, label_path)
                    list_box_icon = box_icon

                    if list_box_icon[0] is None:
                        print(f"image {image_file} has no icon")
                    else:
                        
                        for color in list_color:

                            list_box_text = []

                            # image for each label
                            classification_img = color_classification(crop_icon_image, color)
                            infor = get_text_information(classification_img) # return text coord, text
                            if infor:
                                list_box_text.append(infor)

                            value = combine_text_box(list_box_text)
                            txt_name = img_name + '.txt'
                            save_xy_text(output_data, txt_name, value)

                    cv2.waitKey(0)

                else:
                    print(f'no find label file {img_name}.txt')

        cnt -= 1
        
    print(f"Total excute time: {time.time() - start}s")

# images_path = '/home/minhthanh/Desktop/test_temp_img/' # test image
# labels_path = '/home/minhthanh/Desktop/test_temp_label/'
# output_data = '/home/minhthanh/Desktop/lat_long_text_temp/' # save lat, long, text

images_path = '/home/minhthanh/Desktop/advanced_images/save_images/' # test image
labels_path = '/home/minhthanh/Desktop/advanced_images/save_labels/'
output_data = '/home/minhthanh/Desktop/advanced_images/lat_long-text/'

# get list color detected
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

# bank
lower_bank = np.array([114,90,191])
upper_bank = np.array([117,135,222])


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

    process_images_in_folder(images_path, labels_path)

