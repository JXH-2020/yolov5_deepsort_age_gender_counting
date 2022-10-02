# -*- coding: utf-8 -*-
#############################################################################
# Copyright (c) 2022  - Shanghai Davis Tech, Inc.  All rights reserved
"""
文件名：detect.py
说明：1.检测视频中的行人位置
     2.对行人进行跟踪计数
     3.对行人进行性别和年龄识别并进行统计
2022-08-07: 江绪好, Davy @Davis Tech
"""
import operator
import datetime
import numpy as np
import onnxruntime
import torch
import tracker
from yolov5_onnx import YOLOV5_ONNX
import cv2
import torchvision.transforms as transforms
from bisect import bisect_left


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def detectOnePicture(to_tensor, im, x1, y1, x2, y2, rnet_session):
    """
     detectOnePicture(to_tensor, im, x1, y1, x2, y2, rnet_session) -> gender, age
     .   @brief 说明：检测图片中单个人像的性别和年龄
     .   @param to_tensor: 将resize后的im图像进行tensor数据转换，方便进行推理得到结果
     .   @param im: 需要检测的行人头部的头像
     .   @param x1,x2,y1,y2: 行人头像框的位置（坐标）
     .   @param rnet_session: 性别和年龄检测模型
     .   @return gender,age：返回下行人员的性别和年龄
     """
    names = ['Female', 'Male']
    img = to_tensor(cv2.resize(im[y1:y2, x1:x2], (128, 128))).unsqueeze_(0)
    # compute ONNX Runtime output prediction
    inputs = {rnet_session.get_inputs()[0].name: to_numpy(img)}
    ag_pd, gender_pd = rnet_session.run(None, inputs)
    max_index, max_number = max(enumerate(gender_pd[0]), key=operator.itemgetter(1))
    age = int(ag_pd[0] * 100)
    gender = names[max_index]

    return gender, age


def detRecPerson(yoloV5_head, gender_age_model, video_path, list_pts_blue, list_pts_yellow):
    """
     detRecPerson(yolov5_head, gender_age_model,video_path) -> down_age_gender
     .   @brief 说明：1.检测视频中的行人位置 2.对行人进行跟踪计数 3.对行人进行性别和年龄识别并进行统计
     .   @param yolov5_head: 指定yolov5的人头检测模型的文件路径
     .   @param gender_age_model: 指定行人的性别和年龄检测模型的文件路径
     .   @param video_path: 指定需要检测的视频路径
     .   @return down_age_gender：返回下行人员的性别和年龄列表
     """
    # 根据视频尺寸，填充一个polygon，供撞线计算使用
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # 初始化2个撞线polygon

    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # 填充第二个polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # 撞线检测用mask，包含2个polygon，（值范围 0、1、2），供撞线计算使用
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # 缩小尺寸，1920x1080->960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # 蓝 色盘 b,g,r
    blue_color_plate = [255, 0, 0]
    # 蓝 polygon图片
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # 黄 色盘
    yellow_color_plate = [0, 255, 255]
    # 黄 polygon图片
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # 彩色图片（值范围 0-255）
    color_polygons_image = blue_image + yellow_image
    # 缩小尺寸，1920x1080->960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # list 与蓝色polygon重叠
    list_overlapping_blue_polygon = []

    # list 与黄色polygon重叠
    list_overlapping_yellow_polygon = []

    # 进入数量
    down_count = 0
    # 离开数量
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_TYPE = device.__str__()
    if DEVICE_TYPE == 'gpu':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # 初始化 yolov5 加载人头检测模型
    detector = YOLOV5_ONNX(onnx_path=yoloV5_head, providers=providers)
    # 初始化 加载性别年龄检测模型
    rnet_session = onnxruntime.InferenceSession(gender_age_model, providers=providers)
    to_tensor = transforms.ToTensor()
    # 打开视频
    capture = cv2.VideoCapture(video_path)
    up_age_gender = []
    down_age_gender = []

    count_male = 0
    count_female = 0

    ageList = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    ageCount = [0, 0, 0, 0, 0, 0, 0, 0]

    # Find OpenCV version
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = capture.get(cv2.CAP_PROP_FPS)

    fps_count = 0
    starttime = datetime.datetime.now()
    while capture.read:
        # 读取每帧图片
        _, im = capture.read()
        if im is None:
            break

        # 缩小尺寸，1920x1080->960x540
        im = cv2.resize(im, (960, 540))

        fps_count += 1
        boxes = detector.infer(im)

        list_bboxs = tracker.update(boxes, im)

        output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
        # 输出图片
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            # ----------------------判断撞线----------------------
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                # 撞线检测点，(x1，y1)，y方向偏移比例 0.0~1.0
                y1_offset = int(y1 + ((y2 - y1) * 0.6))

                # 撞线的点
                y = y1_offset
                x = x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    # 如果撞 蓝polygon
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)
                    pass

                    # 判断 黄polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 外出方向
                    if track_id in list_overlapping_yellow_polygon:
                        # 外出+1
                        up_count += 1
                        #
                        # print(
                        #     f'类别: {label} | id: {track_id} | 上行撞线 | 上行撞线总数: {up_count} | 上行id列表: {list_overlapping_yellow_polygon}')
                        up_age_gender.append(label)
                        # 删除 黄polygon list 中的此id
                        list_overlapping_yellow_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    # 如果撞 黄polygon
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)
                    pass

                    # 判断 蓝polygon list 里是否有此 track_id
                    # 有此 track_id，则 认为是 进入方向
                    if track_id in list_overlapping_blue_polygon:
                        # 进入+1
                        down_count += 1

                        gender, age = detectOnePicture(to_tensor, im, x1, y1, x2, y2, rnet_session)
                        label = str(age) + '-' + gender + '-' + str(int(fps_count / fps))
                        if gender == 'Male':
                            count_male += 1
                        elif gender == 'Female':
                            count_female += 1

                        index = bisect_left(ageList, age)
                        ageCount[index - 1] += 1

                        down_age_gender.append(label)
                        # 删除 蓝polygon list 中的此id
                        list_overlapping_blue_polygon.remove(track_id)

                        pass
                    else:
                        # 无此 track_id，不做其他操作
                        pass
                    pass
                else:
                    pass
                pass

            pass

            # ----------------------清除无用id----------------------
            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break
                    pass
                pass

                if not is_found:
                    # 如果没找到，删除id
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    pass
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
                    pass
                pass
            list_overlapping_all.clear()
            pass

            # 清空list
            list_bboxs.clear()

            pass
        else:
            # 如果图像中没有任何的bbox，则清空list
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()
            pass
        pass

        text_draw = 'DOWN:' + str(down_count) + \
                    ', UP:' + str(up_count) + \
                    ', Male:' + str(count_male) + \
                    ', Female:' + str(count_female) + \
                    ', age(10-20):' + str(ageCount[1]) + \
                    ', age(20-30):' + str(ageCount[2]) + \
                    ', age(30-40):' + str(ageCount[3]) + \
                    ', age(40-50):' + str(ageCount[4]) + \
                    ', age(50-60):' + str(ageCount[5]) + \
                    ', age(60-70):' + str(ageCount[6])

        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=0.45, color=(0, 0, 255), thickness=1)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)

        pass
    pass
    endtime = datetime.datetime.now()
    print((endtime - starttime).seconds)
    print(fps_count)
    capture.release()
    cv2.destroyAllWindows()
    return down_age_gender


def detectAgeGender(v_img, v_detector, v_age_session, v_names=['female', 'male'], v_show=True):
    """
     detectAgeGender(v_img, v_boxes, v_to_tensor, v_age_session, v_names, v_show=True) -> list_age, list_gender
     .   @brief 说明：检测一张图片中人的年龄和性别
     .   @param v_img: 指定待检测的图片<class 'numpy.ndarray'>
     .   @param v_detector: 人脸检测模型
     .   @param v_age_session: 性别和年龄检测器
     .   @param v_names: 性别类列表
     .   @param v_show: 是否显示处理后的图片，默认为True
     .   @return list_age, list_gender：返回年龄和性别列表
     """
    v_boxes = v_detector.infer(v_img)
    image = v_img.copy()
    list_age = []
    list_gender = []
    v_to_tensor = transforms.ToTensor()
    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    for (x1, y1, x2, y2, cls_id, pos_id) in v_boxes:
        head = v_img[y1:y2, x1:x2]
        resize_img = v_to_tensor(cv2.resize(head, (112, 112))).unsqueeze_(0)
        inputs = {v_age_session.get_inputs()[0].name: to_numpy(resize_img)}
        ag_pd, gender_pd = v_age_session.run(None, inputs)
        max_index, max_number = max(enumerate(gender_pd[0]), key=operator.itemgetter(1))
        t_age = int(ag_pd[0] * 100)
        t_gender = v_names[max_index]
        list_age.append(t_age)
        list_gender.append(t_gender)
        c1, c2 = (x1, y1), (x2, y2)
        cv2.rectangle(image, c1, c2, (255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
        image = cv2.putText(img=image, text=str(t_age) + t_gender,
                            org=(x1, y1),
                            fontFace=font_draw_number,
                            fontScale=0.45, color=(0, 0, 255), thickness=1)
    if v_show:
        cv2.imshow('fd', image)
        cv2.waitKey(0)

    return list_age, list_gender


if __name__ == '__main__':
    img = cv2.imread('demo.jpg')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_TYPE = device.__str__()
    if DEVICE_TYPE == 'gpu':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']

    # 初始化 加载性别年龄检测模型
    age_session = onnxruntime.InferenceSession('myDefineModel-best.onnx', providers=providers)
    detector = YOLOV5_ONNX(onnx_path='Head_Detect_best.onnx', providers=providers)

    names = ['female', 'male']
    age, gender = detectAgeGender(img, detector, age_session, names, v_show=True)
    print(age)
    print(gender)
