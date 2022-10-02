"""
文件名：main.py
说明：1.检测视频中的行人位置
     2.对行人进行跟踪计数
     3.对行人进行性别和年龄识别并进行统计
     4.对设定好的时间点time1和time2进行上述检测
2022-09-06: 江绪好, Davy @Davis Tech
"""

from detect import detRecPerson
from bisect import bisect_left


def main(yoloV5_head, age_gender_model, video_path, list_pts_blue, list_pts_yellow, time1, time2):
    """
     main(yolov5_head, gender_age_model,video_path,time1,time2)
     .   @brief 说明：1.检测视频中的行人位置 2.对行人进行跟踪计数 3.对行人进行性别和年龄识别并进行统计 4.对设定好的时间点time1和time2进行上述检测
     .   @param yolov5_head: 指定yolov5的人头检测模型的文件路径
     .   @param gender_age_model: 指定行人的性别和年龄检测模型的文件路径
     .   @param video_path: 指定需要检测的视频路径
     .   @param list_pts_blue: 指定需要检测的上线位置，如list_pts_blue = [[0, 393 * 2], [960 * 2, 393 * 2], [960 * 2, 432 * 2], [0, 432 * 2]]
     .   @param list_pts_yellow: 指定需要检测的下线位置，如list_pts_yellow = [[0 * 2, 488 * 2], [960 * 2, 488 * 2], [960 * 2, 526 * 2], [0, 526 * 2]]
     .   @param time1: 指定需要检测视频的初始时间
     .   @param time2: 指定需要检测视频的最终时间
     """

    down_age_gender = detRecPerson(yoloV5_head=yoloV5_head, gender_age_model=age_gender_model, video_path=video_path,
                                   list_pts_blue=list_pts_blue, list_pts_yellow=list_pts_yellow)

    down_age = []
    down_gender = []
    down_count = 0

    count_male = 0
    count_female = 0

    ageList = [0, 10, 20, 30, 40, 50, 60, 70, 100]
    ageCount = [0, 0, 0, 0, 0, 0, 0, 0]

    for l in down_age_gender:
        gender = l.split('-')[1]
        age = int(l.split('-')[0])
        time = int(l.split('-')[2])

        if time1 < time < time2:
            down_age.append(age)
            down_gender.append(gender)
            down_count += 1

            if gender == 'Male':
                count_male += 1
            elif gender == 'Female':
                count_female += 1

            index = bisect_left(ageList, age)
            ageCount[index - 1] += 1

    print('在{}-{}小时内：'.format(time1 / 60 / 60, time2 / 60 / 60))
    print('\t进入总人数：', down_count)
    print('\t进入乘客性别：')
    print('\t' + str(down_gender) + '\n')
    print('\t进入乘客年龄：')
    print('\t' + str(down_age) + '\n')
    print('\t乘客0-10岁共：', str(ageCount[0]))
    print('\t乘客10-20岁共：', str(ageCount[1]))
    print('\t乘客20-30岁共：', str(ageCount[2]))
    print('\t乘客30-40岁共：', str(ageCount[3]))
    print('\t乘客40-50岁共：', str(ageCount[4]))
    print('\t乘客50-60岁共：', str(ageCount[5]))
    print('\t乘客60-70岁共：', str(ageCount[6]))
    print('\t乘客70以上共：', str(ageCount[7]))
    print('\t乘客男性共：', str(count_male))
    print('\t乘客女性共：', str(count_female))


if __name__ == '__main__':
    yoloV5_head = 'Head_Detect_best.onnx'
    age_gender_model = 'MobileNetV3_age_gender-best.onnx'
    video_path = 'video/test4.mp4'
    list_pts_blue = [[0, 393 * 2], [960 * 2, 393 * 2], [960 * 2, 432 * 2], [0, 432 * 2]]
    list_pts_yellow = [[0 * 2, 488 * 2], [960 * 2, 488 * 2], [960 * 2, 526 * 2], [0, 526 * 2]]
    time1 = 0
    time2 = 60 * 60  # 一个小时，换算为秒
    main(yoloV5_head, age_gender_model, video_path, list_pts_blue, list_pts_yellow, time1, time2)

