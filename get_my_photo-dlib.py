# -*- codeing: utf-8 -*-
"""
打开摄像头，获取被识别人的图片
标记特征，处理图片
"""
import os
import dlib
import random
import cv2

# 输出文件位置
output_dir = './my_photo'
# 设定输出图片大小
size = 64

# 没有文件夹创造文件夹也要写代码
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 亮度调整是将图像像素的强度整体变大/变小
# 对比度调整是指图像暗处的像素强度变低，亮处的变高
# 从而拓宽某个区域内的显示精度
def relight(img, light=1, bias=0):
    w = img.shape[1]  # 宽度
    h = img.shape[0]  # 高度
    for i in range(0, w):
        for j in range(0, h):
            for c in range(3):
                tmp = int(img[j, i, c] * light + bias)
                if tmp > 255:
                    tmp = 255
                elif tmp < 0:
                    tmp = 0
                img[j, i, c] = tmp
    return img


# 使用dlib自带的frontal_face_detector作为特征提取器
detector = dlib.get_frontal_face_detector()
# 打开摄像头 参数为输入流，要是视频文件参数直接为文件路径
camera = cv2.VideoCapture(0)

index = 1
while True:
    if (index <= 10000):
        print('[ INFO ] : 正在获取图片--> %s' % index)
        # 从摄像头读取照片
        success, img = camera.read()
        # 转为灰度图片,如果是读取视频文件
        # 这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
        if success is True:
            # 转为灰度图片
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            break
        # 使用detector进行人脸检测
        dets = detector(gray_img, 1)

        # enumerate(dets)将可遍历的数组对象串成索引序列
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            y1 = d.bottom() if d.bottom() > 0 else 0
            x2 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0

            face = img[x1:y1, x2:y2]
            # 调整图片的对比度与亮度， 对比度与亮度值都取随机数，这样能增加样本的多样性
            face = relight(face, random.uniform(0.5, 1.5), random.randint(-50, 50))
            # 重新定义图片大小
            face = cv2.resize(face, (size, size))
            # 打开窗口显示已获取的图片
            cv2.imshow('image', face)
            # 存入文件夹
            cv2.imwrite(output_dir + '/' + str(index) + '.jpg', face)
            index += 1

        # 一般在imshow()后要使用waitKey()，给图像绘制留下时间，不然窗口会出现无响应情况，并且图像无法显示出来
        if cv2.waitKey(30) & 0xff == 27:
            break
    else:
        print('[ INFO ] : Finished!')
        break

# 释放资源
camera.release()
camera.destroyAllWindows()
