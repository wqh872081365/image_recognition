# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


"""
# 简介
# 图6-1中的矩阵
img = np.array([
    [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]],
], dtype=np.uint8)

# 用matplotlib存储  RGB
plt.imsave('img/img_pyplot.jpg', img)

# 用OpenCV存储  BGR
cv2.imwrite('img/img_cv2.jpg', img)
"""

"""
# 存取图像
# 读取一张400x600分辨率的图像
color_img = cv2.imread('img/test_240*240.jpg')
print(color_img.shape)

# 直接读取单通道
gray_img = cv2.imread('img/test_240*240.jpg', cv2.IMREAD_GRAYSCALE)
print(gray_img.shape)

# 把单通道图片保存后，再读取，仍然是3通道，相当于把单通道值复制到3个通道保存
cv2.imwrite('img/test_grayscale.jpg', gray_img)
reload_grayscale = cv2.imread('img/test_grayscale.jpg')
print(reload_grayscale.shape)

# cv2.IMWRITE_JPEG_QUALITY指定jpg质量，范围0到100，默认95，越高画质越好，文件越大
cv2.imwrite('img/test_imwrite_best.jpg', color_img, (cv2.IMWRITE_JPEG_QUALITY, 100))

# cv2.IMWRITE_PNG_COMPRESSION指定png质量，范围0到9，默认3，越高文件越小，画质越差
cv2.imwrite('img/test_imwrite_best.png', color_img, (cv2.IMWRITE_PNG_COMPRESSION, 0))
"""

"""
# 缩放，裁剪和补边
# 读取一张四川大录古藏寨的照片
img = cv2.imread('img/test_240*240.jpg')

# 缩放成200x200的方形图像
img_200x200 = cv2.resize(img, (200, 200))

# 不直接指定缩放后大小，通过fx和fy指定缩放比例，0.5则长宽都为原来一半
# 等效于img_200x300 = cv2.resize(img, (300, 200))，注意指定大小的格式是(宽度,高度)
# 插值方法默认是cv2.INTER_LINEAR，这里指定为最近邻插值
img_120x120 = cv2.resize(img, (0, 0), fx=0.5, fy=0.5,
                              interpolation=cv2.INTER_NEAREST)

# 在上张图片的基础上，上下各贴50像素的黑边，生成300x300的图像
img_240x340 = cv2.copyMakeBorder(img, 50, 50, 0, 0,
                                       cv2.BORDER_CONSTANT,
                                       value=(0, 0, 0))

# 对照片中树的部分进行剪裁  原点在左上角，横轴为第一个元素！
patch_tree = img[20:60, 0:180]

cv2.imwrite('img/cropped_tree.jpg', patch_tree)
cv2.imwrite('img/resized_200x200.jpg', img_200x200)
cv2.imwrite('img/resized_200x300.jpg', img_120x120)
cv2.imwrite('img/bordered_300x300.jpg', img_240x340)
"""


"""
# 色调，明暗，直方图和Gamma曲线
# 通过cv2.cvtColor把图像从BGR转换到HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# H空间中，绿色比黄色的值高一点，所以给每个像素+15，黄色的树叶就会变绿
turn_green_hsv = img_hsv.copy()
turn_green_hsv[:, :, 0] = (turn_green_hsv[:, :, 0]+15) % 180
turn_green_img = cv2.cvtColor(turn_green_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('turn_green.jpg', turn_green_img)

# 减小饱和度会让图像损失鲜艳，变得更灰
colorless_hsv = img_hsv.copy()
colorless_hsv[:, :, 1] = 0.5 * colorless_hsv[:, :, 1]
colorless_img = cv2.cvtColor(colorless_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('colorless.jpg', colorless_img)

# 减小明度为原来一半
darker_hsv = img_hsv.copy()
darker_hsv[:, :, 2] = 0.5 * darker_hsv[:, :, 2]
darker_img = cv2.cvtColor(darker_hsv, cv2.COLOR_HSV2BGR)
cv2.imwrite('darker.jpg', darker_img)
"""

