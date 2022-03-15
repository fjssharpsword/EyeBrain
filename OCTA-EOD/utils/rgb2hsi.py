# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

#Chinese Font
#plt.rcParams['font.sans-serif'] = ['SimHei']
#plt.rcParams['axes.unicode_minus'] = False

# show image
def imshow(image):
    if image.ndim == 2:
        plt.imshow(image, cmap='gray')                     # gray
    else:
        plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))   #color


# Font and size of coordinate 
def label_def():
    plt.xticks(fontproperties='Times New Roman', size=8)
    plt.yticks(fontproperties='Times New Roman', size=8)
    plt.axis('off')                                     # 关坐标，可选


# read image
#img_orig = cv.imread('/data/fjsdata/fundus/IDRID/BDiseaseGrading/OriginalImages/TestingSet/IDRiD_002.jpg', 1)    # read fundus image
img_orig = cv.imread('/data/fjsdata/OCTA-Rose/ROSE-2/test/original/59_OS_SVP.png', 1)    # read octa image

# RGB to HSI
def rgb2hsi(image):
    b, g, r = cv.split(image)                    # 读取通道
    r = r / 255.0                                # 归一化
    g = g / 255.0
    b = b / 255.0
    eps = 1e-6                                   # 防止除零

    img_i = (r + g + b) / 3                      # I分量

    img_h = np.zeros(r.shape, dtype=np.float32)
    img_s = np.zeros(r.shape, dtype=np.float32)
    min_rgb = np.zeros(r.shape, dtype=np.float32)
    # 获取RGB中最小值
    min_rgb = np.where((r <= g) & (r <= b), r, min_rgb)
    min_rgb = np.where((g <= r) & (g <= b), g, min_rgb)
    min_rgb = np.where((b <= g) & (b <= r), b, min_rgb)
    img_s = 1 - 3*min_rgb/(r+g+b+eps)                                            # S分量

    num = ((r-g) + (r-b))/2
    den = np.sqrt((r-g)**2 + (r-b)*(g-b))
    theta = np.arccos(num/(den+eps))
    img_h = np.where((b-g) > 0, 2*np.pi - theta, theta)                           # H分量

    img_h = img_h/(2*np.pi)                                                       # 归一化
    temp_s = img_s - np.min(img_s)
    temp_i = img_i - np.min(img_i)
    img_s = temp_s/np.max(temp_s)
    img_i = temp_i/np.max(temp_i)

    image_hsi = cv.merge([img_h, img_s, img_i])
    return img_h, img_s, img_i, image_hsi


# HSI to RGB
def hsi2rgb(image_hsi):
    eps = 1e-6                                                                  # 防止除零
    img_h, img_s, img_i = cv.split(image_hsi)
    image_out = np.zeros((img_h.shape[0], img_h.shape[1], 3))
    img_h = img_h*2*np.pi
    img_r = np.zeros(img_h.shape, dtype=np.float32)
    img_g = np.zeros(img_h.shape, dtype=np.float32)
    img_b = np.zeros(img_h.shape, dtype=np.float32)

    # 扇区1
    img_b = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), img_i * (1 - img_s), img_b)
    img_r = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3),
                     img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_r)
    img_g = np.where((img_h >= 0) & (img_h < 2 * np.pi / 3), 3 * img_i - (img_r + img_b), img_g)

    # 扇区2
    img_r = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), img_i * (1 - img_s), img_r)
    img_g = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3),
                     img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_g)
    img_b = np.where((img_h >= 2*np.pi/3) & (img_h < 4*np.pi/3), 3 * img_i - (img_r + img_g), img_b)

    # 扇区3
    img_g = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), img_i * (1 - img_s), img_g)
    img_b = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi),
                     img_i * (1 + img_s * np.cos(img_h) / (np.cos(np.pi/3 - img_h))), img_b)
    img_r = np.where((img_h >= 4*np.pi/3) & (img_h <= 2*np.pi), 3 * img_i - (img_b + img_g), img_r)

    temp_r = img_r - np.min(img_r)
    img_r = temp_r/np.max(temp_r)

    temp_g = img_g - np.min(img_g)
    img_g = temp_g/np.max(temp_g)

    temp_b = img_b - np.min(img_b)
    img_b = temp_b/np.max(temp_b)

    image_out = cv.merge((img_b, img_g, img_r))                   # 按RGB合并，后面不用转换通道
    print(image_out.shape)
    return image_out


if __name__ == '__main__':                                       # 运行当前函数

    h, s, i, hsi = rgb2hsi(img_orig)                             # RGB到HSI的变换
    img_revise = np.float32(hsi2rgb(hsi))                        # HSI复原到RGB

    # h, s, i = cv.split(cv.cvtColor(img_orig, cv.COLOR_BGR2HSV))       # 自带库函数HSV模型
    im_b, im_g, im_r = cv.split(img_orig)                        # 获取RGB通道数据

    plt.subplot(241), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('RGB Image'), label_def()
    plt.subplot(242), plt.imshow(im_r, 'gray'), plt.title('R'), label_def()
    plt.subplot(243), plt.imshow(im_g, 'gray'), plt.title('G'), label_def()
    plt.subplot(244), plt.imshow(im_b, 'gray'), plt.title('B'), label_def()

    plt.subplot(245), plt.imshow(hsi), plt.title('HSI Image'), label_def()
    plt.subplot(246), plt.imshow(h, 'gray'), plt.title('H'), label_def()
    plt.subplot(247), plt.imshow(s, 'gray'), plt.title('S'), label_def()
    plt.subplot(248), plt.imshow(i, 'gray'), plt.title('I'), label_def()
    plt.savefig('/data/pycode/EyeBrain/imgs/octa_rgb_hsi.png', dpi=100)

    plt.subplot(121), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('RGB Image'), label_def()
    plt.subplot(122), plt.imshow(cv.cvtColor(img_orig, cv.COLOR_BGR2RGB)), plt.title('HSI2RGB'), label_def()
    plt.savefig('/data/pycode/EyeBrain/imgs/octa_hsi2rgb.png', dpi=100)