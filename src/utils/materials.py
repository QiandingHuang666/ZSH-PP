import cv2
import os
import numpy as np


def addGuassNoisy(img_src, img_tar):
    img = cv2.imread(img_src)
    # 高斯分布的均值
    mean = 0
    # 高斯分布的标准差
    sigma = 36
    #生成高斯分布噪声
    gauss = np.random.normal(mean, sigma, \
                             (img.shape[0], img.shape[1], img.shape[2]))
    #给图片添加高斯噪声
    noisy_img = img + gauss
    #设置图片添加高斯噪声之后的像素值的范围
    noisy_img = np.clip(noisy_img,a_min=0,a_max=255)
    cv2.imwrite(img_tar, noisy_img)

def addSaltPepperNoise(img_src, img_tar):
    img = cv2.imread(img_src)
    # 椒盐噪声比例
    s_vs_p = 0.5
    # 噪声像素数目
    amount = 0.02
    noisy_img = np.copy(img)
    # salt噪声
    num_salt = np.ceil(amount * img.size * s_vs_p)
    # 噪声坐标位
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
    noisy_img[coords[0],coords[1],:] = [255,255,255]
    # pepper噪声
    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    # 噪声坐标
    coords = [np.random.randint(0,i - 1, int(num_pepper)) for i in img.shape]
    noisy_img[coords[0],coords[1],:] = [0,0,0]
    cv2.imwrite(img_tar, noisy_img)

def addPoissonNoise(img_src, img_tar):
    img = cv2.imread(img_src)

    # 生成高斯噪声
    noise = np.random.normal(0.0, 1.0, img.shape)

    # 将高斯噪声转换为泊松噪声，并添加到灰度图像中
    noisy_image = np.exp(np.log(1.0 + noise * 0.1)) * img
    noisy_image = np.uint8(np.clip(noisy_image, 0, 255))

    cv2.imwrite(img_tar, noisy_image)



if __name__ == '__main__':
    root = 'images'
    imgs = os.listdir(os.path.join(root, 'Origin'))

    for i in imgs:
        src = os.path.join(root, 'Origin', i)
        gauss = os.path.join(root, 'Gaussian', f"Gaussian_{i}")
        SP = os.path.join(root, 'SaltPepper', f"SaltPepper_{i}")
        P = os.path.join(root, 'Poisson', f"Poisson_{i}")
        addGuassNoisy(src, gauss)
        addSaltPepperNoise(src, SP)
        addPoissonNoise(src, P)

