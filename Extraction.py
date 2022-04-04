import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def disp(image, name=None):
    """
    This is function disp for image visualization

    Parameters:
    image - input image
    name - image name
    """
    plt.figure(num=name)
    plt.axis('off')
    plt.imshow(image, cmap='gray')


def initialization(image):
    """
    This is function initialization for finger vein image ROI extraction initialization

    Parameters:
    image - input image

    Returns:
    upper - Start point for upper finger edge search
    lower - Start point for lower finger edge search
    baseline - Search starting point baseline
    grad - Horizontal gradient image of finger vein image
    """
    h, w = image.shape
    baseline = w // 2 - 1
    grad_up = cv2.Sobel(image[:h // 2, :], -1, 0, 1)
    grad_down = cv2.Sobel(np.flip(image[h // 2:, :]), -1, 0, 1)
    grad = np.zeros((h, w))
    grad[:h // 2, :] = grad_up
    grad[h // 2:, :] = np.flip(grad_down)
    upper = np.unravel_index(np.argmax(grad[:h // 2, baseline]), (h // 2))[0]
    lower = np.unravel_index(np.argmax(grad[h // 2:, baseline]), (h // 2))[0] + h // 2
    return upper, lower, baseline, grad


def get_edge(image, grad, upper, lower, baseline, rate):
    """
    This is function get_edge for detect the coordinates of the upper and lower finger edge points

    Parameters:
    upper - Start point for upper finger edge search
    lower - Start point for lower finger edge search
    baseline - Search starting point baseline
    grad - Horizontal gradient image of finger vein image
    rate - Original image and gradient image fusion scale rate

    Returns:
    index_ru - Set of coordinate points on the upper right edge
    index_lu - Set of coordinate points on the upper left edge
    index_rl - Set of coordinate points on the lower left edge
    index_ll - Set of coordinate points on the lower right edge
    """
    upper_N = upper;
    lower_N = lower
    index_ru = [[upper, baseline], [upper, baseline]]
    index_rl = [[lower, baseline], [lower, baseline]]
    index_lu = [[upper, baseline], [upper, baseline]]
    index_ll = [[lower, baseline], [lower, baseline]]
    for i in range(baseline + 1):
        pix = (1 - rate) * abs(image[upper - 1:upper + 2, i + 1 + baseline] - image[upper, i + baseline]) + rate * abs(
            (grad[upper - 1:upper + 2, i + 1 + baseline] - grad[upper, i + baseline]))
        pix = np.unravel_index(np.argmin(pix), (3))[0]
        temp_index = [upper - 1 + pix, i + baseline + 1]
        upper = upper - 1 + pix
        index_ru.append(temp_index)
        image[temp_index[0], temp_index[1]] = 255
        grad[temp_index[0], temp_index[1]] = 255

    upper = upper_N
    for i in range(baseline):
        pix = (1 - rate) * abs(image[upper - 1:upper + 2, baseline - i - 1] - image[upper, baseline - i]) + rate * abs(
            (grad[upper - 1:upper + 2, baseline - i - 1] - grad[upper, baseline - i]))
        pix = np.unravel_index(np.argmin(pix), (3))[0]
        temp_index = [upper - 1 + pix, baseline - i - 1]
        upper = upper - 1 + pix
        index_lu.append(temp_index)
        image[temp_index[0], temp_index[1]] = 255
        grad[temp_index[0], temp_index[1]] = 255

    for i in range(baseline + 1):
        pix = (1 - rate) * abs(image[lower - 1:lower + 2, i + 1 + baseline] - image[lower, i + baseline]) + rate * abs(
            (grad[lower - 1:lower + 2, i + 1 + baseline] - grad[lower, i + baseline]))
        pix = np.unravel_index(np.argmin(pix), (3))[0]
        temp_index = [lower - 1 + pix, i + baseline + 1]
        lower = lower - 1 + pix
        index_rl.append(temp_index)
        image[temp_index[0], temp_index[1]] = 255
        grad[temp_index[0], temp_index[1]] = 255

    lower = lower_N
    for j in range(baseline):
        pix = (1 - rate) * abs(
            image[lower - 1:lower + 2, baseline - j - 1] - image[lower, baseline - j]) + rate * abs(
            (grad[lower - 1:lower + 2, baseline - j - 1] - grad[lower, baseline - j]))
        pix = np.unravel_index(np.argmin(pix), (3))[0]
        temp_index = [lower - 1 + pix, baseline - j - 1]
        lower = lower - 1 + pix
        index_ll.append(temp_index)
        image[temp_index[0], temp_index[1]] = 255
        grad[temp_index[0], temp_index[1]] = 255
    # disp(image)
    # plt.show()
    return index_ru, index_lu, index_rl, index_ll


def get_up_down(index_u, index_l):
    """
    This is function get_up_down for get the minimum inner tangent coordinate

    Parameters:
    index_u - Set of coordinate points on the upper edge
    index_l - Set of coordinate points on the lower edge

    Returns:
    up - Upper split line horizontal coordinate
    down - Lower split line horizontal coordinate
    """
    index_up = np.array(index_u)
    index_down = np.array(index_l)
    index = np.argmin(index_down - index_up, axis=0)[0]
    up = index_u[index][0]
    down = index_l[index][0]
    return up, down


def get_ROI(image, shape):
    """
    This is function get_up_down for get finger vein image ROI

    Parameters:
    image - input image
    shape - Shape for size normalization
    rate - Original image and gradient image fusion scale rate

    Returns:
    ROI - finger vein image ROI
    """
    rate = 0.5
    image_c = image.copy()
    upper, lower, baseline, grad = initialization(image)
    index_ru, index_lu, index_rl, index_ll = get_edge(image, grad, upper, lower, baseline, rate)
    up, down = get_up_down(index_ru, index_rl)

    # Upper and lower split line visualization
    # image_c[up,:]=255
    # image_c[down,:]=255
    # disp(image_c)
    # plt.show()

    ROI = image_c[up:down, :]
    ROI = cv2.resize(ROI, dsize=(shape[1], shape[0]))
    return ROI


"""
A simple example of ROI extraction for images in the SDUMLA-FV dataset
Note that: 
1) The FV-USM dataset requires image selection pre-processing (finger horizontal placement)
2) Simple image size change can improve ROI extraction efficiency
"""
# rate=0.5
# image=cv2.imread('./sample_image/SDUMLA-FV.bmp',cv2.IMREAD_GRAYSCALE)
# ROI=get_ROI(image,[150,300],rate)
