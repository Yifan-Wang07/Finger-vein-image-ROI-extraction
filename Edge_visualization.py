from Extraction import *


def edge_visualization(image, index_ru, index_lu, index_rl, index_ll):
    """
        This is function edge_visualization for  finger edge visualization

        Parameters:
        image - input image
        shape - Shape for size normalization
        rate - Original image and gradient image fusion scale rate

        Returns:
        edge_visual - Finger edge visualization

    """
    h, w = image.shape
    edge = np.zeros((h, w, 3))
    for i in index_ru:
        image[i[0] - 2:i[0] + 1, i[1]] = 0
    for i in index_lu:
        image[i[0] - 2:i[0] + 1, i[1]] = 0
    for i in index_rl:
        image[i[0] - 2:i[0] + 1, i[1]] = 0
    for i in index_ll:
        image[i[0] - 2:i[0] + 1, i[1]] = 0
    edge[:, :, 1] = image
    edge[:, :, 0] = image
    for i in index_ru:
        image[i[0] - 2:i[0] + 1, i[1]] = 255
    for i in index_lu:
        image[i[0] - 2:i[0] + 1, i[1]] = 255
    for i in index_rl:
        image[i[0] - 2:i[0] + 1, i[1]] = 255
    for i in index_ll:
        image[i[0] - 2:i[0] + 1, i[1]] = 255
    edge[:, :, 2] = image
    # disp(edge.astype(np.uint8))
    # plt.show()
    edge_viusal=edge.astype(np.uint8)
    return edge_viusal


"""
A simple example of edge visualization for images in publicly accessible dataset
Note that: 
1) The FV-USM dataset requires image selection pre-processing (finger horizontal placement)
2) Simple image size change can improve ROI extraction efficiency
"""

rate = 0.5
image = cv2.imread('./sample_image/SDUMLA-FV.bmp', cv2.IMREAD_GRAYSCALE)
# image = image.T
# image = image[:, :-100]  # processing for FV-USM
image_c = image.copy()
upper, lower, baseline, grad = initialization(image)
index_ru, index_lu, index_rl, index_ll = get_edge(image, grad, upper, lower, baseline, rate)
edge_visual=edge_visualization(image_c, index_ru, index_lu, index_rl, index_ll)
