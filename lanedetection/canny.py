import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import os


image = mpimg.imread('test_images/road.jpg')

print('This image is:', type(image), 'with dimensions:', image.shape)

def grayscale(img):

    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):

    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)  #Returns an array of zeros with the same shape and type as a given array
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:   #returns the length of input, in this case img.shape
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


"""Define The Lines"""

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):

    imshape = img.shape
    left_x1 = []
    left_x2 = []
    right_x1 = []
    right_x2 = []
    y_min = img.shape
    y_max = int(img.shape[0]*0.611)
    for line in lines:
        for x1,y1,x2,y2 in line:
             if((y2-y1)/(x2-x1)) < 0:
                 mc=np.polyfit([x1, x2], [y1, y2], 1)
                 left_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                 left_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#       cv2.lines(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
             elif ((y2-y1)/(x2-x1)) > 0:
                 mc=np.polyfit([x1, x2], [y1, y2], 1)
                 right_x1.append(np.int(np.float((y_min - mc[1]))/np.float(mc[0])))
                 right_x2.append(np.int(np.float((y_max - mc[1]))/np.float(mc[0])))
#       cv2.lines(img, (xone, imshape[0]), (xtwo, 330), color, thickness)
    l_avg_x1 = np.int(np.nanmean(left_x1))
    l_avg_x2 = np.int(np.nanmean(left_x2))
    r_avg_x1 = np.int(npnanmean(right_x1))
    r_avg_x2 - np.int(np.nanmean(right_x2))
#     print([l_avg_x1, l_avg_x2, r_avg_x2, r_avg_x2])
    cv2.line(img, (l_avg_x1, y_min), (l_avg_x2, y_max), color, thickness)
    cv2.line(img, (r_avg_x1, y_min), (r_avg_x2, y_max), color, thickness)


def draw_avg_lines(img, lines, color=[255, 0, 0], thickness=10, info=dict()):
    """
    This function draws two lines(left and right) by averaging line segments

    Args:
        img (np.array): image input where lines will be drawn
        lines (list): [line1, line2, ...] 
                      where line1 = (x1,y1, x2,y2)
                      which means a line segment 
                      from pointA(x1, y1) to pointB(x2, y2)
        color (list): [r, g, b] rgb value of color in list
        thickness (int): thickness value
        info (dict): store information to carry on
                     so if no line detected, we can keep assuming from the previous observation

    """

    left_slope_data = info.get('left', [])[-50:]
    right_slope_data = info.get('right', [])[-50:]

    left_slope, left_bias = info.get('left_median', (0, 0))
    right_slope, right_bias = info.get('right_median', (0, 0))

    def get_slope(x1, y1, x2, y2):
        """Returns the slope and bias given two points
        y = slope * x + bias        
        """
        slope = (y2 - y1) / (x2 - x1+1e-7)
        bias = y1 - slope * x1
        return slope, bias

    def compute_avg_slope(slope_data, old_avg_slope, old_avg_bias, discount_rate = 0.5):

        def get_running_avg(old, new, discount_rate):
            if old == 0:
                return new
            return (1 - discount_rate) * old + discount_rate * new

        if len(slope_data) > 0:
            slope, bias = np.mean(slope_data, 0)
            new_slope = get_running_avg(old_avg_slope, slope, discount_rate)
            new_bias = get_running_avg(old_avg_bias, bias, discount_rate)

            return new_slope, new_bias
        return 0, 0

    def draw_one_line(img, x, slope, bias):
        point_a = (x, int(x * slope + bias))
        
        point_b_y = img.shape[0] // 100 * 68
        point_b_x = int((point_b_y - bias) / (slope + 1e-7))
        
        point_b = (point_b_x, point_b_y)
        
        cv2.line(img, point_a, point_b, color, thickness)
        
        return point_a, point_b

    # if lines is not None
    # for each line in lines
    # get slope and bias from two given points in line
    # if slope is < 0
    # save to the left line
    # else
    # save to the right line
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope, bias = get_slope(x1, y1, x2, y2)
                if slope < 0:
                    left_slope_data.append((slope, bias))
                else:
                    right_slope_data.append((slope, bias))

    # discount rate is used to compute an average on runtime
    left_check = False
    discount_rate = 0.9
    left_slope, left_bias = compute_avg_slope(left_slope_data, left_slope, left_bias, discount_rate)
    if left_slope != 0:
        point_a, point_b = draw_one_line(img, 0, left_slope, left_bias)
        left_check = True
    
    right_check = False
    right_slope, right_bias = compute_avg_slope(right_slope_data, right_slope, right_bias, discount_rate)
    if right_slope != 0:
        point_c, point_d = draw_one_line(img, img.shape[1], right_slope, right_bias)
        right_check = True
    
    poly_buffer = np.zeros_like(img)
    
    if left_check and right_check:
        cv2.fillPoly(poly_buffer, np.array([[point_a, point_c, point_d, point_b]], dtype=np.int32), [0, 255, 0])
        img = cv2.addWeighted(poly_buffer, 0.1, img, 1.0, 0.)
    
    info['left'] = left_slope_data
    info['left_median'] = (left_slope, left_bias)
    info['right'] = right_slope_data
    info['right_median'] = (right_slope, right_bias)
    
    return img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, thickness=10, info=dict()):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    
    line_img = draw_avg_lines(line_img, lines, thickness=thickness, info=info)

    return line_img

def weighted_img(img, initial_img, theta=0.8, beta=1., lamda=0.):

    return cv2.addWeighted(initial_img, theta, img, beta, lamda)

os.listdir("test_images/")

imgs = os.listdir("test_images/")
imgpath = os.path.join("test_images", imgs[5])
image1 = (mpimg.imread(imgpath)*255).astype('uint8')
plt.imshow(image1)

image1.shape


def process_image(img):
    img_test = grayscale(img)
    img_test = gaussian_blur(img_test, 7)
    img_test = canny(img_test, 50, 150)
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]),(imshape[1]*.45, imshape[0]*0.6), (imshape[1]*.55, imshape[0]*0.6), (imshape[1],imshape[0])]], dtype=np.int32)
    img_test = region_of_interest(img_test, vertices)
    rho = 2
    theta = np.pi/180
    threshold = 55
    min_line_len = 40
    max_line_gap = 100
    line_image = np.copy(img)*0
    img_test = hough_lines(img_test, rho, theta, threshold, min_line_len, max_line_gap)
    img_test = weighted_img(img_test, img)
    result = img_test
    return result

plt.imshow(process_image(image1))
plt.show()

