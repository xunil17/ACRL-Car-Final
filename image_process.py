import numpy as np
import cv2


# Processes image output by environment
# Parameters:
# rgb_image: The RGB image output by the environment. A (nxmx3) array of floats
# flip: A boolean. True if we wish to flip/reflect images
# detect_edges: A boolean. True if we wish to apply canny edge detection to 
#                 the images
# Outputs:
# An (nxmx1) array of floats representing the processed image
def process_image(rgb_image, flip, detect_edges=False):
    if flip:
        rgb_image = flip_image(rgb_image)
    gray = grayscale_img(rgb_image)
    gray = unit_image(gray)
    result = gray
    return result

# Mirrors the provided image vertically while preserving some parameters
# Parameters:
# - image: A (nxmx1) array of floats representing a grayscale image. Assumes
#          the provided image is a snapshot from the CarRacing game.
# Outputs:
# - A (nxmx1) array of floars representing a mirrored version of 'image'.
#   Reflects the gameplay screen (road + car + grass) vertically. 
#   Reflects gyro and steering bars vertically
#   Preserves score, true speed, and ABS
def flip_image(image):
    steer_left = (image[88, 47] == [0, 255, 0]).all()
    gyro_left = (image[88, 71] == [255, 0, 0]).all()
    if steer_left:
        steer_length = sum(map(all, (image[88, :48] == [0, 255, 0])))
    else:
        steer_length = sum(map(all, (image[88, 48:] == [0, 255, 0])))
    if gyro_left:
        gyro_length = sum(map(all, (image[88, :72] == [255, 0, 0])))
    else:
        gyro_length = sum(map(all, (image[88, 72:] == [255, 0, 0])))

    image[84:, 28:] = 0

    if steer_left:
        image[86:91, 48:(48 + steer_length)] = [0, 255, 0]
    else:
        image[86:91, (48 - steer_length):48] = [0, 255, 0]
    if gyro_left:
        image[86:91, 72:(72 + gyro_length)] = [255, 0, 0]
    else:
        image[86:91, (72 - gyro_length):72] = [255, 0, 0]

    image[:84, :] = cv2.flip(image[:84, :], 1)
    return image

# Converts an RGB image to grayscale
# Parameters:
# - image: An RGB (nxmx3) array of floats
# Outputs:
# - A (nxmx1) array of floats in the range [0,255] representing a 
#   weighted average of the color channels of 'image'
def grayscale_img(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])


# Sets all pixel values to be between (0,1)
# Parameters:
# - image: A grayscale (nxmx1) or RGB (nxmx3) array of floats
# Outputs:
# - image rescaled so all pixels are between 0 and 1
def unit_image(image):
    image = np.array(image)
    return np.true_divide(image, 255)

