import cv2
import numpy as np

# PencilSketch filter class
class PencilSketch:
    def __init__(self):
        pass

    def render(self, img_rgb):
        # Convert to grayscale
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian Blur
        img_blur = cv2.GaussianBlur(img_gray, (21, 21), 0)
        # Blend the grayscale and blurred images
        img_blend = cv2.divide(img_gray, img_blur, scale=256)
        # Convert back to BGR format
        return cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)

# CoolingFilter class
class CoolingFilter:
    def __init__(self):
        # Lookup tables for adjusting color channels
        self.decr_ch_lut = np.arange(256, dtype=np.uint8)
        self.incr_ch_lut = np.arange(256, dtype=np.uint8)
        for i in range(256):
            # Decrease the red channel
            self.decr_ch_lut[i] = max(0, i - 50)
            # Increase the blue channel
            self.incr_ch_lut[i] = min(255, i + 50)

    def render(self, img_rgb):
        # Split the image into RGB channels
        c_r, c_g, c_b = cv2.split(img_rgb)
        # Apply the lookup tables to the red and blue channels
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        # Merge the channels back
        img_rgb = cv2.merge((c_r, c_g, c_b))
        return img_rgb

# Cartoonizer filter class
class Cartoonizer:
    def __init__(self):
        pass

    def render(self, img_rgb):
        # Downsample the image using Gaussian pyramid
        numDownSamples = 2
        numBilateralFilters = 7
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)
        # Apply bilateral filter multiple times
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)
        # Convert the image to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        # Detect edges using adaptive thresholding
        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 9, 2)
        # Convert the edges to RGB format
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        # Resize the edge image to match the color image
        img_edge_resized = cv2.resize(img_edge, (img_color.shape[1], img_color.shape[0]))
        # Combine the color and edge images using bitwise AND
        return cv2.bitwise_and(img_color, img_edge_resized)

# Sepia filter class
class SepiaFilter:
    def __init__(self):
        pass

    def render(self, img_rgb):
        # Sepia kernel to apply the effect
        kernel = np.array([[0.393, 0.769, 0.189],
                           [0.349, 0.686, 0.168],
                           [0.272, 0.534, 0.131]])
        # Apply the transformation
        img_sepia = cv2.transform(img_rgb, kernel)
        # Ensure the pixel values are in the range [0, 255]
        img_sepia = np.clip(img_sepia, 0, 255)
        return img_sepia.astype(np.uint8)

# Negative filter class
class NegativeFilter:
    def __init__(self):
        pass

    def render(self, img_rgb):
        # Invert the colors of the image
        img_negative = cv2.bitwise_not(img_rgb)
        return img_negative

# Gaussian Blur filter class
class GaussianBlurFilter:
    def __init__(self):
        pass

    def render(self, img_rgb):
        # Apply Gaussian blur to the image
        img_blur = cv2.GaussianBlur(img_rgb, (15, 15), 0)
        return img_blur

# Emboss filter class
class EmbossFilter:
    def __init__(self):
        # Emboss kernel for edge detection
        self.kernel = np.array([[-2, -1, 0],
                                [-1, 1, 1],
                                [0, 1, 2]])

    def render(self, img_rgb):
        # Apply the emboss effect using filter2D
        img_emboss = cv2.filter2D(img_rgb, -1, self.kernel)
        return img_emboss
