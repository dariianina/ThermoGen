import cv2
import numpy as np


class ThermalGrayAligner:


    @staticmethod
    def flexible_crop(img, zoom_factor=1.0, shift_x=0, shift_y=0, aspect_ratio=(16, 9)):
        """
        Crops the image by zooming in, with optional shift from the center, and keeps a fixed aspect ratio.
        Args:
            img (np.ndarray): Input image.
            zoom_factor (float): How much to zoom in (>1 means zoom in).
            shift_x (int): Horizontal shift from center (pixels, +right, -left).
            shift_y (int): Vertical shift from center (pixels, +down, -up).
            aspect_ratio (tuple): Desired aspect ratio as (width, height).
        Returns:
            Cropped image (original scale, not resized).
        """
        h, w = img.shape[:2]
        aspect_w, aspect_h = aspect_ratio
        target_ratio = aspect_w / aspect_h

        # Compute crop size based on zoom and aspect ratio
        crop_w = int(w / zoom_factor)
        crop_h = int(crop_w / target_ratio)

        # If crop_h is too big for the image, adjust crop_w and crop_h
        if crop_h > h:
            crop_h = int(h / zoom_factor)
            crop_w = int(crop_h * target_ratio)

        center_x, center_y = w // 2, h // 2
        center_x += shift_x
        center_y += shift_y

        x1 = max(center_x - crop_w // 2, 0)
        y1 = max(center_y - crop_h // 2, 0)
        x2 = min(x1 + crop_w, w)
        y2 = min(y1 + crop_h, h)

        cropped = img[y1:y2, x1:x2]
        return cropped

    @staticmethod
    def canny_edges(img, low_thresh=50, high_thresh=150):
        """Compute Canny edge map."""
        return cv2.Canny(img, low_thresh, high_thresh)

    @staticmethod
    def equalize(img):
        """Apply histogram equalization."""
        return cv2.equalizeHist(img)

    @staticmethod
    def gradient_magnitude(img, ksize=5):
        """Compute gradient magnitude image."""
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
        grad_mag = cv2.magnitude(sobelx, sobely)
        grad_mag = np.uint8(255 * grad_mag / np.max(grad_mag)) if np.max(grad_mag) > 0 else np.zeros_like(img)
        return grad_mag

    @staticmethod
    def align_thermal_to_gray(gray_img, thermal_img, use_edges=True):
        """
        Aligns the thermal image to the grayscale image using SIFT and homography.

        Args:
            gray_img (np.ndarray): Grayscale image (reference).
            thermal_img (np.ndarray): Thermal image to be aligned.

        Returns:
            aligned_thermal (np.ndarray): Warped thermal image aligned to gray_img.
            gray_img (np.ndarray): The original grayscale image.
        """

        # Optionally use edge maps for feature extraction
        if use_edges:
            proc_gray = ThermalGrayAligner.canny_edges(gray_img, 10, 40)
            proc_thermal = ThermalGrayAligner.canny_edges(thermal_img, 50, 150)
        else:
            proc_gray = gray_img
            proc_thermal = thermal_img

        # 1. Detect SIFT features and compute descriptors
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(proc_gray, None)
        kp2, des2 = sift.detectAndCompute(proc_thermal, None)

        if des1 is None or des2 is None:
                raise ValueError("No descriptors found in one or both images.")

        # 2. Match features using FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # 3. Lowe's ratio test to filter good matches
        good = []
        for m, n in matches:
            if m.distance < 0.9 * n.distance:
                good.append(m)

        # 4. Estimate transformation (homography) if enough matches
        if len(good) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            h, w = gray_img.shape
            aligned_thermal = cv2.warpPerspective(thermal_img, H, (w, h))
            return gray_img, aligned_thermal
        else:
            raise ValueError("Not enough matches found to compute homography.")