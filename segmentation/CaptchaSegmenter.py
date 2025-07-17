import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

class CaptchaSegmenter:
    def __init__(self, image_path, output_dir='extracted_characters', mistmach_dir='mismatched_images'):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f'{image_path} could not be loaded.')
        # self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.output_dir = output_dir
        self.mismatch_dir = mistmach_dir
        self.captcha_text = None
        self.extracted_characters = []

    def get_captcha_text_from_file_name(self):
        root, _ = os.path.splitext(self.image_path)
        self.captcha_text = os.path.basename(root).split('-')[0]

    def segment_characters(self):
        # Convert the image to grayscale
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization to improve contrast
        hist_adjusted = cv2.equalizeHist(grayscale)

        # Apply adaptive thresholding to create a binary image
        adaptive_thresh = cv2.adaptiveThreshold(hist_adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
        
        # Apply median filtering to reduce noise
        median_filtered = cv2.medianBlur(adaptive_thresh, 3)

        # Apply morphological opening to remove small noise and enhance character separation
        morphology_kernel = np.ones((3, 3), np.uint8)
        opened_img = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, morphology_kernel, iterations=1)

        # Find contours in the processed image
        opened_img_contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours are found, exit process
        if not opened_img_contours:
            # print('No contours found.')
            return
        
        # Store all points from contours to find the bounding box
        all_points = np.vstack(opened_img_contours).squeeze()
        all_x, all_y = all_points[:, 0], all_points[:, 1]

        # Calculate the bounding box area of the CAPTCHA text
        pad = 0
        height, width = opened_img.shape
        x_min, x_max = max(0, min(all_x) - pad), min(width, max(all_x) + pad)
        y_min, y_max = max(0, min(all_y) - pad), min(height, max(all_y) + pad)

        x_max_crop = x_max

        # Crop the CAPTCHA area
        captcha_area = median_filtered[y_min:y_max, x_min:x_max]
        captcha_area_colored = self.image[y_min:y_max, x_min:x_max]

        # Find contours in the cropped CAPTCHA area
        captcha_contours, _ = cv2.findContours(captcha_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours based on their x-coordinate (left-to-right)
        bounding_boxes_coords = sorted([cv2.boundingRect(contour) for contour in captcha_contours], key=lambda x: x[0])

        grouped_boxes = []
        current_group = bounding_boxes_coords[0]

        # Group overlapping bounding boxes
        for box in bounding_boxes_coords[1:]:
            x, y, w, h = box
            prev_x, prev_y, prev_w, prev_h = current_group
            overlap_width = min(prev_x + prev_w, x + w) - max(prev_x, x)

            if overlap_width > 0 and (overlap_width >= 0.1 * w or overlap_width >= 0.1 * prev_w):
                # Merge overlapping boxes
                current_group = (
                    min(prev_x, x),
                    min(prev_y, y),
                    max(prev_x + prev_w, x + w) - min(prev_x, x),
                    max(prev_y + prev_h, y + h) - min(prev_y, y)
                )

            else:
                # No overlap, save the current group and start a new one
                grouped_boxes.append(current_group)
                current_group = box

        grouped_boxes.append(current_group)

        # Sort grouped boxes by their x-coordinate (left-to-right)
        grouped_boxes.sort(key=lambda x: x[0])

        characters = []

        for i in range(len(grouped_boxes)):
            x_box, y_box, w_box, h_box = grouped_boxes[i]
            segment = captcha_area_colored[:, x_box:min(x_box + w_box, x_max_crop)]

            # Skip empty segments
            if segment is None or segment.size == 0 or segment.shape[1] == 0:
                continue

            # Convert segment to HSV color space and split into channels
            hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv_segment)

            # Create a mask for the segment
            segment_mask = captcha_area[:, x_box:x_box + w_box]
            segment_mask_resized = cv2.resize(segment_mask, (hue.shape[1], hue.shape[0])) 
            segment_mask_binary = cv2.threshold(segment_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]

            # Calculate histogram of hue values and find peaks to identify character segments
            hist = cv2.calcHist([hue], [0], segment_mask_binary, [180], [0, 180]).flatten()
            peak_threshold = 0.3 * max(hist)
            peaks = np.sort(np.where(hist > peak_threshold)[0])

            # Combine peaks that are close together to avoid redundant processing
            combined_peaks = [peaks[0]] if len(peaks) > 0 else []

            for peak in peaks[1:]:
                if abs(peak - combined_peaks[-1]) >= 3:
                    combined_peaks.append(peak)

            peaks = np.array(combined_peaks)

            # segment_contours = []

            for peak in peaks:
                if peak == 0:
                    # Count black and non-black pixels with hue 0
                    black_pixels_with_hue_0 = np.sum((hue == 0) & (value < 50))
                    non_black_pixels_with_hue_0 = np.sum((hue == 0) & (value >= 50))
                    total_pixels = hue.size
                    non_black_non_white_pixels_with_hue_0 = np.sum((hue == 0) & (value >= 50) & (value < 250))

                    # Skip processing if more than 50% of total pixels are black and there are no non-black pixels
                    if black_pixels_with_hue_0 / total_pixels < 0.3:
                        if non_black_non_white_pixels_with_hue_0:
                            non_black_mask = (value >= 50).astype(np.uint8) * 255
                            hsv_segment[non_black_mask == 0] = [0, 0, 255]  # White in HSV
                        
                        else:
                            continue            

                # Create a mask for the peak hue value 
                lower_bound = np.array([max(peak - 2, 0)], dtype=np.uint8)
                upper_bound = np.array([min(peak + 2, 179)], dtype=np.uint8)
                peak_mask = cv2.inRange(hue, lower_bound, upper_bound)

                # Filter the segment using the peak mask
                filtered_segment = cv2.bitwise_and(hsv_segment, hsv_segment, mask=peak_mask)

                # Convert the filtered segment to grayscale, apply histogram equalization and thresholding
                gray = cv2.cvtColor(filtered_segment, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                _, bin = cv2.threshold(eq, 1, 255, cv2.THRESH_BINARY)

                # Find contours in the binary image
                contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Get x, y coordinates of characters
                all_x, all_y = [], []
                for contour in contours:
                    for point in contour:
                        all_x.append(point[0][0])
                        all_y.append(point[0][1])

                # Calculate the bounding box of the character
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)

                # Skip if the bounding box is empty or out of bounds
                if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
                    continue
                
                padding = 2
                padded_bin = cv2.copyMakeBorder(
                    bin[:, x_min:x_max],  # The sliced binary image
                    padding,              # Padding for the top
                    padding,              # Padding for the bottom
                    padding,              # Padding for the left
                    padding,              # Padding for the right
                    cv2.BORDER_CONSTANT,  # Border type
                    value=0               # Padding value
                )
        
                characters.append((padded_bin, (x_box + x_min, y_box + y_min, w_box, h_box)))

        # Sort characters based on their x-coordinate (left-to-right)
        characters.sort(key=lambda letter: letter[1][0])

        # Store the segmented characters
        self.extracted_characters = [character[0] for character in characters]
        
        # gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # char_images = []
        # for contour in contours:
        #     x, y, w, h = cv2.boundingRect(contour)
        #     if w > 5 and h > 10:  # Filter out noise
        #         char_image = self.image[y:y+h, x:x+w]
        #         char_images.append(char_image)

        # return char_images

    def resize_image(self, image, width, height):
        """
        Resize the image to the specified width and height.
        """
        dim = None
        (h, w) = image.shape[:2]

        if width is None and height is None:
            return image
        
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)

        elif height is None:
            r = width / float(w)
            dim = (width, int(h * r))

        else:
            dim = (width, height)

        return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def show_segmented_characters(self):
        char_images = self.segment_characters()
        for i, char_img in enumerate(char_images):
            plt.subplot(1, len(char_images), i + 1)
            plt.imshow(cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()
    
    def save_segmented_characters(self):
        skipped = False
        char_index = 0
        
        # char_images = self.segment_characters()
        # for i, char_img in enumerate(char_images):
        #     output_path = os.path.join(self.output_dir, f'char_{i}.png')
        #     cv2.imwrite(output_path, char_img)
        #     print(f"Saved character image to {output_path}")

        if len(self.extracted_characters) != len(self.captcha_text):
            # print('Number of extracted characters does not match CAPTCHA text length.')

            if not os.path.exists(self.mismatch_dir):
                os.makedirs(self.mismatch_dir)

            image_name = os.path.basename(self.image_path)
            cv2.imwrite(os.path.join(self.mismatch_dir, image_name), self.image)
            
            skipped = True
            return skipped
        
        # for i, char_img in enumerate(self.extracted_characters):
        #     output_path = os.path.join(self.output_dir, f'{self.captcha_text}_{i}.png')
        #     cv2.imwrite(output_path, char_img)
        #     print(f"Saved character image to {output_path}")

        for i in range(len(self.extracted_characters)):
            char_img = self.resize_image(self.extracted_characters[i], 40, 40)

            output_path = os.path.join(self.output_dir, self.captcha_text[char_index])
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            image_num = len(os.listdir(output_path)) + 1
            cv2.imwrite(os.path.join(output_path, f'{image_num}.png'), char_img)
            
            char_index += 1

            return skipped

if __name__ == "__main__":
    # Example usage
    skipped_images = 0

    for image_path in os.listdir('main'):
        if image_path.endswith('.png'):
            full_image_path = os.path.join('main', image_path)
            segmenter = CaptchaSegmenter(full_image_path)
            segmenter.get_captcha_text_from_file_name()
            segmenter.segment_characters()
            skipped = segmenter.save_segmented_characters()

            if skipped:
                skipped_images += 1
            # segmenter.show_segmented_characters()

    # segmenter = CaptchaSegmenter('path_to_captcha_image.png')
    # segmenter.get_captcha_text_from_file_name()
    # segmenter.segment_characters()
    # segmenter.save_segmented_characters()
    # segmenter.show_segmented_characters()

    print(f'{skipped_images} images skipped')
