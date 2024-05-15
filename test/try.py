import cv2
import numpy as np
from matplotlib import pyplot as plt

def extract_roi(source_image, target_image):
    # Read the source image
    source = cv2.imread(source_image)
    
    # Read the target image
    target = cv2.imread(target_image)

    # Convert the images to grayscale
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    # Extract the ROI from the source face image
    roi = source_gray # Replace this with your code to extract the ROI
    
    return roi, target_gray

def calculate_normalized_cross_correlation(roi, target_gray):
    # Perform normalized cross-correlation
    cross_correlation = cv2.matchTemplate(target_gray, roi, cv2.TM_CCORR_NORMED)

    # Find the coordinates of the peak
    _, _, _, max_loc = cv2.minMaxLoc(cross_correlation)
    peak_coords = max_loc

    # Display normalized cross-correlation as a surface plot
    plt.imshow(cross_correlation, cmap='jet', interpolation='nearest')
    plt.colorbar()
    plt.show()

    return peak_coords
def find_offset(peak_coords):
    # Calculate the total offset between the images
    total_offset = (peak_coords[0], peak_coords[1])

    return total_offset

def check_face_extraction(target_image, roi, total_offset):
    # Figure out where the face exactly matches inside the target image
    # Replace this with your code to check if the face is extracted from the target image
    extracted_face_coords = (total_offset[0], total_offset[1], total_offset[0] + roi.shape[1], total_offset[1] + roi.shape[0])

    return extracted_face_coords

def pad_face_image(face_image, target_image, total_offset):
    # Pad the face image to the size of the target image using the offset
    target_height, target_width = target_image.shape[:2]
    padded_face_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    face_height, face_width = face_image.shape[:2]
    padded_face_image[total_offset[1]:total_offset[1]+face_height, total_offset[0]:total_offset[0]+face_width] = face_image

    return padded_face_image

def alpha_blend(face_image, target_image, total_offset):
    # Use alpha blending to show images together
    alpha = 0.5
    blended_image = cv2.addWeighted(face_image, alpha, target_image, 1-alpha, 0)

    return blended_image
# Specify the paths to the source and target images
source_image_path = 'E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/Dataset/FaceData/raw/TranVanDucSon/WIN_20240330_12_56_34_Pro.jpg'
target_image_path = 'E:/1. Bachkhoa/3. Year 3 Seminar 2/3.PBL5/PBL5/Dataset/FaceData/processed/TranVanDucSon/blur_WIN_20240330_12_56_33_Pro.jpg_4.jpg'

# Step 1: Extract the ROI from the source face image
roi, target_gray = extract_roi(source_image_path, target_image_path)

# Step 2: Calculate normalized cross-correlation and find peak coordinates
peak_coords = calculate_normalized_cross_correlation(roi, target_gray)

# Step 3: Find the total offset between the images
total_offset = find_offset(peak_coords)

# Step 4: Check if the face is extracted from the target image
extracted_face_coords = check_face_extraction(target_image_path, roi, total_offset)

# Step 5: Pad the face image to the size of the target image
padded_face_image = pad_face_image(roi, target_gray, total_offset)

# Step 6: Alpha blend the images
print(total_offset)
blended_image = alpha_blend(padded_face_image, target_image_path, total_offset)

# Display the blended image
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()