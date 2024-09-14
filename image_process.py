import cv2
import matplotlib.pyplot as plt
import os



image_url = '/path/to/your/folder/tyres_picture/defective/Defective (555).jpg' #<--CHANGE ME
good_tyre_image_url = '/path/to/your/folder/tyres_picture/good/good (66).jpg' #<--CHANGE ME



#input
base_path_input = '/path/to/your/folder/tyres_picture/good/' #<--CHANGE ME
filename_pattern_input = 'good ({}).jpg'


#output
base_path_output = '//path/to/your/folder/tyres_picture_plus_contours/good_c/' #<--CHANGE ME
filename_pattern_output = 'good_c ({}).jpg'






image = cv2.imread(image_url) 

# Convert BGR to RGB to display with matplotlib (since OpenCV uses BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



#GREYSCALE CONVERSION
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



#CONTRAST  ENHANCEMENT
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_image = clahe.apply(gray_image)


#NOISE REDUCTION- TOGLIERE
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)

# Apply Bilateral Filter for better noise reduction while preserving edges
#blurred_image = cv2.bilateralFilter(enhanced_image, d=9, sigmaColor=75, sigmaSpace=75)


# Apply Median Filter for salt-and-pepper noise reduction
#blurred_image = cv2.medianBlur(enhanced_image, ksize=5)

#EDGE DETECTION
edges = cv2.Canny(blurred_image, 100, 200)



#MORPHOLOGICAL OPERTIONS
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morphed_image = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)


#TRESHOLDING
ret, thresholded_image = cv2.threshold(morphed_image, 175, 255, cv2.THRESH_BINARY)


#CONTOURS FINDING AND DRAWING
contours, hierarchy = cv2.findContours(image=thresholded_image, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

original_image_with_contours_on = image.copy()
cv2.drawContours(image=original_image_with_contours_on, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)




# Set up the figure and subplots
fig, axs = plt.subplots(2, 4, figsize=(10, 5))

# Display the original image
axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')  # Hide the axis

# Display the grayscale image
axs[0, 1].imshow(gray_image, cmap='gray')
axs[0, 1].set_title('Grayscale Image')
axs[0, 1].axis('off')  # Hide the axis

# Display the contrast enhanced image
axs[0, 2].imshow(enhanced_image, cmap='gray')
axs[0, 2].set_title('Contrast enhanced Image')
axs[0, 2].axis('off')  # Hide the axis


# Display the NOISE REDUCTION image
axs[0, 3].imshow(blurred_image, cmap='gray')
axs[0, 3].set_title('NOISE REDUCTION Image')
axs[0, 3].axis('off')  # Hide the axis

# Display the EDGE DETECTION image
axs[1, 0].imshow(edges, cmap='gray')
axs[1, 0].set_title('EDGE DETECTION Image')
axs[1, 0].axis('off')  # Hide the axis


# Display the #MORPHOLOGICAL OPERTIONS image
axs[1, 1].imshow(edges, cmap='gray')
axs[1, 1].set_title('MORPHOLOGICAL OPERTIONS Image')
axs[1, 1].axis('off')  # Hide the axis


# Display the TRESHOLDING image
axs[1, 2].imshow(thresholded_image)
axs[1, 2].set_title('TRESHOLDING Image')
axs[1, 2].axis('off')  # Hide the axis


# Display the COUNTOURS image
axs[1, 3].imshow(original_image_with_contours_on)
axs[1, 3].set_title('COUNTOURS Image')
axs[1, 3].axis('off')  # Hide the axis




# Show the plots
plt.tight_layout()
plt.show()


