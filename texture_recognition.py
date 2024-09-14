import cv2
import os
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import matplotlib.pyplot as plt
from skimage.filters import gabor



contrast_tot = []
correlation_tot = []
energy_tot = []
homogeneity_tot = []





image_url = '/path/to/your/folder/tyres_picture/defective/Defective (555).jpg' #<--CHANGE ME
good_tyre_image_url = '/path/to/your/folder/tyres_picture/good/good (66).jpg' #<--CHANGE ME



base_path = '/path/to/your/folder/tyres_picture/defective/' #<--CHANGE ME
filename_pattern = 'Defective ({}).jpg'





image = cv2.imread(image_url, cv2.IMREAD_GRAYSCALE)


#Gabor filter


from skimage.filters import gabor

# Apply Gabor filter
frequency = 0.6
theta = np.pi / 4
filtered_image, _ = gabor(image, frequency=frequency, theta=theta)

# Display the filtered image
plt.imshow(filtered_image, cmap='gray')
plt.title('Gabor Filtered Image')
plt.axis('off')
plt.show()
