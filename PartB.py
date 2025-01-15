import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label

bacteria = cv2.imread("bacteria.bmp", cv2.IMREAD_GRAYSCALE)

intensities = []
binarized = []
for y in range(bacteria.shape[0]):
    col = []
    bin = []
    for x in range(bacteria.shape[1]):
        intensities.append(bacteria[y, x])
        bin.append(1 if bacteria[y, x] < 80 else 0)
    binarized.append(bin)

plt.hist(intensities, bins=range(0, 255, 5))
#plt.show()

# Threshold of 80
# TODO: compute total area of bacteria
print(f"bacteria takes up {len([i for i in intensities if i < 80])} pixels out of {len(intensities)} total pixels")

print(binarized)
labeled_array, num_features = label(bacteria)
print(bacteria, num_features)

