import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_info():
    cv2.imwrite("gray_image.jpg", gray_image)
    cv2.imshow("Gray Image", gray_image)

    height, width = gray_image.shape
    min_intensity = gray_image.min()
    max_intensity = gray_image.max()
    min_coords = np.unravel_index(np.argmin(gray_image), gray_image.shape)
    max_coords = np.unravel_index(np.argmax(gray_image), gray_image.shape)

    print(f"Width {width}, Height {height}")
    print(f"Minimum Intensity: {min_intensity} at {min_coords[0]}, {min_coords[1]} (row, col)")
    print(f"Maximum Intensity: {max_intensity} at {max_coords[0]}, {max_coords[1]} (row, col)")

    original_size = image.nbytes
    gray_size = gray_image.nbytes
    print(f"\nSize of original image: {original_size} bytes")
    print(f"Size of grayscale image: {gray_size} bytes")

def downsample(image, factor):
    # Take every nth pixel
    return image[::factor, ::factor]

def graph_downsampling():
    original_size = gray_image.nbytes
    factors = range(1, 21)  # Downsampling factors from 1 to 20
    sizes = []

    for factor in factors:
        downsampled_image = downsample(gray_image, factor)
        sizes.append(downsampled_image.nbytes)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(factors, sizes, marker='o', label="Image Size vs. Downsampling Factor")
    plt.axhline(y=original_size, color='r', linestyle='--', label="Original Size")
    plt.xlabel("Downsampling Factor")
    plt.ylabel("Image Size (Bytes)")
    plt.title("Effect of Downsampling Factor on Image Size")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    image = cv2.imread("YosemiteFalls.jpg")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    display_info()

    factor = 15
    downsampled_image = downsample(gray_image, factor)  # Halve the size
    gray_size = gray_image.nbytes
    resized_image_size = downsampled_image.nbytes
    print(f"Size of resized (factor={factor}) image: {resized_image_size} bytes")
    cv2.imshow("Downsampled", downsampled_image)
    cv2.imwrite(f"downsampled_factor_{factor}.jpg", downsampled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    graph_downsampling()