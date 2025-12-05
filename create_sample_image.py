import cv2
import numpy as np

def create_uneven_illumination_image(filename='sample_image.jpg'):
    # Create a blank image
    width, height = 800, 600
    image = np.zeros((height, width), dtype=np.uint8)

    # Create a gradient background (simulating uneven lighting)
    for y in range(height):
        for x in range(width):
            # Gradient from top-left (dark) to bottom-right (bright)
            image[y, x] = (x + y) / (width + height) * 200

    # Add some text to the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Computer Vision', (50, 100), font, 2, (50), 3, cv2.LINE_AA) # Darker text on dark background
    cv2.putText(image, 'Thresholding Demo', (100, 250), font, 2, (255), 3, cv2.LINE_AA) # White text
    cv2.putText(image, 'Uneven Lighting', (300, 400), font, 2, (100), 3, cv2.LINE_AA) # Mid-gray text
    
    # Add some shapes
    cv2.circle(image, (600, 100), 50, (255), -1)
    cv2.rectangle(image, (50, 450), (250, 550), (80), -1)

    # Add some noise
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)

    # Save the image
    cv2.imwrite(filename, image)
    print(f"Created {filename}")

if __name__ == "__main__":
    create_uneven_illumination_image()
