import cv2
import numpy as np
import matplotlib.pyplot as plt

def main():
    # --- Part 1: Load and Preprocess ---
    # Load the chosen image and convert it to grayscale
    image_path = 'sample_image.jpg'
    # Read as grayscale directly (0 flag) or read color and convert
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img_gray is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Save original for display (though it's same as gray since we read grayscale)
    # If we wanted color, we'd read it normally. Let's assume gray is our base.
    
    print("Image loaded successfully.")

    # --- Part 2A: Standard (Simple) Thresholding ---
    # Manually choose a global threshold value (e.g., 127)
    global_thresh_value = 127
    # Apply simple binary thresholding
    ret_simple, thresh_simple = cv2.threshold(img_gray, global_thresh_value, 255, cv2.THRESH_BINARY)
    
    print(f"Simple Thresholding done. Threshold: {global_thresh_value}")

    # --- Part 2B: Otsu's Binarization ---
    # Pass 0 as threshold value, use THRESH_OTSU flag
    # returns optimal threshold value in 'ret_otsu'
    ret_otsu, thresh_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Otsu's Thresholding done. Optimal Threshold Found: {ret_otsu}")

    # --- Part 2C: Adaptive Thresholding ---
    # 1. Adaptive Mean Thresholding
    # blockSize: Size of a pixel neighborhood that is used to calculate a threshold value. Must be odd (e.g., 11, 15).
    # C: Constant subtracted from the mean or weighted mean (e.g., 2, 5).
    block_size = 15
    c_value = 5
    
    thresh_adaptive_mean = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c_value
    )

    # 2. Adaptive Gaussian Thresholding
    thresh_adaptive_gauss = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c_value
    )
    
    print(f"Adaptive Thresholding done. BlockSize: {block_size}, C: {c_value}")

    # --- Part 2D: Band Thresholding (Manual Implementation) ---
    # Define lower and upper bounds
    lower_bound = 100
    upper_bound = 200
    
    # Method: Create mask for pixels > lower_bound AND pixels < upper_bound
    
    # Step 1: Threshold for lower bound (pixels > 100 become 255)
    _, mask_lower = cv2.threshold(img_gray, lower_bound, 255, cv2.THRESH_BINARY)
    
    # Step 2: Threshold for upper bound (pixels > 200 become 255). 
    # We want pixels < 200, so we can invert this result later OR use THRESH_BINARY_INV
    _, mask_upper_inv = cv2.threshold(img_gray, upper_bound, 255, cv2.THRESH_BINARY_INV)
    
    # Step 3: Combine using Bitwise AND
    # Logic: (Intensity > 100) AND (Intensity < 200)
    thresh_band = cv2.bitwise_and(mask_lower, mask_upper_inv)
    
    print(f"Band Thresholding done. Range: [{lower_bound}, {upper_bound}]")

    # --- Display Results ---
    # We will use matplotlib to display all results in a single window structure
    titles = [
        'Original Grayscale', 
        f'Simple Global (v={global_thresh_value})',
        f'Otsu (v={ret_otsu})',
        f'Adaptive Mean (Blk={block_size}, C={c_value})',
        f'Adaptive Gaussian (Blk={block_size}, C={c_value})',
        f'Band Thresholding ({lower_bound}-{upper_bound})'
    ]
    
    images = [
        img_gray, 
        thresh_simple, 
        thresh_otsu, 
        thresh_adaptive_mean, 
        thresh_adaptive_gauss, 
        thresh_band
    ]

    plt.figure(figsize=(12, 8))
    
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()

    # Optional: Save results to disk
    cv2.imwrite('result_simple.jpg', thresh_simple)
    cv2.imwrite('result_otsu.jpg', thresh_otsu)
    cv2.imwrite('result_adaptive_mean.jpg', thresh_adaptive_mean)
    cv2.imwrite('result_adaptive_gauss.jpg', thresh_adaptive_gauss)
    cv2.imwrite('result_band.jpg', thresh_band)
    print("Results saved to disk.")

if __name__ == "__main__":
    main()
