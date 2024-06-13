# pip install -r requirements.txt
import cv2
import easyocr
import matplotlib.pyplot as plt

def Preproceesing(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # Remove vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    remove_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Combine the two masks to remove lines
    mask = remove_horizontal + remove_vertical
    cleaned_img = cv2.bitwise_and(thresh, thresh, mask=~mask)

    # Invert the image back to original
    cleaned_img = cv2.bitwise_not(cleaned_img)
    
    return cleaned_img



# Read the image
image_path = 'trainingimages/5x5_2.png'
img = cv2.imread(image_path)
cleaned_img = Preproceesing(img)

reader = easyocr.Reader(['en'])
original_text = reader.readtext(cleaned_img)

for t in original_text:
    print(t)

    bbox, text, score = t

    # bbox[0] = top left, bbox[2] = bottom right
    cv2.rectangle(cleaned_img, bbox[0], bbox[2], (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(cleaned_img, cv2.COLOR_BGR2RGB))
plt.show()
