import pytesseract
from PIL import Image, ImageEnhance, ImageFilter

# Path to your Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load the image
image_path = 'trainingimages/5x5_3.jpg'
img = Image.open(image_path)

# Convert the image to grayscale
img = img.convert('L')

# Enhance image contrast (adjust as needed)
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(3.0)  # Increase contrast

# Apply thresholding to get a binary image
threshold = 200
img = img.point(lambda p: p > threshold and 255)

# Optional: Apply Gaussian blur to reduce noise
img = img.filter(ImageFilter.GaussianBlur(radius=2))

# Perform OCR
custom_config = r'--oem 3 --psm 7'  # Page segmentation mode 6
text = pytesseract.image_to_string(img,lang='eng',config='-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 6')

print(text)
