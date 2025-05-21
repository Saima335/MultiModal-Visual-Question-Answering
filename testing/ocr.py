from PIL import Image
import pytesseract
import os

def get_ocr_text(image_path):
    image = Image.open(image_path).convert("RGB")
    text = pytesseract.image_to_string(image)
    return text.strip()

# Example usage
if __name__ == "__main__":
    img_path = "sample.jpg"
    print("OCR Text:", get_ocr_text(img_path))
