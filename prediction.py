import cv2
import label_image

# Opens entered image file (jpg, png, gif or bmp format)
# returns image, prediction of what food it is and that food's catogory
def load_image(image):
    text, classification = label_image.main(image)
    img = cv2.imread(image)
    return img, text, classification


img, text, classification = load_image('./test/chocolateCakeTest.jpg')

# Displays an image in a new window with text about food name and class
cv2.putText(img, text + " = " + classification, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('prediction', img)
cv2.waitKey(0) # Display the window infinitely until any keypress