import cv2
import numpy as np
import imutils

image = cv2.imread('images/blobs.png', cv2.IMREAD_COLOR)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape
s = int(w / 8)
gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, s, 7.0)

# Downsize image (by factor 4) to speed up morphological operations
gray = cv2.resize(gray, dsize=(0, 0), fx=0.25, fy=0.25)
# cv2.imshow("Gray resized", gray)
# cv2.imwrite('Gray_resized.png', gray)

# Morphological opening: Get rid of the stuff at the top of the ellipse
gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
# cv2.imshow("Gray removed noise", gray)
# cv2.imwrite('Gray_removed_noise.png', gray)

# Resize image to original size
gray = cv2.resize(gray, dsize=(image.shape[1], image.shape[0]))

# Find contours
cnts, hier = cv2.findContours(gray, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Draw found contours in input image
image = cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)

for i, cont in enumerate(cnts):
    h = hier[0, i, :]
    print(h)
    if h[3] != -1:
        elps = cv2.fitEllipse(cnts[i])
    elif h[2] == -1:
        elps = cv2.fitEllipse(cnts[i])
    cv2.ellipse(image, elps, (0, 255, 0), 2)

# Downsize image
out_image = cv2.resize(image, dsize=(0, 0), fx=0.25, fy=0.25)
cv2.imshow("Output image", out_image)
cv2.imwrite('Output_image.png', out_image)
cv2.waitKey(0)
cv2.destroyAllWindows()