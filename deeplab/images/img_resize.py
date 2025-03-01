import cv2
img = cv2.imread("UI/up.jpeg")
resize_img = cv2.resize(img, (300, 300))
cv2.imwrite("UI/up380.jpeg", resize_img)