import numpy as np
import cv2 as cv

car_cascade = cv.CascadeClassifier('cars.xml')

img = cv.imread('par.png')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.02, 2)
for (x, y, w, h) in cars:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv.imshow('Test', img)
cv.waitKey(0)
cv.destroyAllWindows()
