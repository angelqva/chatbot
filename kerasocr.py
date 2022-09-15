
import matplotlib.pyplot as plt
import cv2
import keras_ocr

# keras-ocr will automatically download pretrained
# weights for the detector and recognizer.
img_texto = cv2.imread("kerasocr.jpg")
cv2.imshow("texto", img_texto)
cv2.waitKey(0)
pipeline = keras_ocr.pipeline.Pipeline()
prediction = pipeline.recognize([img_texto])
print(prediction)
keras_ocr.tools.drawAnnotations(plt.imread("kerasocr.jpg"), prediction[0])
plt.show()
