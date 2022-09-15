import pytesseract
import cv2
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

img_texto = cv2.imread("ia.png")
imh, imw, _ = img_texto.shape
cajas_texto = pytesseract.image_to_boxes(
    img_texto, lang="spa")
texto_texto = pytesseract.image_to_string(
    img_texto, lang="spa")
for caja in cajas_texto.splitlines():
    b = caja.split(" ")
    x, y, w, h = int(b[1]), int(b[2]), int(b[3]), int(b[4])
    img_texto = cv2.rectangle(img_texto, (x, imh-y),
                              (w, imh-h), (0, 255, 0), 2)
print(texto_texto)
cv2.imshow("img_texto", img_texto)
cv2.waitKey(0)
