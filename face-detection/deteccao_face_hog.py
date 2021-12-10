from google.colab.patches import cv2_imshow #so utiliza essa funçao pro colab, pq ele n tem suporte a imshow
import cv2
import dlib

img = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg', cv2.IMREAD_UNCHANGED)
detector = dlib.get_frontal_face_detector()
facesDetectadas = detector(img)
print(facesDetectadas)
print('Faces Detectadas: ',len(facesDetectadas))
for face in facesDetectadas:
  #print(face)
  #print(face.left())
  #print(face.top())
  #print(face.right())
 # print(face.bottom())
  e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
  cv2.rectangle(img, (e, t), (d, b), (0, 255, 255), 2)
cv2_imshow(img) #mostrando a imagem