from google.colab.patches import cv2_imshow
import cv2
import dlib

img = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg', cv2.IMREAD_UNCHANGED)
detector = dlib.cnn_face_detection_model_v1('/content/drive/My Drive/imagem_e_recurso/recursos/mmod_human_face_detector.dat')
facesDetectadas = detector(img, 1) #o 1 serve pra aumentar a imagem em uma vez
print(facesDetectadas)
print('Faces Detectadas: ',len(facesDetectadas))
for face in facesDetectadas:
  e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
  print(c)
  cv2.rectangle(img, (e, t), (d, b), (0, 255, 255), 2)
cv2_imshow(img)