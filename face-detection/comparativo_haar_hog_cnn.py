from google.colab.patches import cv2_imshow
import cv2
import dlib

fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.1.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.2.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.3.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.4.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.5.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.6.jpg')
imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.7.jpg')

detectorHaar = cv2.CascadeClassifier('/content/drive/My Drive/imagem_e_recurso/recursos/haarcascade_frontalface_default.xml')
imagemCinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
facesDetectadasHaar = detectorHaar.detectMultiScale(imagemCinza, scaleFactor = 1.1, minSize = (10,10))

detectorHog = dlib.get_frontal_face_detector()
facesDetectadasHog, pontuacao, idx = detectorHog.run(imagem, 2)

detectorCNN = dlib.cnn_face_detection_model_v1('/content/drive/My Drive/imagem_e_recurso/recursos/mmod_human_face_detector.dat')
facesDetectadasCNN = detectorCNN(imagem, 2)

for (x, y, l, a) in facesDetectadasHaar:
  cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 255, 0), 2)
  cv2.putText(imagem, "Haar", (x, y - 5), fonte, 0.5, (0, 255, 0))

for face in facesDetectadasHog:
  e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
  cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
  cv2.putText(imagem, "Hog", (d, t), fonte, 0.5, (0, 255, 255)) #(d, t) vai ser o ponto onde a imagem vai aparecer, no caso direita topo

for face in facesDetectadasCNN:
  e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
  cv2.rectangle(imagem, (e, t), (d, b), (255, 255, 0), 2)
  cv2.putText(imagem, "CNN", (d, t), fonte, 0.5, (255, 255, 0))
cv2_imshow(imagem)