import cv2
import dlib

#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.1.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.2.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.3.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.4.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.5.jpg')
#imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.6.jpg')
imagem = cv2.imread('/content/drive/My Drive/fotos/grupo.7.jpg')

detectorHog = dlib.get_frontal_face_detector()
facesDetectadasHog, pontuacao, idx = detectorHog.run(imagem, 2)

detectorCNN = dlib.cnn_face_detection_model_v1('/content/drive/My Drive/imagem_e_recurso/recursos/mmod_human_face_detector.dat')
facesDetectadasCNN = detectorCNN(imagem, 2)

for i, d in enumerate(facesDetectadasHog):
  print(pontuacao[i]) #significa que vai pegar a pontuaçao da face 0, 1, 2 e etc
print()
for face in facesDetectadasCNN:
  print(face.confidence)