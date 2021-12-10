from google.colab.patches import cv2_imshow
import cv2
import dlib
import numpy as np
def imprimePontos(imagem, pontosFaciais):
  for p in pontosFaciais.parts(): 
    cv2.circle(imagem, (p.x, p.y), 2, (0, 255, 0), 2)

def imprimeNumeros(imagem, pontosFaciais):
  for i, p in enumerate(pontosFaciais.parts()): #ao inves de imprimir os pontos, ele imprime numeros no lugar dos pontos 
    cv2.putText(imagem, str(i), (p.x, p.y), fonte, .55, (0, 0, 255), 2)

def imprimeLinhas(imagem, pontosFaciais):
  p68 = [[0, 16, False], #linha do queixo  (false = linha aberta) (true = linha fechada) aberta e quando nao tem ligaçao do inicio ao fim
         [17, 21, False], #sombrancelha direita
         [22, 26, False], #sombrancelha esquerda
         [27, 30, False], #ponte nasal (nariz)
         [30, 35, True], #nariz inferior
         [36, 41, True], #olho esquerdo
         [42, 47, True], #olho direito
         [48, 59, True], #labio externo
         [60, 67, True]] #labio interno
  for k in range(0, len(p68)):
    pontos = []
    for i in range(p68[k][0], p68[k][1] + 1):
      ponto = [pontosFaciais.part(i).x, pontosFaciais.part(i).y]
      pontos.append(ponto)
    pontos = np.array(pontos, dtype = np.int32)
    cv2.polylines(imagem, [pontos], p68[k][2], (255, 0, 0), 2)


fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL
imagem = cv2.imread('/content/drive/My Drive/imagem_e_recurso/fotos/treinamento/ronald.0.1.jpg')
detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('/content/drive/My Drive/imagem_e_recurso/recursos/shape_predictor_68_face_landmarks.dat')
facesDetectadas = detectorFace(imagem, 2)
for face in facesDetectadas:
  pontos = detectorPontos(imagem, face)
  print(pontos.parts())
  print(len(pontos.parts()))
  #imprimePontos(imagem, pontos)
  #imprimeNumeros(imagem, pontos)
  imprimeLinhas(imagem, pontos)
cv2_imshow(imagem)
