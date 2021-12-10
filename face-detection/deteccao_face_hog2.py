from google.colab.patches import cv2_imshow #so utiliza essa funçao pro colab, pq ele n tem suporte a imshow
import cv2
import dlib

subdetector = ["Olhar a frente", "Vista a esquerda", "Vista a direita", "A frente girando a esquerda", "A frente girando a direita"] 

img = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg', cv2.IMREAD_UNCHANGED)
detector = dlib.get_frontal_face_detector()
facesDetectadas, pontuacao, idx = detector.run(img, 0, -1) #idx e o metodo utilizado pela imagem (subdetector)
#print(facesDetectadas)
#print(pontuacao)
#print(idx)
for i, d in enumerate(facesDetectadas):
  #print(i)
  #print(d)
  print('Deteccao: {}, Pontuacao: {}, Subdetector: {}'.format(d, pontuacao[i], subdetector[idx[i]]))
  e, t, d, b = (int(d.left()), int(d.top()), int(d.right()), int(d.bottom()))
  cv2.rectangle(img, (e, t), (d, b), (0, 255, 255), 2)
cv2_imshow(img) #mostrando a imagem