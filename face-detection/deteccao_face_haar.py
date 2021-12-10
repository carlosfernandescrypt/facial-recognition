from google.colab.patches import cv2_imshow #so utiliza essa funçao pro colab, pq ele n tem suporte a imshow
import cv2

img = cv2.imread('/content/drive/My Drive/fotos/grupo.0.jpg', cv2.IMREAD_UNCHANGED)
classificador = cv2.CascadeClassifier('/content/drive/My Drive/imagem_e_recurso/recursos/haarcascade_frontalface_default.xml') #importando um arquivo ja treinado
imagemCinza = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convertendo a imagem para cinza
facesDetectadas = classificador.detectMultiScale(imagemCinza, scaleFactor = 1.2, minSize = (50, 50)) #detectando as faces, aumentando a imagem, e passando um paramentro de tamanho minimo requirido da face para que ele possa identificar
print(facesDetectadas)
print("Faces Detectadas: ", len(facesDetectadas))
for (x, y, l, a) in facesDetectadas:
  cv2.rectangle(img, (x, y), (x + l, y + a), (0, 255, 0), 2) #colocado as bordas na imagem 
cv2_imshow(img) #mostrando a imagem


####################################################################################
#import cv2
#imagem = cv2.imread("/content/drive/My Drive/fotos")
#cv2.imshow("Detector haar", imagem)
#cv2.waitKey(0) #carregar a imagem, e apos apertar alguma tecla, a imagem ira fechar
#cv2.destroyAllWindows()
####################################################################################