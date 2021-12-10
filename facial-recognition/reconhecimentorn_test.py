from google.colab.patches import cv2_imshow
import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor('/content/drive/My Drive/imagem_e_recurso/recursos/shape_predictor_68_face_landmarks.dat')
reconhecimentoFacial = dlib.face_recognition_model_v1('/content/drive/My Drive/imagem_e_recurso/recursos/dlib_face_recognition_resnet_model_v1.dat')
indices = np.load('/content/drive/My Drive/imagem_e_recurso/recursos/indices_rn.pickle', allow_pickle=True)
descritoresFaciais = np.load('/content/drive/My Drive/imagem_e_recurso/recursos/descritores_rn.npy')
limiar = 0.5


for arquivo in glob.glob(os.path.join('/content/drive/My Drive/fotos', "*.jpg")): #serve para percorrer apenas arquivos jpg pra evitar conflitos
  imagem = cv2.imread(arquivo) #lendo os arquivos
  facesDetectadas = detectorFace(imagem, 2) #criando os bounding box
  for face in facesDetectadas:
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    pontosFaciais = detectorPontos(imagem, face)
    descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
    listaDescritorFacial = [fd for fd in descritorFacial]
    npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype = np.float64)
    npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]
    distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis = 1)
    print("Distancias: {}".format(distancias))
    min = np.argmin(distancias)
    print(min)
    distanciaMinima = distancias[min]

    if distanciaMinima <= limiar:
      nome = os.path.split(indices[min])[1].split('.')[0] #passando os parametros para qual nome a face encontrada vai receber
    else:
      nome = ''
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
    text = "{} {:.4f}".format(nome, distanciaMinima)
    cv2.putText(imagem, text, (d,t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))
  cv2_imshow(imagem)
