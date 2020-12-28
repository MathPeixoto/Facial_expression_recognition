import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import model_from_json

diretorio = "./Resources"
cascade_faces = diretorio + "/haarcascade_frontalface_default.xml"
caminho_modelo_h5 = "modelo_expressoes.h5"
caminho_modelo_json = "modelo_expressoes.json"
expressoes = ["Raiva", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
arquivo_video = diretorio + "/video_teste06.MOV"
redimensionar = True
largura_maxima = 600    # Define o tamanho da largura máxima do vídeo a ser salvo
cap = cv2.VideoCapture(arquivo_video) # Carrega vídeo
conectado, video = cap.read()


def carregar_modelo():
    with open(caminho_modelo_json, "r") as json_file:
        modelo_json = json_file.read()
        
    modelo_carregado = model_from_json(modelo_json)
    modelo_carregado.summary()
    modelo_carregado.load_weights(caminho_modelo_h5)
    return modelo_carregado

def detectar_emocao(frame):
    face_cascade = cv2.CascadeClassifier(cascade_faces)
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # converte pra grayscale
    faces = face_cascade.detectMultiScale(cinza,scaleFactor=1.2, minNeighbors=5,minSize=(30,30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:

            frame = cv2.rectangle(frame, (x,y), (x+w,y+h+10), (255,50,50), 2) # Desenha retângulo ao redor da face

            roi = cinza[y:y + h, x:x + w]      # Extrai apenas a região de interesse (ROI) que é onde contém o rosto 
            roi = cv2.resize(roi, (48, 48))    # Antes de passar pra rede neural redimensiona para o tamanho das imagens de treinamento
            roi = roi.astype("float") / 255.0  # Normaliza
            roi = img_to_array(roi)            # Converte para array para que a rede possa processar
            roi = np.expand_dims(roi, axis=0)  # Muda o shape do array

            result = classificador_emocoes.predict(roi)[0]
                
            if result is not None:
                resultado = np.argmax(result) # Encontra a emoção com maior probabilidade
                cv2.putText(frame, expressoes[resultado], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255,255,255), 1, cv2.LINE_AA) # Escreve a emoção acima do rosto



if (redimensionar and video.shape[1]>largura_maxima):
  # Para que a imagem não fique com aparência esticada
  # deve-se calcular a proporção (largura/altura) e usar esse valor para calcular
  # a altura com base na largura definida acima
  proporcao = video.shape[1] / video.shape[0]
  video_largura = largura_maxima
  video_altura = int(video_largura / proporcao)
else:
  video_largura = video.shape[1]
  video_altura = video.shape[0]

fourcc = cv2.VideoWriter_fourcc(*"XVID")    # Definição do codec
                                            # Codecs mais usados: XVID, MP4V, MJPG, DIVX, X264... 
                                            # Por exemplo, para salvar em formato mp4 utiliza-se o codec mp4v
                                            # (o nome do arquivo também precisa possuir a extensão .mp4)
saida_video = cv2.VideoWriter("resultado_video.avi", fourcc, 20, (video_largura, video_altura))
    
classificador_emocoes = carregar_modelo()

while (cv2.waitKey(1) < 0):
    conectado, frame = cap.read()
    
    if not conectado:
        break  # Se ocorreu um problema ao carregar a imagem então interrompe o programa
    
    if redimensionar: # Se redimensionar = True então redimensiona o frame para os novos tamanhos
      frame = cv2.resize(frame, (video_largura, video_altura))
      
    detectar_emocao(frame)
    cv2.imshow("image", frame) 
    saida_video.write(frame) # Grava o frame atual


cap.release()
saida_video.release() 
cv2.destroyAllWindows()
