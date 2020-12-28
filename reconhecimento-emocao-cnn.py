import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Definição de variáveis
arquivo_modelo = "modelo_expressoes.h5"
arquivo_modelo_json = "modelo_expressoes.json"
diretorio = "./Resources"
largura, altura = 48, 48

num_features = 64
num_labels = 6
batch_size = 64
epochs = 200

true_y = []
pred_y = []

# Acessando a base com fotos de expressões faciais
data = pd.read_csv(diretorio + "/fer2013/fer2013.csv")
sem_nojo = data["emotion"] != 1
data = data[sem_nojo]
data.tail()

plt.figure(figsize=(12,6))
plt.hist(data["emotion"], bins = 30)
plt.title("Imagens x emoções")


# Pré-processamento das imagens e emocoes
def pre_processamento():
    faces = []
    pixels = data["pixels"].tolist()
    
    for sequencia_pixel in pixels:
      face = [int(pixel) for pixel in sequencia_pixel.split(" ")]
      face = np.asarray(face).reshape(largura, altura)
      faces.append(face)
    
    faces = np.asarray(faces)
    faces.shape
    
    faces = np.expand_dims(faces, -1)
    faces.shape
    
    faces = faces.astype("float32")
    faces = faces / 255.0
    
    emocoes = pd.get_dummies(data["emotion"]).values
    
    return (faces, emocoes)

def arquitetura_rede():
    model = Sequential()
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation="relu",
                     input_shape=(largura, altura, 1), kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    
    model.add(Dense(2*2*2*num_features, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2*2*num_features, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(2*num_features, activation="relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_labels, activation="softmax"))
    
    model.compile(loss=categorical_crossentropy, optimizer="adam", metrics = ["accuracy"])
    
    return model

# Gerando gráfico da melhora em cada etapa do treinamento
def plota_historico_modelo(historico_modelo):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    axs[0].plot(range(1, len(historico_modelo.history["accuracy"]) + 1), 
                historico_modelo.history["accuracy"], "r")
    axs[0].plot(range(1, len(historico_modelo.history["val_accuracy"]) + 1),
                historico_modelo.history["val_accuracy"], "b")
    axs[0].set_title("Acurácia do modelo")
    axs[0].set_ylabel("Acurácia")
    axs[0].set_xlabel("Epoch")
    axs[0].set_xticks(np.arange(1, len(historico_modelo.history["accuracy"]) + 1),
                      len(historico_modelo.history["accuracy"]) / 10)
    axs[0].legend(["training accuracy", "validation accuracy"], loc = "best")
    
    axs[1].plot(range(1, len(historico_modelo.history["loss"]) + 1),
                historico_modelo.history["loss"], "r")
    axs[1].plot(range(1, len(historico_modelo.history["val_loss"]) + 1),
                historico_modelo.history["val_loss"], "b")
    axs[1].set_title("Loss do modelo")
    axs[1].set_ylabel("Loss")
    axs[1].set_xlabel("Epoch")
    axs[1].set_xticks(np.arange(1, len(historico_modelo.history["loss"]) + 1),
                      len(historico_modelo.history["loss"]) / 10)
    axs[1].legend(["training loss", "validation loss"], loc = "best")
    fig.savefig("historico_modelo.png")

def set_variaveis_teste():
    y_pred = model.predict(x_test)
    yp = y_pred.tolist()
    yt = y_test.tolist()
    count = 0

    for i in range(len(y_test)):
      yy = max(yp[i])
      yyt = max(yt[i])
      pred_y.append(yp[i].index(yy))
      true_y.append(yt[i].index(yyt))
      if (yp[i].index(yy) == yt[i].index(yyt)):
        count += 1
    
# Gerando a Matriz de Confusão
def plotar_matriz_confusao():
    set_variaveis_teste()
    cm = confusion_matrix(true_y, pred_y)
    expressoes = ["Raiva", "Medo", "Feliz", "Triste", "Surpreso", "Neutro"]
    titulo = "Matriz de confusao"
    print(cm)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(titulo)
    plt.colorbar()
    tick_marcks = np.arange(len(expressoes))
    plt.xticks(tick_marcks, expressoes, rotation = 45)
    plt.yticks(tick_marcks, expressoes, rotation = 45);
    
    fmt = "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
               color="white" if cm[i,j] > thresh else "black")
    
    plt.ylabel("Classificaçãão Correta")
    plt.xlabel("Prediçãão")
    plt.savefig("matriz_confusao.png")


faces, emocoes = pre_processamento()

# Divide em conjuntos para treinamento e validação
x_train, x_test, y_train, y_test = train_test_split(faces, emocoes, test_size = 0.1, random_state = 42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state = 42)

print("Número de imagens no conjunto de treino", len(x_train))
print("Número de imagens no conjunto de teste", len(x_test))
print("Número de imagens no conjunto de validação", len(x_val))

# Data Augmentation
datagen = ImageDataGenerator(
      rotation_range=30,       # Valor de alcance para randomicamente rotacionar a imagem (exemplo: se for 30 vai rotacionar no maximo 30 graus)
      shear_range=0.1,         # Aleatoriamente distorce a imagem (por cisalhamento)
      zoom_range=0.3,          # Aleatoriamente aplica zoom na imagem 
      width_shift_range=0.1,   # Aleatoriamente alterna as imagens horizontalmente (o valor do parâmetro corresponde à fração da largura total)
      height_shift_range=0.1,  # Aleatoriamente alterna as imagens verticalmente (o valor do parâmetro corresponde à fração da largura total)
      horizontal_flip=True,    # Aleatoriamente vira as imagens na horizontal
      vertical_flip=False,     # Aleatoriamente vira as imagens na vertical
      fill_mode="nearest")

# Callbacks usadas quando o modelo é treinado
lr_reducer = ReduceLROnPlateau(monitor="val_loss", factor=0.9, patience=3, verbose=1)
early_stopper = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=1, mode="auto")
checkpointer = ModelCheckpoint(arquivo_modelo, monitor="val_loss", verbose=1, save_best_only=True)

model = arquitetura_rede()

# Salvando a arquitetura do modelo em um arquivo JSON
model_json = model.to_json()
with open(arquivo_modelo_json, "w") as json_file:
  json_file.write(model_json)

# Treinando o modelo
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    validation_steps = len(x_val) // batch_size,
                    steps_per_epoch=len(x_train) // batch_size,
                    callbacks=[lr_reducer, early_stopper, checkpointer])

session.close()

plota_historico_modelo(history)

# Verificando a acurácia do modelo
scores = model.evaluate(np.array(x_test), np.array(y_test), batch_size = batch_size)

print("Acurácia: " + str(scores[1]))
print("Erro: " + str(scores[0]))

plotar_matriz_confusao()