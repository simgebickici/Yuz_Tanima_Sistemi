import cv2
import numpy as np
from PIL import Image
import os

yol = 'veriseti'

taniyici = cv2.face.LBPHFaceRecognizer_create()
yuz_denetleyici = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml');

def goruntuEtiketleme(yol):

    goruntu_yollari = [os.path.join(yol,f) for f in os.listdir(yol)]     
    yuz_ornekler=[]
    idler = []

    for goruntu_yolu in goruntu_yollari:

        PIL_goruntu = Image.open(goruntu_yolu).convert('L') 
        numpy_goruntu = np.array(PIL_goruntu,'uint8')

        id = int(os.path.split(goruntu_yolu)[-1].split(".")[1])
        yuzler = yuz_denetleyici.detectMultiScale(numpy_goruntu)

        for (x,y,w,h) in yuzler:
            yuz_ornekler.append(numpy_goruntu[y:y+h,x:x+w])
            idler.append(id)

    return yuz_ornekler,idler

print ("\n Alınan görüntüler eğitiliyor ...")
yuzler,idler = goruntuEtiketleme(yol)
taniyici.train(yuzler, np.array(idler))

taniyici.write('egitimislemi/trainer.yml') 

print("\n {0} tane yüz eğitildi.".format(len(np.unique(idler))))