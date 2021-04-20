import cv2
import datetime

taniyici = cv2.face.LBPHFaceRecognizer_create()
taniyici.read('egitimislemi/trainer.yml')
cascadeYolu = "cascades/haarcascade_frontalface_default.xml"
yuzCascade = cv2.CascadeClassifier(cascadeYolu);

yazi_tipi = cv2.FONT_HERSHEY_PLAIN

id = 0

isimler = ['Unknown', 'Isim1'] 

kamera = cv2.VideoCapture(0)

minGenislik = 0.1*kamera.get(3)
minYukseklik = 0.1*kamera.get(4)

while True:

    ret, videoGoruntu =kamera.read()

    gri = cv2.cvtColor(videoGoruntu,cv2.COLOR_BGR2GRAY)

    yuzler = yuzCascade.detectMultiScale( 
        gri,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minGenislik), int(minYukseklik)),
       )

    for(x,y,w,h) in yuzler:

        cv2.rectangle(videoGoruntu, (x,y), (x+w,y+h), (0,255,0), 2)

        id, guvenilirlik = taniyici.predict(gri[y:y+h,x:x+w])

        f=open("girisYapanlar.txt", "a+")
        if (guvenilirlik < 100):
            id = isimler[id]
            guvenilirlik = "  {0}%".format(round(100 - guvenilirlik))
            f.write("%s burada" % id)
            an = datetime.datetime.now()
            tarih = datetime.datetime.strftime(an, '%c')
            f.write(" -------> %s\n" % tarih)
            f.close()
            cv2.putText(videoGoruntu, str(id), (x+5,y-5), yazi_tipi, 1, (255,255,255), 2)
            cv2.putText(videoGoruntu, str(guvenilirlik), (x+5,y+h-5), yazi_tipi, 1, (255,255,0), 1) 
            cv2.imshow('camera',videoGoruntu) 
            kamera.release()
            cv2.destroyAllWindows()
        else:
            id = "unknown"
            guvenilirlik = "  {0}%".format(round(100 - guvenilirlik))
            cv2.putText(videoGoruntu, str(id), (x+5,y-5), yazi_tipi, 1, (255,255,255), 2)
            cv2.putText(videoGoruntu, str(guvenilirlik), (x+5,y+h-5), yazi_tipi, 1, (255,255,0), 1)
            cv2.imshow('camera',videoGoruntu)
            k = cv2.waitKey(10) & 0xff 
            if k == 27:
                break
            kamera.release()
            cv2.destroyAllWindows()

print("\n Program sonlandırılıyor ...")