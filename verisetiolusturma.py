import cv2
 
kamera = cv2.VideoCapture(0)

yuz_denetleyici = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

id=input("\n id numarası giriniz:  ") 

sayac=0

while (True):
    ret, videoGoruntu = kamera.read()
    gri = cv2.cvtColor(videoGoruntu, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_denetleyici.detectMultiScale(gri, 1.3, 5)
    
    for (x,y,w,h) in yuzler:
        cv2.rectangle(videoGoruntu, (x,y), (x+w,y+h), (255,0,0), 2)
        sayac+=1
        cv2.imwrite("veriseti/User." + str(id) + '.' + str(sayac) + ".jpg", gri[y:y+h,x:x+w])
        cv2.imshow('goruntu', videoGoruntu)
    
    a=cv2.waitKey(50) & 0xFF

    if sayac>=45:
        break
    elif a== ord('q'):
        break 
    
print("Program kapatılıyor...")
kamera.release()
cv2.destroyAllWindows()
