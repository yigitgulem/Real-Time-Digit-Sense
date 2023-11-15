import cv2
import numpy as np
from keras.models import load_model
from siniflandirma import preProcess as pp

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

model = load_model("model.h5")

while True:
    success, frame = cap.read()

    img = np.asarray(frame)
    img = cv2.resize(img, (32, 32))
    img = pp(img)
    img = img.reshape(1, 32, 32, 1)

    
    predictions = model.predict(img)
    classindex = np.argmax(predictions, axis=1)[0]  
    probVal = np.max(predictions)  

    print(classindex, probVal)

    if probVal > 0.7:  
        cv2.putText(frame, str(classindex) + " " + str(probVal), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (225, 255, 250), 1)

    cv2.imshow("Rakam Sınıflandırma", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

