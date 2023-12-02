import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import HandTrackingModule as htm
import time
import subprocess

def main():
    # los anchos y altos de la pantalla y la camara
    wScr, hScr = pyautogui.size()
    wCam, hCam = 640, 480


    pTime = 0

    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = htm.handDetector(maxHands= 1)

    # para delimitar el recorrido del mouse
    frameR = 100
    # Para que el mouse se mueva mas acorde al movimiento del dedo
    smoothening = 3
    plocX, plocY = 0, 0

    while True:
        success, img = cap.read()
        # Obtengo los landmarks ...
        img = detector.findHands(img)
        # ... y las posiciones de cada landmark
        lmList = detector.findPositions(img)

        # Obtengo la punta de los dedos indice y medio
        if len(lmList) != 0:
            x1, y1 = lmList[8][1:]

        # Chequeo que dedos están arriba
        fingers = detector.fingersUp()

        # Hago un marco para asi poder controlar el alto y bajo del mouse
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if len(fingers) != 0:
            # Si el indice esta arriba y el del medio esta abajo, entro en modo de movimiento
            if fingers[1] == 1 and fingers[2] == 0:

                # Posicion del dedo indice
                # Restringir el rango de movimiento
                x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

                # Limitar el rango de movimiento
                x3 = np.clip(x3, 0, wScr)
                y3 = np.clip(y3, 0, hScr)

                # Ajustar el factor de suavizado
                smoothening = 10

                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                pyautogui.moveTo(wScr - clocX, clocY)
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                plocX, plocY = clocX, clocY

            # Si el indice y el dedo medio estan arriba, entro en modo click
            if fingers[1] == 1 and fingers[2] == 1:
                length, img, lineInfo = detector.findDistance(8, 12, img)
                if length < 32:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    pyautogui.click()

            if fingers[1] == 1 and fingers[4] == 1:
                subprocess.Popen(['open', '-a', 'Calendar'])

            if sum(fingers[1:4]) > 2:
                subprocess.Popen(['open', '-a', 'Notes'])
        # Escribe el fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Result", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()