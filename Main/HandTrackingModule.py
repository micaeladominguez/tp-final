import cv2
import mediapipe as mp
import time
import math


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        # Parameters basics de hands; mode es false para que asi pueda detectar y si el tracking confidence es
        # bueno, seguirá rastreando
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # Mapeo de las manos
        self.mpHands = mp.solutions.hands
        # El objeto manos
        self.hands = self.mpHands.Hands(self.mode, self.maxHands)
        # Para dibujar las landmarks
        self.mpDraw = mp.solutions.drawing_utils
        # Ubicación de la punta de los dedos
        self.tipIds = [4, 8, 12, 16, 20]

    # Para encontrar las manos
    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Si detecta algo
        if self.results.multi_hand_landmarks:
            # Dibuja el mapa de cada mano
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # para identificar los landmarks
    def findPositions(self, img, handNo=0):
        # Listas para el id del landmark y sus posiciones
        self.lmList = []
        if self.results.multi_hand_landmarks:
            # Busco en mi primer mano
            myHand = self.results.multi_hand_landmarks[handNo]
            # Busco todos los landmarks de la mano
            for id, lm in enumerate(myHand.landmark):
                # Chequeo el largo, ancho y canal de la imagen
                h, w, c = img.shape
                # Obtengo el centro en x e y del landmark
                cx, cy = int(lm.x * w), int(lm.y * h)

                self.lmList.append([id, cx, cy])
        return self.lmList

    # para ver que dedos estan arriba
    def fingersUp(self):
        fingers = []
        if len(self.lmList) != 0:
            # Pulgar

            # Primero chequeo que mano estoy mostrando, si la punta del meñique esta a la derecha de la punta del
            # pulgar [4] , es la mano derecha. Y como el pulgar no puede ir abajo, lo que hago es ver si la punta
            # del pulgar esta a la izquierda del nudillo del pulgar [3], y asi lo considero como abierto
            if self.lmList[self.tipIds[4]][1] < self.lmList[self.tipIds[0]][1]:
                if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                # si la punta del meñique esta a la izquierda, entonces es la mano izquierda y veo que el
                # dedo este a la derecha del nudillo para asi considerarlo abierto
                if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            # Dedos
            for id in range(1, 5):
                # Cheque que el valor y de la punta de cada dedo [8, 12, 16, 20] sea de menor que el de la posicion
                # de los nudillos [6, 10, 14, 18], si se cumple, los dedos están arriba
                if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)

        return fingers


    # para obtener distancia entre 2 dedos
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        # Obtengo los valores x e y de los puntos p1 y p2 asi como tambien del centro entre esos puntos
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Resalto los puntos y dibujo una linea entre ellos
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        # Obtengo la distancia entre los puntos
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector(maxHands=1)

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPositions(img)
        fingers = detector.fingersUp()
        print(fingers)

        # Escribe el fps
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 78), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(2)


if __name__ == "__main__":
    main()