import cv2
import threading
import copy
import math
import numpy as np
from pydub import AudioSegment
from pydub.playback import play
from random import randint

model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
songBuzzer = AudioSegment.from_wav('BuzzerWav.wav')
songDoll = AudioSegment.from_wav('DollSing.wav')
songCongrats = AudioSegment.from_wav('Congratulations.wav')
karakter = cv2.imread('IlNam.jpg')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

xAim = randint(80, 560)
yAim = randint(80, 400)
karakterLoc, karakterLoc2 = 80, 560
threshold, motion = 20, 1
isBgCaptured, count, mulai, opening, finish, batas, lock, res = 0,0,0,0,0,0,0,0
timerWaktu, done, mins, secs = 0, 0, 0, 37

ret, frame1 = cap.read()
ret, frame2 = cap.read()

def timer(lock, finish, timerW, done, mins, secs):
    timerW1, done1, mins1, secs1 = timerW, done, mins, secs
    if done1 == 0:
        if finish < 1 and lock == 0:
            if timerW1 == 11:
                timerW1 = 0
                secs1 = secs - 1
            if secs1 < 0:
                secs1 = 59
                mins1 = mins - 1
            timerW1 += 1
        elif finish < 0 and lock == 1:
            mins1 = mins1
            secs1 = secs1

    if mins1 == 0 and secs1 == 0:
        done1 = 1
        mins1 = 0
        secs1 = 0
    timerFormat = '{:02d}:{:02d}'.format(mins1, secs1)
    cv2.rectangle(frame1, (520, 10), (610, 50), (0, 0, 155), 1)
    cv2.putText(frame1, timerFormat, (530, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 155), 2)
    return timerW1, done1, mins1, secs1;

def calculateFingers(res,drawing):
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 4:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):
            cnt = 1
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                #angle = (math.acos((b**2 + c**2 - a**2)/(2*b*c))*180)/3.14
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))*57
                if angle <= 90 and d>30:
                    cnt += 1
                    cv2.circle(drawing, far, 5, [255, 0, 0], -2)
                #print (start)
            return True, cnt
    return False, 0

def fingerDetection(img, threshold, lokasi, batas):
    batas2 = batas
    lokasi2 = lokasi
    imgF = img[0:300, 0:250]
    gray2 = cv2.cvtColor(imgF, cv2.COLOR_BGR2GRAY)
    blur2 = cv2.GaussianBlur(gray2, (41, 41), 0)
    ret, thresh2 = cv2.threshold(blur2, threshold, 255, cv2.THRESH_BINARY)
    thresh2 = copy.deepcopy(thresh2)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length2 = len(contours2)
    maxArea2 = -5
    if length2 > 0:
        for j in range(length2):
            temp2 = contours2[j]
            area2 = cv2.contourArea(temp2)
            if area2 > maxArea2:
                maxArea2 = area2
                ci2 = j

        res2 = contours2[ci2]
        hull2 = cv2.convexHull(res2)
        drawing2 = np.zeros(imgF.shape, np.uint8)
        #cv2.drawContours(frame1, [res2], 0, (0, 255, 0), 1)
        # cv2.drawContours(frame1, [hull2], 0, (0, 0, 255), 1)
        isFinishCal, cnt2 = calculateFingers(res2, drawing2)
        areahull2 = cv2.contourArea(hull2)
        areacnt2 = cv2.contourArea(res2)
        arearatio2 = ((areahull2 - areacnt2) / areacnt2) * 100
        if arearatio2 < 10:
            cnt2 = 0
        if cnt2 == 0:
            batas2 = batas + 1
        elif cnt2 > 0:
            batas2 = 0
        if batas2 == 1:
            cv2.putText(frame1, "GO!", ((640-lokasi2), 440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2,
                        (0, 156, 0), 2)
            lokasi2 = lokasi - 5
            batas2 += 1
        #print(batas2)
        #print(cnt2)
    return batas2, lokasi2;

def faceDetection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    wajah = model.detectMultiScale(gray)
    for (x, y, w, h) in wajah:
        #cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 1)
        xAim = x + (w / 2)
        yAim = y + 30
        threading.Thread(target=aimRed(int(xAim), yAim, 10, 0)).start()
        #frame1 = cameraZoom(int(xAim), yAim, frame)

def karakterIlnam(karakter, frame, lokasi):
    loc = lokasi
    size = 50
    karakterX = cv2.resize(karakter, (size, size))
    karakterGray = cv2.cvtColor(karakterX, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(karakterGray, 1, 255, cv2.THRESH_BINARY)
    roi = frame[-size-10:-10, -size-loc:-loc]
    roi[np.where(mask)] = 0
    roi += karakterX

def bgGame(karakter, frame1, karakterLoc):
    cv2.rectangle(frame1, (0, 410), (640, 480), (255, 255, 255), -1)
    cv2.line(frame1, (40, 470), (600, 470), [100, 100, 100], 2)
    karakterIlnam(karakter, frame1, karakterLoc)
    cv2.putText(frame1, "{}".format('Start'), (10, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 48), 2)
    cv2.putText(frame1, "{}".format('Finish'), (590, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 0, 48), 2)

def aimRed(x, y, scale, line):
    cv2.circle(frame1, (x, y), 20 + scale, [0, 0, 255], 2+line)
    cv2.circle(frame1, (x, y), 35 + scale, [0, 0, 255], 1+line)
    cv2.line(frame1, (x-45-scale, y), (x+45+scale, y), [0, 0, 255], 2+line)
    cv2.line(frame1, (x, y-45-scale), (x, y+45+scale), [0, 0, 255], 2+line)

def gameOver(done):
    cv2.putText(frame1, "{}".format('Game Over!'), (210, 455),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
    if done == 0:
        cv2.putText(frame1, "{}".format('Motion Detected!'), (195, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(frame1, "{}".format('Time Out!'), (250, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def lampuMerah():
    cv2.circle(frame1, (300, 40), 25, [0, 0, 255], -10)
    cv2.circle(frame1, (300, 40), 25, [50, 50, 50], 2)
    cv2.circle(frame1, (360, 40), 22, [50, 50, 50], 2)

def lampuHijau():
    cv2.circle(frame1, (300, 40), 22, [50, 50, 50], 2)
    cv2.circle(frame1, (360, 40), 25, [0, 255, 0], -10)
    cv2.circle(frame1, (360, 40), 25, [50, 50, 50], 2)

def playSongDoll():
    play(songDoll)

def playSongBuzzer():
    play(songBuzzer)

def playSongCongrats():
    play(songCongrats)

def removeBG(frame, bgModel):
    fgmask = bgModel.apply(frame,learningRate=0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

while cap.isOpened():
    frame3 = cv2.bilateralFilter(frame1, 5, 50, 100)
    if isBgCaptured == 1:
        img = removeBG(frame3, bgModel)
        if mulai == 1:
            if lock != 1:
                if count == 0:
                    threading.Thread(target=playSongDoll).start()
                count = count + 1
                if count > 0 and count < 50:
                    motion = 1
                elif count >= 50 and count <= 110:
                    motion = 0
                else:
                    count = 0
            #print(timerWaktu)
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            imgX = img[0:int(1 * frame3.shape[0]),
                   int(0 * frame3.shape[1]):frame3.shape[1]]

            gray1 = cv2.cvtColor(imgX, cv2.COLOR_BGR2GRAY)
            blur1 = cv2.GaussianBlur(gray1, (41, 41), 0)
            ret, thresh1 = cv2.threshold(blur1, threshold, 255, cv2.THRESH_BINARY)

            thresh1 = copy.deepcopy(thresh1)
            contours1, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            length1 = len(contours1)
            maxArea = -5
            if length1 > 0:
                for i in range(length1):
                    temp = contours1[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
                res = contours1[ci]
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    if cv2.contourArea(contour) < 1000:
                        continue
                    if motion == 0:
                        if lock == 0:
                            lock = 1
                            xAim = x + (w / 2)
                            yAim = y + (h / 2)
                            aimRed(int(xAim), int(yAim), 10, 0)
                            cv2.drawContours(frame1, [res], 0, (0, 0, 255), 4)
                            gameOver(done)
                        else:
                            cv2.drawContours(frame1, [res], 0, (255, 255, 255), 1)
                    else:
                        cv2.drawContours(frame1, [res], 0, (255, 255, 255), 1)

            timerWaktu, done, mins, secs = timer(lock, finish, timerWaktu, done, mins, secs)
            bgGame(karakter, frame1, karakterLoc)

            if karakterLoc <= 40:
                mulai = 0
                finish = 1

    if motion == 1 and mulai == 1 and finish == 0:
        karakterLoc2 = karakterLoc
        batas, karakterLoc = fingerDetection(img, threshold, karakterLoc, batas)
        #karakterLoc = int(karakterLoc[1])
        if karakterLoc2 > karakterLoc:
            batas = 500
        cv2.rectangle(frame1, (0, 0), (250, 300), (0, 0, 0), 1)
        lampuHijau()
    elif motion == 0 and mulai == 1  and finish == 0:
        lampuMerah()
        if lock == 1:
            cv2.line(frame1, (40, 470), (600, 470), [0, 0, 255], 2)
            threading.Thread(target=faceDetection(frame1)).start()
            threading.Thread(target=playSongBuzzer).start()
            # cv2.rectangle(frame1, (x,y), (x+w, y+h), (0, 0, 255), 1)
            #cv2.drawContours(frame1, [res], 0, (0, 255, 255), 1)
            gameOver(done)
        else:
            cv2.drawContours(frame1, [res], 0, (0, 255, 0), 0)

    if motion == 0 and lock == 0 and mulai == 1:
        if count > 60 and count % 10 == 0:
            xAim = randint(80, 560)
            yAim = randint(80, 400)
        threading.Thread(target=aimRed(xAim, yAim, 0, 0)).start()

    if finish >= 1:
        cv2.putText(frame1, "{}".format('CONGRATULATIONS!'), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        bgGame(karakter, frame1, karakterLoc)
        timerWaktu, done, mins, secs = timer(lock, finish, timerWaktu, done, mins, secs)
        if finish == 1:
            threading.Thread(target=playSongCongrats).start()
            finish += 1

    cv2.imshow("Lampu Merah Lampu Hijau", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    k = cv2.waitKey(10)
    if k == 27:
        break
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        isBgCaptured = 1
        print('\nBackground Captured!')
    elif k == ord('m'):
        opening = 60
        print('Memulai...')
    elif k == ord('r'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, 50)
        timerWaktu, done, mins, secs = 0, 0, 2, 0
        karakterLoc, karakterLoc2 = 550, 560
        threshold, motion = 20, 1
        isBgCapture, count, mulai, opening, finish, batas, lock, res = 0, 0, 0, 0, 0, 0, 0, 0
        print('Reset Background!')
    if opening != 0:
        cv2.putText(frame1, "{}".format('GAME STARTS IN'), (200, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame1, "{}".format(math.ceil(opening/20)), (310, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
        opening -= 1
        if opening == 1:
            mulai, opening, motion, isBgCapture = 1, 0, 1, 1

cv2.destroyAllWindows()
cap.release()