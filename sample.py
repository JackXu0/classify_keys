import os
import cv2

def sample_training_videos():
    files = [f for f in os.listdir('dataset/training_videos') if not f.startswith('.')]
    for file in files:
        capture = cv2.VideoCapture('dataset/training_videos/' + file)
        id = file.split(".")[0]
        c = 0
        while capture.isOpened():
            r, f = capture.read()
            if r == False:
                break
            cv2.imwrite('dataset/train/'+id+'_' + str(c) + '.jpg', f)
            c += 1
        capture.release()
        cv2.destroyAllWindows()


def sample_testing_videos():
    files = [f for f in os.listdir('dataset/testing_videos') if not f.startswith('.')]
    for file in files:
        capture = cv2.VideoCapture('dataset/testing_videos/' + file)
        id = file.split(".")[0]
        c = 0
        while capture.isOpened():
            r, f = capture.read()
            if r == False:
                break
            cv2.imwrite('dataset/test/'+id+'_' + str(c) + '.jpg', f)
            c += 1
        capture.release()
        cv2.destroyAllWindows()


def sample_prediction_videos(url):
    capture = cv2.VideoCapture(url)
    c = 0
    while capture.isOpened():
        r, f = capture.read()
        if r == False:
            break
        if c % 5 == 0:
            cv2.imwrite('dataset/prediction/' + str(c//5) + '.jpg', f)
        c += 1
    capture.release()
    cv2.destroyAllWindows()
