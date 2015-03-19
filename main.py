import cv2
import numpy
import sys

def main():
    device = 0
    capture = cv2.VideoCapture(device)
    waitForStabilization(capture)


def waitForStabilization(capture):
    ok, frame = capture.read()
    w = frame.shape[1]
    h = frame.shape[0]
    prev_avg = numpy.empty_like(frame).astype(float)
    new_avg = numpy.empty_like(frame).astype(float)

    while 1:

        new_avg = numpy.empty_like(frame).astype(float)

        for i in range(10):
            ok, frame = capture.read()
            new_avg += frame

        new_avg = (new_avg / 10).astype('uint8')
            
        differences = new_avg.astype(float) - prev_avg.astype(float)
        norm = numpy.sqrt(differences*differences).astype('uint8')


        sum = numpy.sum(norm)
        print sum
        prev_avg = new_avg

        if sum < 7000000:

            cv2.putText(norm, "STABLE", (16, h-16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,0,0), 3, cv2.CV_AA)

            cv2.putText(norm, "STABLE", (16, h-16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255,255,255), 1, cv2.CV_AA)
        else:
            cv2.putText(norm, "UNSTABLE", (16, h-16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0,0,0), 3, cv2.CV_AA)

            cv2.putText(norm, "UNSTABLE", (16, h-16), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255,255,255), 1, cv2.CV_AA)

        cv2.imshow('Video', norm)

        



if __name__ == '__main__':
    main()

