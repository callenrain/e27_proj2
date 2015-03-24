import cv2
import numpy
import sys
import cvk2

def main():

    device = 0
    capture = cv2.VideoCapture(device)
    M = calibrateCamera(capture)

    ok, frame = capture.read()
    while True:
        ok, frame = capture.read()

        # Convert dimensions of frame into array of corners
        corners = getCorners(frame)

        # Get the location size of the transformed corners
        boxOrig, boxDims = getBoundingBox(img1Corners, M)

        # make a canvas for the rectified frame
        rectified = numpy.zeros( (boxDims[1], boxDims[0], 3), dtype='uint8' )

        # Transform the original image using the homography in addition to the 
        # translation
        cv2.warpPerspective(frame, M, tuple(boxDims), rectified)

        cv2.imshow('recified_frame', rectified)

# Given an image, return an array of the corner location
def getCorners(img):
    h,w,d = img.shape
    return numpy.array( [ [[ 0, 0 ]],
                       [[ w, 0 ]],
                       [[ w, h ]],
                       [[ 0, h ]] ], dtype='float32' )

# Given a set of corners and a homography, determine the bounding box for 
# the transformation
def getBoundingBox(corners, M):
    transImgCorners = cv2.perspectiveTransform(corners,M)
    box = cv2.boundingRect(transImgCorners)
    return box[0:2], box[2:4]

def calibrateCamera(device):
    h, w = 1000,1000
    white_grid = numpy.empty((h, w, 3))
    white_grid[:] = (255,255,255)
    circles = [(h/4, w/4), (h/2, w/4), (3*h/4, w/4), (h/4, w/2), (h/2, w/2), \
        (3*h/4, w/2), (h/4, 3*w/4), (h/2, 3*w/4), (3*h/4, 3*w/4)]
    
    #will be the destination points used in the homgraphy
    #image_points = numpy.empty((0, 1, 2), dtype='float32')
    image_points = []

    #needed to index image_points array down below
    for circle in circles:
        dot_grid = numpy.empty((h, w, 3))
        dot_grid[:] = (255,255,255)
        cv2.circle(dot_grid, circle, 15, (0,0,0), -1)
        
        cv2.imshow('white_grid', white_grid)

        #try to think of a way to get a good averaged image that
        #we can use to background subtract each stable image

        #for now, this will be the stable blank image we use
        stable_blank_frame = waitForStabilization(device)
        cv2.waitKey(10)

        cv2.imshow("dot", dot_grid)
        stable_frame = waitForStabilization(device)
        cv2.waitKey(10)

        #background subtraction
        no_background = stable_blank_frame.astype('float') - stable_frame.astype('float')
        no_background = numpy.sqrt(no_background*no_background).astype('uint8')

        #threshold this image (need to figure good values)
        mask = cv2.threshold(no_background, 50, 255, cv2.THRESH_BINARY)[1]

        #cv2.imshow('grid', mask)
        #while cv2.waitKey(15) < 0: pass

        #dilate
        size = 5
        elt = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        mask = cv2.erode(mask, elt)
        mask = cv2.dilate(mask, elt)

        #cv2.imshow('grid', mask)
        #while cv2.waitKey(15) < 0: pass

        # Find the contours in the image
        contours = cv2.findContours(mask, cv2.RETR_CCOMP,
                        cv2.CHAIN_APPROX_SIMPLE)

        #get info from contours
        info = cvk2.getcontourinfo(contours[0][0])

        #add point to array
        image_points.append(info['mean'])

        #numpy.insert(image_points, len(image_points), info['mean'], axis=0)


    circle_floats = [[float(x), float(y)] for x,y in circles]

    #now we get a homography using collected points
    homography = cv2.findHomography(numpy.array(image_points), numpy.array(circle_floats))
    return homography


#new implementation using eroding (original implementation commented out
#below. We still need to figure out a way to get a good inital average
#image to do background subtraction
def waitForStabilization(capture):
    NUM_FRAMES = 2
    ok, frame = capture.read()

    w = frame.shape[1]
    h = frame.shape[0]
    frame_gray = numpy.empty((h, w), 'uint8')

    prev_avg = numpy.empty((h, w), 'float')
    
    while 1:
        new_avg = numpy.zeros((h, w), 'float')

        for i in range(NUM_FRAMES):
            ok, frame = capture.read()
            frame_gray = numpy.empty((h, w), 'uint8')
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            new_avg += frame_gray

        new_avg = (new_avg / NUM_FRAMES).astype('uint8')

        differences = new_avg.astype(float) - prev_avg.astype(float)
        differences = numpy.sqrt(differences*differences).astype('uint8')

        #threshold this image
        mask = cv2.threshold(differences.astype('uint8'), 20, 255, cv2.THRESH_BINARY)[1]

        erodeSize = 20
        erodeElt = cv2.getStructuringElement(cv2.MORPH_RECT,(erodeSize,erodeSize))
        mask = cv2.erode(mask, erodeElt)

        #still need to figure out a good value (may want to ensure multiple
        #frames pass criteria
        
        if numpy.sum(mask) == 0:
            #label = "STABLE"
            return frame_gray
        else: 
            label = "UNSTABLE"

        # cv2.putText(mask, label, (16, h-16),
        #        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #        (0,0,0), 3, cv2.CV_AA)

        # cv2.putText(mask, label, (16, h-16),
        #        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        #        (255,255,255), 1, cv2.CV_AA)


        # cv2.imshow('Video', mask)
        prev_avg = new_avg


if __name__ == '__main__':
    main()

