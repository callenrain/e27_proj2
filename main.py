import cv2
import numpy
import sys
import cvk2


# MAGIC NUMBERS
NUM_STABILIZATION_FRAMES = 5
CALIBRATE_SIZE = 15
GAME_DIM = 10
FRAME_H, FRAME_W = 700,700
OFFSET = 25
LINE_THICKNESS = 7
BOARD_COLOR = (0, 0, 0)
ALTERNATE_COLOR = (255, 255, 255)

def main():
    X, O = getTemplates()
    for x in X:
        cv2.imshow("X", x)
        cv2.waitKey(-1)
    for o in O:
        cv2.imshow("O", o)
        cv2.waitKey(-1)

    device = 0
    capture = cv2.VideoCapture(device)

    # get a homography and the edges of the board
    

    corners = numpy.array([[[ 371.51908397,  138.27226463]],
         [[ 695.34638197,  150.58600237]],
         [[ 332.39711465,  501.37661352]],
         [[ 700.88139535,  507.14496124]]]).astype('float')

    M = numpy.array([[  2.01198912e+00,   2.27148562e-01,  -7.52986677e+02],
         [ -7.91593761e-02,   2.09536853e+00,  -2.35262232e+02],
         [ -7.21211720e-05,   3.84060651e-04,   1.00000000e+00]])

    #M, corners = calibrateCamera(capture)

    # get the box that bounds the board
    box = getRectifiedBox(corners)

    # get the game board with lines drawn
    board = getGameBoard()

    # keeps track of the moves made so far
    entries = [[None, None, None], [None, None, None], [None, None, None]]

    # TODO: need to create templates based on the size of the box
    templates = getTemplates()


    #cv2.imshow("board", board)
    #cv2.waitKey(-1)

    # obtain a starting reference image
    for i in range(30):
        cv2.imshow("board", board)
        cv2.moveWindow("board", 100, 300)
        reference = getRectifiedImg(capture, M, box)

    # main game look
    while True:
        cv2.imshow("board", board)
        #cv2.moveWindow("board", 100, 300)
        current = getRectifiedImg(capture, M, box)
        cv2.imshow("camera", current)
        cv2.moveWindow("camera", 800, 300)

        if isNotObstructed(current, reference):
            continue
        #cv2.imshow("obstruction", obstruction)
        # if not obstructed
            # check for end of game scenario
            # check for marks
            # if mark is found, match templates
            # if we can classify mark, update the entry list
            # update reference image

# generate the templates for tic tac toe
def getTemplates():
    X, O = [], []
    for size in [.9, .8, .7]:
        for thickness in [4, 2, 1]:
            width = int(size*FRAME_H/3)

            template = numpy.zeros((width, width))

            #generate an X
            cv2.line(template, (0, 0), (width, width), ALTERNATE_COLOR, thickness)
            cv2.line(template, (0, width), (width, 0), ALTERNATE_COLOR, thickness)
            X.append(template)

            template = numpy.zeros((width, width))

            #generate an O
            cv2.circle(template, (int(size*FRAME_H/6), int(size*FRAME_H/6)), int(size*FRAME_H/6)-5, ALTERNATE_COLOR, thickness)
            O.append(template)
    return X, O



# determines if there is a large object in the way of the reference image
def isNotObstructed(current, reference):
    # change to grayscale
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

    # take difference of images
    diff = current.astype('float') - reference.astype('float')
    
    # take the absolute value of the difference and use a high threshold
    norm_diff = numpy.sqrt(diff*diff).astype('uint8')
    mask = cv2.threshold(norm_diff.astype('uint8'), 50, 255, cv2.THRESH_BINARY)[1]

    # erode the iamge so that only large objects remain
    size = 10
    elt = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    mask = cv2.erode(mask, elt)
    #return mask
    #return whether there is an obstruction or not
    return numpy.sum(mask) == 0

# draws the lines for tic tac toe
def getGameBoard():
    board = numpy.empty((FRAME_H, FRAME_W, 3))
    board[:] = BOARD_COLOR
    cv2.line(board, (FRAME_W/3, 0), (FRAME_W/3, FRAME_H), ALTERNATE_COLOR, (LINE_THICKNESS))
    cv2.line(board, (2*FRAME_W/3, 0), (2*FRAME_W/3, FRAME_H), ALTERNATE_COLOR, (LINE_THICKNESS))
    cv2.line(board, (0, FRAME_H/3), (FRAME_W, FRAME_H/3), ALTERNATE_COLOR, (LINE_THICKNESS))
    cv2.line(board, (0, 2*FRAME_H/3), (FRAME_W, 2*FRAME_H/3), ALTERNATE_COLOR, (LINE_THICKNESS))
    return board

# gets the bounding box from the corners of the rectified frame
def getRectifiedBox(corners):
    return cv2.boundingRect(corners.astype('float32'))

# get the rectified image from the bounding box, the camera frame, and the homography
def getRectifiedImg(capture, M, box):
    x, y, w, h = box[0], box[1], box[2], box[3]
    ok, frame = capture.read()
    sm_frame = frame[0:y+h+OFFSET, 0:x+w +OFFSET]
    rectified = cv2.warpPerspective(sm_frame, M, (FRAME_W, FRAME_H))
    return rectified

# use a sequence of dots to obtain a homography and the outline of the board
def calibrateCamera(device):

    blank_grid = numpy.empty((FRAME_H, FRAME_W, 3))
    blank_grid[:] = ALTERNATE_COLOR

    cv2.imshow('calibration', blank_grid)
    cv2.moveWindow("calibration", 100, 300)
    cv2.waitKey(-1)
    
    circles = [(OFFSET, OFFSET), (FRAME_W/2, OFFSET), (FRAME_W-OFFSET, OFFSET), \
                (OFFSET, FRAME_H/2), (FRAME_W/2, FRAME_H/2), (FRAME_W-OFFSET, FRAME_H/2), \
                (OFFSET, FRAME_H-OFFSET), (FRAME_W/2, FRAME_H-OFFSET), (FRAME_W-OFFSET, FRAME_H-OFFSET)]

    corners = [(OFFSET, OFFSET), (FRAME_W-OFFSET, OFFSET), \
                (OFFSET, FRAME_H-OFFSET), (FRAME_W-OFFSET, FRAME_H-OFFSET)]
    
    # the detected locations of the dots
    trans_corners = []
    
    #the destination points used in the homgraphy
    image_points = []

    #needed to index image_points array down below
    for circle in circles:

        dot_grid = numpy.empty((FRAME_W, FRAME_H, 3))
        dot_grid[:] = ALTERNATE_COLOR
        cv2.circle(dot_grid, circle, CALIBRATE_SIZE, BOARD_COLOR, -1)
        

        #for now, this will be the stable blank image we use
        cv2.imshow('calibration', blank_grid)
        blank_frame = waitForStabilization(device)
        #cv2.waitKey(10)

        cv2.imshow("calibration", dot_grid)
        dot_frame = waitForStabilization(device)
        #cv2.waitKey(100)

        # cv2.imshow("calibration", blank_frame)
        # cv2.waitKey(-1)

        # cv2.imshow("calibration", dot_frame)
        # cv2.waitKey(-1)

        #background subtraction
        diff = blank_frame.astype('float') - dot_frame.astype('float')
        norm_diff = numpy.sqrt(diff*diff).astype('uint8')

        #threshold this image 
        mask = cv2.threshold(norm_diff, 50, 255, cv2.THRESH_BINARY)[1]

        # erode and then dilate
        size = 5
        elt = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
        mask = cv2.erode(mask, elt)
        mask = cv2.dilate(mask, elt)

        # Find the contours in the image
        contours = cv2.findContours(mask, cv2.RETR_CCOMP,
                        cv2.CHAIN_APPROX_SIMPLE)



        #get info from contours
        if contours[0] != []:
            info = cvk2.getcontourinfo(contours[0][0])
        else:
            print "ERROR: DID NOT FIND ANY CALIBRATION DOTS"
            sys.exit(1)

        #add point to array
        image_points.append(info['mean'])

        if (circle in corners):
            trans_corners.append(info['mean'])

    cv2.destroyAllWindows()

    # convert everything to the format needed later on
    circle_floats = [[float(x), float(y)] for x,y in circles]
    trans_corners = [[[float(x), float(y)]] for x,y in trans_corners]

    #now we get a homography using collected points
    homography = cv2.findHomography(numpy.array(image_points), \
        numpy.array(circle_floats))

    print homography[0], numpy.array(trans_corners)

    #return the homography and the boundaries of the board
    return homography[0], numpy.array(trans_corners)


#new implementation using eroding (original implementation commented out
#below. We still need to figure out a way to get a good inital average
#image to do background subtraction
def waitForStabilization(capture):
    ok, frame = capture.read()

    w = frame.shape[1]
    h = frame.shape[0]
    frame_gray = numpy.empty((h, w), 'uint8')

    prev_avg = numpy.empty((h, w), 'float')
    
    while 1:
        new_avg = numpy.zeros((h, w), 'float')

        for i in range(NUM_STABILIZATION_FRAMES):
            ok, frame = capture.read()
            frame_gray = numpy.empty((h, w), 'uint8')
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            new_avg += frame_gray

        new_avg = (new_avg / NUM_STABILIZATION_FRAMES).astype('uint8')

        diff = new_avg.astype(float) - prev_avg.astype(float)
        norm_diff = numpy.sqrt(diff*diff).astype('uint8')

        #threshold this image
        mask = cv2.threshold(norm_diff.astype('uint8'), 20, 255, cv2.THRESH_BINARY)[1]

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

        prev_avg = new_avg


if __name__ == '__main__':
    main()

