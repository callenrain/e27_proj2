import cv2
import numpy
import sys
import cvk2
# Lab 2: Tic-Tac-Toe
# Callen Rain and Dylan Jeffers
# Computer Vision Spring 2015 
# Lisence: GPLv3

# MAGIC NUMBERS
NUM_STABILIZATION_FRAMES = 5
CALIBRATE_SIZE = 15
GAME_DIM = 10
FRAME_H, FRAME_W = 600,600
FEEDBACK_SIZE = 600
OFFSET = 25
LINE_THICKNESS = 7
BOARD_COLOR = (0, 0, 0)
ALTERNATE_COLOR = (255, 255, 255)

def main():
	#obtains two lists of templates
    X, O = getTemplates()

	#setting up video capture for webcam
    device = 0
    capture = cv2.VideoCapture(device)

    # get a homography and the edges of the board
    M, corners = calibrateCamera(capture)

    # get the box that bounds the board
    box = getRectifiedBox(corners)

    # get the game board with lines drawn
    board = getGameBoard()

    # keeps track of the moves made so far
    entries = [[None, None, None], [None, None, None], [None, None, None]]
	
	#obtains computer-generated board we use to maintain current game state
    feedback_board = getFeedbackBoard()

    # obtain a starting reference image
    for i in range(30):
        cv2.imshow("board", board)
        cv2.moveWindow("board", 100, 300)
        reference = getRectifiedImg(capture, M, box)

    # main game loop
	
    while True:

		# printing the 3 boards (draw board, rectified image, and CG game-state
		# and move them into position
        cv2.imshow("board", board)
        cv2.imshow("feedback_board", feedback_board)
        cv2.moveWindow("feedback_board", 1320, 300)
        current = getRectifiedImg(capture, M, box)
        cv2.moveWindow("camera", 710, 300)
		
		#determine if image is obstructed
        obstruction = isObstructed(current, reference)
		
        if obstruction: printOnImage(current, "OBSTRUCTED")

        cv2.imshow("camera", current)

	    if not obstruction:
			#check to see if move has occured
            index, symbol = checkForMoves(current, reference, X, O)

			#if mark was found
            if index != None:
				#update our reference image
                reference = current
                entries[index/3][index % 3] = symbol
				#update the CG board
                updateFeedbackBoard(feedback_board, (index/3, index % 3), symbol)

				#check for a winner or a stalmate
                isWinner = checkWinner(entries, feedback_board)
                if isWinner != None:
                    winner = 'O' if isWinner == True else 'X'
                    print "winner is", winner

                if checkStalemate(entries):
                    print "stalemate"

                
# this funtion checks the current state of the board and determines if there
# is a winner 
def checkWinner(entries, feedback_board):
  #used for easy computation 
   unit = FRAME_H/3
	#checks rows for winner
    for e, i in enumerate(entries):
        if i[0] == i[1] and i[0] == i[2] and i[0] != None:
            cv2.line(feedback_board, (0, e*unit+unit/2), (unit*3, e*unit+unit/2), (0, 0, 255), (LINE_THICKNESS))
            return i[0]
	# checks columns for winner
    for i in range(3):
        if entries[0][i] == entries[1][i] and entries[0][i] == entries[2][i] and entries[0][i]!=None:
            cv2.line(feedback_board, (i*unit+unit/2, 0), (i*unit+unit/2, unit*3), (0, 0, 255), (LINE_THICKNESS))
            return entries[0][i]
	# checks top left diagonal for winner
    if entries[0][0] == entries[1][1] and entries[0][0] == entries[2][2] and entries[0][0] != None:
        cv2.line(feedback_board, (0, 0), (unit*3, unit*3), (0, 0, 255), (LINE_THICKNESS))
        return entries[0][0]
	# checks top right diagonal for winner
    if entries[0][2] == entries[1][1] and entries[0][2] == entries[2][0] != None:
        cv2.line(feedback_board, (unit*3, 0), (0, unit*3), (0, 0, 255), (LINE_THICKNESS))
        return entries[0][2]  

    return None     

# checks to see if all boxes are filled, which indicates a stalemate
def checkStalemate(entries):
    for i in entries:
        for j in i:
            if j == None:
                return False
    return True

# checks the current frame and determines if an X or O was drawn. 
def checkForMoves(current, reference, X, O):
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

    # take difference of images
    diff = current.astype('float') - reference.astype('float')
    
    # take the absolute value of the difference and use a high threshold
    norm_diff = numpy.sqrt(diff*diff).astype('uint8')
    mask = cv2.threshold(norm_diff.astype('uint8'), 20, 255, cv2.THRESH_BINARY)[1].astype('uint8')


    cv2.line(mask, (FRAME_W/3, 0), (FRAME_W/3, FRAME_H), BOARD_COLOR, (LINE_THICKNESS*5))
    cv2.line(mask, (2*FRAME_W/3, 0), (2*FRAME_W/3, FRAME_H), BOARD_COLOR, (LINE_THICKNESS*5))
    cv2.line(mask, (0, FRAME_H/3), (FRAME_W, FRAME_H/3), BOARD_COLOR, (LINE_THICKNESS*5))
    cv2.line(mask, (0, 2*FRAME_H/3), (FRAME_W, 2*FRAME_H/3), BOARD_COLOR, (LINE_THICKNESS*5))

    cv2.imshow("mask", mask)
    cv2.moveWindow("mask", 1200, 1400)

    quadrant_corners = [(0, 0), (FRAME_W/3, 0), (2*FRAME_W/3, 0), \
                (0, FRAME_H/3), (FRAME_W/3, FRAME_H/3), (2*FRAME_W/3, FRAME_H/3), \
                (0, 2*FRAME_H/3), (FRAME_W/3, 2*FRAME_H/3), (2*FRAME_W/3, 2*FRAME_H/3)]

    max_val = 0
    arg_max = None
    index = 0
    for i, corner in enumerate(quadrant_corners):
        x, y = corner
        quadrant = mask[int(y):int(y+FRAME_H/3), int(x):int(x+FRAME_W/3)]
        if numpy.sum(quadrant) > max_val:
            max_val = numpy.sum(quadrant)
            arg_max = quadrant
            index = i

    if arg_max == None:
        return None, None

    x_max = 0
    for x in X:
        result = numpy.max(cv2.matchTemplate(arg_max, x, cv2.TM_CCORR_NORMED))
        x_max = max(x_max, result)

    o_max = 0
    for o in O:
        result = numpy.max(cv2.matchTemplate(arg_max, o, cv2.TM_CCORR_NORMED))
        o_max = max(o_max, result)

    difference = o_max - x_max

    if difference > .2:
        #found an O
        print "found a 0"
        return index, True
    elif difference < -.2:
        #found an X
        print "found a X"
        return index, False
    else: 
        #found nothing
        return None, None



# generate the templates for tic tac toe
def getTemplates():
    X, O = [], []
    for size in [.9, .8, .7]:
        for thickness in [4, 2, 1]:
            width = int(size*FRAME_H/3)

            template = numpy.zeros((width, width), dtype='uint8')

            #generate an X
            cv2.line(template, (0, 0), (width, width), ALTERNATE_COLOR, thickness)
            cv2.line(template, (0, width), (width, 0), ALTERNATE_COLOR, thickness)
            X.append(template)

            template = numpy.zeros((width, width), dtype='uint8')

            #generate an O
            cv2.circle(template, (int(size*FRAME_H/6), int(size*FRAME_H/6)), int(size*FRAME_H/6)-5, ALTERNATE_COLOR, thickness)
            O.append(template)
    return X, O



# determines if there is a large object in the way of the reference image
def isObstructed(current, reference):
    # change to grayscale
    current = cv2.cvtColor(current, cv2.COLOR_RGB2GRAY)
    reference = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)

    # take difference of images
    diff = current.astype('float') - reference.astype('float')
    
    # take the absolute value of the difference and use a high threshold
    norm_diff = numpy.sqrt(diff*diff).astype('uint8')
    mask = cv2.threshold(norm_diff.astype('uint8'), 50, 255, cv2.THRESH_BINARY)[1]

    # erode the iamge so that only large objects remain
    size = 15
    elt = cv2.getStructuringElement(cv2.MORPH_RECT,(size,size))
    mask = cv2.erode(mask, elt)
    #return mask
    #return whether there is an obstruction or not
    return numpy.sum(mask) != 0

# updates the feedback board (the board that prints the current state of the board) after a new move is made.
def updateFeedbackBoard(feedback_board, location, symbol):
    i, j = location

    unit = FEEDBACK_SIZE/3

    if symbol:    
        #generate an O
        cv2.circle(feedback_board, (unit*j+unit/2, unit*i+unit/2), int(unit/2)-15, BOARD_COLOR, LINE_THICKNESS)
    else:
        #generate an X
        cv2.line(feedback_board, (unit*j, unit*i), (unit*j+unit, unit*i+unit), BOARD_COLOR, LINE_THICKNESS)
        cv2.line(feedback_board, (unit*j, unit*i+unit), (unit*j+unit, unit*i), BOARD_COLOR, LINE_THICKNESS)

# initializes the feeback board, the board that shows the current state of the game
def getFeedbackBoard():
    board = numpy.empty((FEEDBACK_SIZE, FEEDBACK_SIZE, 3))
    board[:] = ALTERNATE_COLOR
    cv2.line(board, (FEEDBACK_SIZE/3, 0), (FEEDBACK_SIZE/3, FEEDBACK_SIZE), BOARD_COLOR, (LINE_THICKNESS))
    cv2.line(board, (2*FEEDBACK_SIZE/3, 0), (2*FEEDBACK_SIZE/3, FEEDBACK_SIZE), BOARD_COLOR, (LINE_THICKNESS))
    cv2.line(board, (0, FEEDBACK_SIZE/3), (FEEDBACK_SIZE, FEEDBACK_SIZE/3), BOARD_COLOR, (LINE_THICKNESS))
    cv2.line(board, (0, 2*FEEDBACK_SIZE/3), (FEEDBACK_SIZE, 2*FEEDBACK_SIZE/3), BOARD_COLOR, (LINE_THICKNESS))
    return board

# intializes the main game board that is drawn on
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

# uses a sequence of dots to obtain a homography and the outline of the board
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

    #return the homography and the boundaries of the board
    return homography[0], numpy.array(trans_corners)


# obtains image used to do background subtraction to establish homography
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

        #frames pass criteria
        
        if numpy.sum(mask) == 0:
            #label = "STABLE"
            return frame_gray
        else: 
            label = "UNSTABLE"

        prev_avg = new_avg

#copied from tutorial.py, used to print text on our images
def printOnImage(image, text):
    h = image.shape[0]
    cv2.putText(image, text, (16, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (0,0,0), 3, cv2.CV_AA)

    cv2.putText(image, text, (16, h-16), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (255,255,255), 1, cv2.CV_AA)

if __name__ == '__main__':
    main()

