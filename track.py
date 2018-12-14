import numpy as np
import argparse
import cv2

'''
frame 		: current frame of the video that we are processing
roi_points	: list of points corresponding to the ROI in our video
input_mode  : Used as a boolean flag, indicating whether or not we are 
			  currentop_lefty selecting the object we want to track in the video.
'''
frame = None
roi_points = []
input_mode = False

def select_roi(event, x, y, flags, param):
	global frame, roi_points, input_mode

	'''
	If we are in selection mode, the mouse was clicked,
	and we do not already have four points, then update
	the list of ROI points with the (x, y) location of
	the click and draw the circle.
	'''
	if input_mode and event == cv2.EVENT_LBUTTONDOWN and len(roi_points) < 4:
		roi_points.append((x, y))
		cv2.circle(frame, (x, y), 4, (0, 255, 0), 2)
		cv2.imshow("frame", frame)

def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("-v", "--video",
		help = "path to the (optional) video file")
	args = vars(ap.parse_args())

	global frame, roi_points, input_mode

	# if video path not given then open the system camera
	if not args.get("video", False):
		camera = cv2.VideoCapture(0)

	else:
		camera = cv2.VideoCapture(args["video"])

	# setup the mouse callback
	cv2.namedWindow("frame")
	cv2.setMouseCallback("frame", select_roi)

	'''
	initialize the termination condition criteria for camshift,
	indicating a maximum of ten iterations or movement by a least 
	one pixel along with the bounding box of the ROI.

	Termination Condition:
	The CamShift algorithm is iterative, meaning that it seeks to optimize the tracking criterion. In this
	case, we’ll set the termination criterion to perform two checks.
	The first check is the epsilon associated with the centroids of our selected ROI and the tracked ROI
	according to the CamShift algorithm. If the tracked centroid has not changed by at least one pixel,
	then terminate the CamShift algorithm.
	The second check controls the number of iterations of CamShift. Using more iterations will allow
	CamShift to (ideally) find a closer centroid match between the selected ROI and the tracked ROI;
	however, this comes at the cost of runtime. If the iterations are set too high, then we will drop below
	real-time performance, which is substantially less than ideal in most situations. Let’s go ahead and
	use a maximum of 10 iterations so we don’t fall into this scenario.
	'''
	termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
	roi_box = None

	# keep looping over the frames
	while True:
		(grabbed, frame) = camera.read()

		if not grabbed:
			break

		if roi_box is not None:
			# convert the current frame to the HSV color space and perform mean shift
			hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
			back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

			'''
			Once we have the back projection, we apply the CamShift algorithm by making a call to
			cv2.CamShift. This function expects three arguments:
				1. back_proj : Which is the output of the histogram back projection.
				2. roi_box : The estimated bounding box containing the object that we want to track.
				3. termination : Our termination criterion which we defined.
			The cv2.CamShift function then returns two values to us. The first contains the estimated position,
			size, and orientation of the object we want to track. We then take this estimation and draw a rotated
			bounding box.
			Finally, the second output of the cv2.CamShift function is the newly estimated position of the ROI,
			which will be re-fed into subsequent calls into the cv2.CamShift function.
			'''
			(r, roi_box) = cv2.CamShift(back_proj, roi_box, termination)
			pts = np.int0(cv2.boxPoints(r))
			cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

		cv2.imshow("frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the 'i' key is pressed then go to ROI selection mode
		if key == ord("i") and len(roi_points) < 4:
			input_mode = True
			orig = frame.copy()

			'''
			keep looping till 4 ROI points have been selected
			press any key to exit selection mode
			'''
			while len(roi_points) < 4:
				cv2.imshow("frame", frame)
				cv2.waitKey(0)

			# determine the top-left and bottom-right points
			roi_points = np.array(roi_points)
			s = roi_points.sum(axis = 1)
			top_left = roi_points[np.argmin(s)]
			bottom_right = roi_points[np.argmax(s)]

			# grab the ROI for the bounding box and convert it
			# to the HSV color space
			roi = orig[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
			#roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

			# compute a HSV histogram for the ROI and store the
			# bounding box
			roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
			roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
			roi_box = (top_left[0], top_left[1], bottom_right[0], bottom_right[1])

		# if the 'q' key is pressed, stop the loop
		elif key == ord("q"):
			break

	# cleanup the camera and close any open windows
	camera.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()