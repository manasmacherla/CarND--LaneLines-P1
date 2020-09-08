#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

cap = cv2.VideoCapture('test_videos/solidYellowLeft.mp4')

while(cap.isOpened()):
    ret, image = cap.read()

    #converting to grayscale 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #applying gaussian smoothing
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    #canny edge detection
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

    #hough transform
    left_bottom = [0, 539]
    right_bottom = [959, 539]
    apex = [480, 310]

    # Perform a linear fit (y=Ax+B) to each of the three sides of the triangle
    # np.polyfit returns the coefficients [A, B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, 939), np.arange(0, 539))
    region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                        (YY > (XX*fit_right[0] + fit_right[1])) & \
                        (YY < (XX*fit_bottom[0] + fit_bottom[1]))

    mask = np.zeros_like(edges)   
    ignore_mask_color = 255   

    vertices = np.array([[(left_bottom,apex, right_bottom)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 2 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 15    # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 27 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segments
    line_image = np.copy(image)*0 # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                                min_line_length, max_line_gap)

    # Iterate over the output "lines" and draw lines on a blank image
    negative_slopes, positive_slopes = [], []
    negative_intercepts, positive_intercepts = [], []
    y_maximum = 539
    for line in lines:
        for x1,y1,x2,y2 in line:
            try:
                line_slope = (y2 - y1)/(x2-x1)
            except ZeroDivisionError:
                line_slope = np.inf

            if line_slope < 0.0 and line_slope > -np.inf:
                negative_slopes.append(line_slope)
                negative_intercepts.append(y1 - (line_slope*x1))
                #y_minimum_neg = min(y_minimum_pos, y1, y2)
            if line_slope > 0.0 and line_slope < np.inf:
                positive_slopes.append(line_slope)
                positive_intercepts.append(y1 - (line_slope*x1))
                #y_minimum_pos = min(y_minimum_pos, y1, y2)
    y_minimum_neg = 350
    y_minimum_pos = 350
    if len(negative_slopes) > 0:
        negative_slope_mean = np.mean(negative_slopes)
        negative_intercept_mean = np.mean(negative_intercepts)
        x_minimum = int((y_minimum_neg - negative_intercept_mean)/negative_slope_mean)
        x_maximum = int((y_maximum - negative_intercept_mean)/negative_slope_mean)
        cv2.line(line_image, (x_minimum, y_minimum_neg), (x_maximum, y_maximum),(0,0,255),10)

    if len(positive_slopes) > 0:
        positive_slope_mean = np.mean(positive_slopes)
        positive_intercept_mean = np.mean(positive_intercepts)
        x_minimum = int((y_minimum_pos - positive_intercept_mean)/positive_slope_mean)
        x_maximum = int((y_maximum - positive_intercept_mean)/positive_slope_mean)
        cv2.line(line_image, (x_minimum, y_minimum_pos), (x_maximum, y_maximum),(0,0,255),10)


    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = cv2.addWeighted(image, 0.8, line_image, 1, 0) 

    cv2.imshow('Window',lines_edges)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()