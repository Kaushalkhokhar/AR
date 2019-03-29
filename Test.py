import numpy as np 
import cv2
import matplotlib.pyplot as plt
import os

data_path = "D:\Programming\Python\AR\Images\Ring with finger"

img = cv2.imread(os.path.join(data_path, 'IMG_5.jpg'))    

MIN_MATCHES = 15
cap = cv2.imread(os.path.join(data_path, 'IMG_5.jpg'))   

# Just to tramsfoem a image
rows,cols,ch = img.shape

pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[150,250]])

M = cv2.getAffineTransform(pts1,pts2)

img = cv2.warpAffine(cap,M,(cols,rows))

model = img
orb = cv2.ORB_create() # ORB keypoint detector
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # create brute force  matcher object
kp_model, des_model = orb.detectAndCompute(model, None)  # Compute model keypoints and its descriptors
kp_frame, des_frame = orb.detectAndCompute(cap, None) # Compute scene keypoints and its descriptors
matches = bf.match(des_model, des_frame) # Match frame descriptors with model descriptors
matches = sorted(matches, key=lambda x: x.distance) # Sort them in the order of their distance


if len(matches) > MIN_MATCHES:
    # draw first 15 matches.
    cap = cv2.drawMatches(model, kp_model, cap, kp_frame,
                          matches[:MIN_MATCHES], 0, flags=2)
    # show result
    cv2.imshow('frame', cap)
    cv2.waitKey(0)
else:
    print ("Not enough matches have been found - {0}/{1}".format(len(matches),MIN_MATCHES))


'''# assuming matches stores the matches found and 
# returned by bf.match(des_model, des_frame)
# differenciate between source points and destination points
src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
# compute Homography
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Draw a rectangle that marks the found model in the frame
print(model.shape)
h, w = model.shape[:-1]
pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
# project corners into frame
dst = cv2.perspectiveTransform(pts, M)  
# connect them with lines
img2 = cv2.polylines(img, [np.int32(dst)], True, 255, 3, cv2.LINE_AA) 
cv2.imshow('frame', cap)
cv2.waitKey(0)'''