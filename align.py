import cv2
import numpy as np
import pickle


# Read the images to be aligned
imRef =  cv2.imread("p2.png");
im1 =  cv2.imread("b2.jpg");
im2 =  cv2.imread("a2.jpg");

# Convert images to grayscale
imRef_gray = cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY)
im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

# Define the motion model
warp_mode = cv2.MOTION_HOMOGRAPHY

# Define 2x3 or 3x3 matrices and initialize the matrix to identity
warp_matrix1 = np.eye(3, 3, dtype=np.float32)
warp_matrix2 = np.eye(3, 3, dtype=np.float32)


# Specify the number of iterations.
number_of_iterations = 100;

# Specify the threshold of the increment
# in the correlation coefficient between two iterations
termination_eps = 1e-10;

# Define termination criteria
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)

# Run the ECC algorithm. The results are stored in warp_matrix.
cv2.findTransformECC (imRef_gray,im1_gray,warp_matrix1, warp_mode, criteria,None, 5)
print(warp_matrix1)
cv2.findTransformECC (imRef_gray,im2_gray,warp_matrix2, warp_mode, criteria,None, 5)
print(warp_matrix2)

#save warp matrix
with open('warpMatrix.pickle', 'wb') as f:
    pickle.dump([warp_matrix1, warp_matrix2], f)

shift_matrix = np.float32([[0,0,-200],[0,0,-100],[0,0,0]])
warp_matrix1+=np.matmul(warp_matrix1,shift_matrix)
warp_matrix2+=np.matmul(warp_matrix2,shift_matrix)


# Use warpPerspective for Homography 
sz = imRef.shape
im1_aligned = cv2.warpPerspective (im1, warp_matrix1, (sz[1]+400,sz[0]+200), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
im2_aligned = cv2.warpPerspective (im2, warp_matrix2, (sz[1]+400,sz[0]+200), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

imRef = cv2.copyMakeBorder(imRef, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[0,0,0])


# Show final results

im1_aligned = cv2.cvtColor(cv2.cvtColor(im1_aligned,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
im2_aligned = cv2.cvtColor(cv2.cvtColor(im2_aligned,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
imRef = cv2.cvtColor(cv2.cvtColor(imRef,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)



im1_aligned[:, :, 2] = 0
im2_aligned[:, :, 0] = 0
im2_aligned[:, :, 1] = 0


cv2.imshow("combo", cv2.add(im1_aligned,im2_aligned))


#ret,mask = cv2.threshold(im2_aligned,0,255,cv2.THRESH_BINARY)
#background = cv2.bitwise_and(imRef, cv2.bitwise_not(mask))


#cv2.imshow("fitting", cv2.add(imRef,cv2.add(im1_gray,im2_gray)))
cv2.waitKey(0)
 


