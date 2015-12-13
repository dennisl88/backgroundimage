# Determines if an image can be used as a background image
# Image is good if overall colors similar, colors come in large homogenous patches
# Uses k-means to group image by areas of similar color
# Todo: Make script callable by other programs/break up into functions
# Todo: Vectorize functions
# Todo: Multiple random initializations
# Todo: Make code readable

import numpy as np
import scipy
from PIL import Image
import random
import matplotlib.pyplot as plt

# Constants
THUMBNAIL_SIZE = (128, 128)
NUM_CENTROIDS = 5
NUM_ITERS = 10
CHANGE_THRESHOLD = 8

# Opens image, converts to thumbnail to reduce computation
img = Image.open('test2.jpg')
img.thumbnail(THUMBNAIL_SIZE)
img = np.array(img)
h, w = img.shape[0], img.shape[1]

# Random initialization of NUM_CENTROIDS centroids by picking random pixels in image
# centroids - list of RGB centroids
# Todo: weighted random initialization instead of pure random
centroids = np.array([np.copy(img[random.random()*h][random.random()*w])])
for _ in range(NUM_CENTROIDS-1):
    next = np.array([np.copy(img[random.random()*h][random.random()*w])])
    centroids = np.vstack((centroids, next))

# centroid_sum - sum of R, G, B values in for each centroid
# centroid_count - number of pixels associated with each centroid
# soln[h][w][0] - error between this pixel and associated centroid
# soln[h][w][1] - centroid associated with pixel
centroid_sum = np.zeros([NUM_CENTROIDS, 3])
centroid_count = np.zeros(NUM_CENTROIDS)
soln = np.full((h, w, 2), -1)

# Recalculates centroids for NUM_ITERS iterations or until change in 1 cycle is less than CHANGE_THRESHOLD
iters, change = 0, CHANGE_THRESHOLD + 1
while iters < NUM_ITERS and change > CHANGE_THRESHOLD:
    change = 0
    centroid_sum, centroid_count = np.zeros([NUM_CENTROIDS, 3]), np.zeros(NUM_CENTROIDS)
    # Finds closest centroid to each pixel
    for i in range(h):
        for j in range(w):
            for l in range(NUM_CENTROIDS):
                test = np.linalg.norm(img[i][j] - centroids[l])
                if test < soln[i][j][0] or soln[i][j][0] == -1:
                    soln[i][j][0], soln[i][j][1] = test, l
            centroid_sum[soln[i][j][1]] += img[i][j]
            centroid_count[soln[i][j][1]] += 1
    # Calculates new centroids
    for l in range(NUM_CENTROIDS):
        if centroid_count[l] > 0:
            new = centroid_sum[l]/centroid_count[l]
            change += np.linalg.norm(centroids[l]-new)
            centroids[l] = new
    iters+=1

# Outputs image of pixels associated with each centroid
centroid_map = np.zeros([h, w, 3], dtype="uint8")
for i in range(h):
    for j in range(w):
        centroid_map[i][j] = centroids[soln[i][j][1]]
Image.fromarray(centroid_map, 'RGB').save('outfile.png')

# Calculate the center of mass of pixels associated with each centroid
centers_of_mass = np.zeros([NUM_CENTROIDS, 2])
for i in range(h):
    for j in range(w):
        centers_of_mass[soln[i][j][1]][0] += i
        centers_of_mass[soln[i][j][1]][1] += j
for l in range(NUM_CENTROIDS):
    if centroid_count[l] > 0:
        centers_of_mass[l] = centers_of_mass[l]/centroid_count[l]

# Calculates spread of centroids. If spread is too high, reject this picture
spread = 0
for i in range(h):
    for j in range(w):
        cent = soln[i][j][1]
        spread += abs(i - centers_of_mass[cent][0]) + abs(j - centers_of_mass[cent][1])
print(spread)









