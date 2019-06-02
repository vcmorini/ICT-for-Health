import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import time
import math
import csv

# -*- coding: utf-8 -*-
"""
@author: victor
"""

import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os




np.set_printoptions(precision=2)# use only two decimal digits when printing numbers
plt.close('all')# close previously opened pictures
## 4 color levels filein='low_risk_10.jpg';# file to be analyzed
## 4 color levels filein='melanoma_21.jpg';# file to be analyzed
## 6 color levels filein='melanoma_27.jpg';# file to be analyzed
## 4 color levels filein='medium_risk_1.jpg';# file to be analyzed

with open('moles_characteristics.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['file', 'area', 'perimeter', 'ratio'])


    all_paths = []
    all_names = []
    for file in os.listdir(os.path.join(os.getcwd(), 'lab02_resources')):
        if file.endswith(".jpg"):
            all_paths.append(os.path.join(os.getcwd(), 'lab02_resources', file))
            all_names.append(file)

    end_table = []
    for idx, filein in enumerate(all_paths):
        im_or = mpimg.imread(filein)
        # im_or is Ndarray 583 x 584 x 3 unint8
        # plot the image, to check it is correct:
        f_handle = plt.figure()
        f_handle.tight_layout(False)
        plt.imshow(im_or)
        plt.title('original image')
        #plt.draw()
        plt.pause(0.1)  # this makes the plot to occur
        #%% reshape the image from 3D to 2D
        N1, N2, N3 = im_or.shape  # note: N3 is 3, the number of elementary colors, i.e. red, green ,blue
        # im_or(i,j,1) stores the amount of red for the pixel in position i,j
        # im_or(i,j,2) stores the amount of green for the pixel in position i,j
        # im_or(i,j,3) stores the amount of blue for the pixel in position i,j
        # we resize the original image
        im_2D=im_or.reshape((N1*N2,N3))# im_2D has N1*N2 rows and N3 columns
        # pixel in position i.j goes to position k=(i-1)*N2+j)
        # im_2D(k,1) stores the amount of red of pixel k
        # im_2D(k,2) stores the amount of green of pixel k
        # im_2D(k,3) stores the amount of blue of pixel k
        # im_2D is a sequence of colors, that can take 2^24 different values
        Nr, Nc = im_2D.shape
        #%% get a simplified image with only Ncluster colors
        # number of clusters/quantized colors we want to have in the simpified image:
        Ncluster=3
        # instantiate the object K-means:
        kmeans = KMeans(n_clusters=Ncluster, random_state=0)
        # run K-means:
        kmeans.fit(im_2D)
        # get the centroids (i.e. the 3 colors). Note that the centroids
        # take real values, we must convert these values to uint8
        # to properly see the quantized image
        kmeans_centroids = kmeans.cluster_centers_.astype('uint8')  # Coordinates of cluster centers.
        # copy im_2D into im_2D_quant
        im_2D_quant=im_2D.copy()
        for kc in range(Ncluster):
            quant_color_kc = kmeans_centroids[kc, :]
            # kmeans.labels_ stores the cluster index for each of the Nr pixels
            # find the indexes of the pixels that belong to cluster kc
            ind = (kmeans.labels_ == kc)  # labels_: Labels of each point
            # set the quantized color to these pixels
            im_2D_quant[ind, :] = quant_color_kc
        im_quant = im_2D_quant.reshape((N1,N2,N3))
        fig_handle = plt.figure()
        fig_handle.tight_layout(False)
        plt.imshow(im_quant,interpolation=None)
        plt.title('image with quantized colors')
        #plt.draw()
        plt.pause(0.1)


        #%% Find the centroid of the main mole

        #%% Preliminary steps to find the contour after the clustering
        #
        # 1: find the darkest color found by k-means, since the darkest color
        # corresponds to the mole:
        centroids = kmeans_centroids
        sc = np.sum(centroids, axis=1)  # sum in the column direction
        i_col = sc.argmin()  # index of the cluster that corresponds to the darkest color

        # 2: define the 2D-array where in position i,j you have the number of
        # the cluster pixel i,j belongs to
        im_clust = kmeans.labels_.reshape(N1, N2)  # the matrix now is 2D
        plt.matshow(im_clust, cmap='binary')
        plt.title('image separated by clusters')
        plt.show()
        # 3: find the positions i,j where im_clust(all indexes of the img) is equal to i_col(the index of the darkest cluster center)
        # the 2D Ndarray zpos stores the coordinates i,j only of the pixels
        # in cluster i_col
        zpos = np.argwhere(im_clust == i_col)  # zpos contains all locations of the darkest cluster


        # 4: ask the user to write the number of objects belonging to
        # cluster i_col in the image with quantized colors

        N_spots_str = input("How many distinct dark spots can you see in the image? ")
        N_spots = int(N_spots_str)
        # N_spots = 2

        # 5: find the center of the mole
        # this if else is just finding the nearest cluster center to the center of the image
        if N_spots == 1:
            center_mole = np.median(zpos, axis=0).astype(int)  # this works because the numbers are in order(up-down or down-up)
        else:
            # use K-means to get the N_spots clusters of zpos
            kmeans2 = KMeans(n_clusters=N_spots, random_state=0)
            kmeans2.fit(zpos)
            centers = kmeans2.cluster_centers_.astype(int)
            # the mole is in the middle of the picture:
            center_image = np.array([N1//2, N2//2])
            center_image.shape = (1, 2)
            d = np.zeros((N_spots, 1))
            for k in range(N_spots):
                d[k] = np.linalg.norm(center_image-centers[k,:])
            center_mole = centers[d.argmin(), :]


        # 6: take a subset of the image that includes the mole
        c0 = center_mole[0]
        c1 = center_mole[1]
        RR, CC = im_clust.shape  # im_clust is N1xN2 image, each of its field belongs to a cluster, 1,2 or 3.
        stepmax = min([c0, RR-c0, c1, CC-c1])
        cond = True
        area_old = 0
        surf_old = 1
        step = 10  # each time the algorithm increases the area by 2*step pixels
        # horizontally and vertically
        im_sel = (im_clust == i_col)  # im_sel is a boolean NDarray with N1 rows and N2 columns.. N1xN2 matrix, with 1s in the pixels belonging to the darkest cluster, and zero otherwise
        im_sel = im_sel*1  # im_sel is now an integer NDarray with N1 rows and N2 columns

        while cond:
            subset = im_sel[c0-step:c0+step+1, c1-step:c1+step+1]
            area = np.sum(subset)  # sum all matrix. every 1, correspond to a pixel of the darkest cluster
            Delta = np.size(subset)-surf_old  # np.size(subset) counts the total amount of pixels
            surf_old = np.size(subset)
            if area > area_old+0.01*Delta:  # here, it asks to increase the area if the length increases. if not, it is not a mole anymore
                step = step+10
                area_old = area
                cond = True
                if step > stepmax:
                    cond = False
            else:
                cond = False
                # subset is the search area
        plt.matshow(subset, cmap='binary')
        plt.show()

        img = np.zeros([subset.shape[0], subset.shape[1], 3], dtype=np.int8)
        img[:, :, 0] = np.multiply(subset, 255)
        img[:, :, 1] = np.multiply(subset, 255)
        img[:, :, 2] = np.multiply(subset, 255)
        img = img.astype(np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Binary', img)
        kernel = np.ones((math.floor(subset.shape[0]*0.05), math.floor(subset.shape[0]*0.05)), np.uint8)  # structuring element also called kernel
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        cv2.imshow('Opening', opening)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        cv2.imshow('Closing', closing)
        laplacian = cv2.Laplacian(closing, cv2.CV_8U)
        borders_index = np.argwhere(laplacian == np.max(laplacian))

        # cv2.waitKey()
        dict = {}
        for i in range(len(borders_index)):
            dict[borders_index[i,0]] = {'min':0, 'max':0, 'array': []}

        for i in range(len(borders_index)):
            dict[borders_index[i,0]]['array'].append(borders_index[i,1])
            dict[borders_index[i,0]]['min'] = np.min(dict[borders_index[i,0]]['array'])
            dict[borders_index[i,0]]['max'] = np.max(dict[borders_index[i,0]]['array'])

        for key, values in dict.items():
            dict[key]['min'] = np.min(dict[key]['array'])
            dict[key]['max'] = np.max(dict[key]['array'])

        location_list = []
        for key, values in dict.items():
            laplacian[key, dict[key]['min']:dict[key]['max']] = 255
        print('')
        cv2.imshow('Solid Image', laplacian)
        border = cv2.Laplacian(laplacian, cv2.CV_8U)
        cv2.imshow('Borders', border)

        area = np.sum(laplacian/255)
        perimeter = np.sum(border/255)
        radius = math.sqrt(area/math.pi)
        circle_perimeter = 2*math.pi*radius
        ratio = perimeter/circle_perimeter
        print('area: {}\nperimeter: {}\nratio: {}'.format(area, perimeter, ratio))
        end_table.append([all_names[idx], area, perimeter, ratio])
        writer.writerow([all_names[idx], area, perimeter, ratio])
        plt.matshow(laplacian, cmap='binary')
        plt.matshow(border, cmap='binary')

# cv2.waitKey()
csv_file.close()