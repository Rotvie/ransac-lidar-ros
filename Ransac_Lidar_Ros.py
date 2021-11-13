#!/usr/bin/env python
import rospy
import time
import math
import numpy as np
import matplotlib.pyplot as plt
from sensor_msgs.msg import LaserScan

rospy.init_node('p3dx_lidar_ransac', anonymous=True)
laser_ranges = []
laser_angles = []

bp = [[-0.1, 1.9], [1.3, 2.2], [1.8, 1.8], [3.1, 2.1], [1.75, 1.9], [1.95, 3.0]]

def fit_with_least_squares(X, y):

    b = np.ones((X.shape[0], 1))
    A = np.hstack((X, b))
    theta = np.linalg.lstsq(A, y , rcond=None)[0]
    return theta

def evaluate_model(X, y, theta, inlier_threshold):

    b = np.ones((X.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    A = np.hstack((y, X, b))
    theta = np.insert(theta, 0, -1.)
    
    distances = np.abs(np.sum(A*theta, axis=1)) / np.sqrt(np.sum(np.power(theta[:-1], 2)))
    inliers = distances <= inlier_threshold
    num_inliers = np.count_nonzero(inliers == True)
    
    return num_inliers
    
def ransac(X, y, max_iters=200, samples_to_fit=2, inlier_threshold=0.15, min_inliers=35):  
  
    best_model = None
    best_model_performance = 0
    
    num_samples = X.shape[0]
    
    for i in range(max_iters):
        sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
        model_params = fit_with_least_squares(X[sample], y[sample])
        model_performance = evaluate_model(X, y, model_params, inlier_threshold)
        
        if model_performance < min_inliers:
            continue
        
        if model_performance > best_model_performance:
            best_model = model_params
            best_model_performance = model_performance
    
    return best_model

def scan_callback(msg):

    global laser_ranges
    global laser_angles
    global X
    global y
    global bp

    laser_ranges = msg.ranges   
    
    for i in range(0,len(laser_ranges)):
       laser_angles.append(msg.angle_min+i*msg.angle_increment)

    for j in range(0,len(bp),2):
        X = []
        y = []

        for i in range(0, len(laser_ranges)):
    	    pn_x = laser_ranges[i]*math.cos(laser_angles[i])
    	    pn_y = laser_ranges[i]*math.sin(laser_angles[i])

	    if (math.isinf(pn_x)==False and math.isinf(pn_y)==False and bp[j][0]<pn_x<bp[j+1][0] and bp[j][1]<pn_y<bp[j+1][1]):
            X.append([pn_x])
            y.append([pn_y])

	    else: 
            pass

        X = np.array(X)
        y = np.array(y)

        result = ransac(X, y)

        m = result[0][0]
        b = result[1][0]
        plt.plot(X, y,'bo')
        plt.plot(X, m*X+b,'r', linewidth=4)

    plt.ylim([0, 3.5])
    plt.xlim([0, 3.5])
    plt.grid()
    plt.title("Ransac para Datos del Lidar RHK")
    plt.show()
    

if __name__ == '__main__':

    msg = rospy.wait_for_message("/p3dx/laser/scan",LaserScan,timeout=None)
    scan_callback(msg)

 
