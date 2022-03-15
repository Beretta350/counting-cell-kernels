# -*- coding: utf-8 -*-
"""
Simple example of virtual camera rendering of a cube
@author: crjung
"""

import numpy as np
from scipy.io import loadmat
from math import cos, sin
import cv2

# Intitial angular values and focal length
ax = 0
ay = 0
az = 0
f = 200

def deg2rad(rad):
	#
	#  Converts from deg to rad
	#
	return rad*np.pi/180

def project_polygon(pts, P):
	#
	#  Given a set of points pts (3 x n matrix) and 3 x 4 camera matrix, generates
	#  a 3 x n matrix with projected points
	#
	
	# Creates homogeneous coordinates
	n = pts.shape[1]
	pts2 = np.vstack([pts, np.ones(n)])
	
	# Projection (homogeneous coordinates)
	y = P@pts2
	
	# Transform to Euclidian coordinates
	y = y[0:2]/y[2,:]
	
	return y
	
def plot_face(img, pts, color):
	#
	#  plots list of points in pts onto image img
	#
	pts = np.int0(np.round(pts))
	cv2.polylines(img, [pts.T],True, color, thickness = 2)
	

def plot_cube(P, data):
	#
	#  Projects each face of the cube
	#
	y1 = project_polygon(data['x1'],P);
	y2 = project_polygon(data['x2'],P);
	y3 = project_polygon(data['x3'],P);
	y4 = project_polygon(data['x4'],P);
	y5 = project_polygon(data['x5'],P);
	y6 = project_polygon(data['x6'],P);
	
	#
	# Creates synthetic image
	#
	
	img = 255*np.ones((480, 640, 3)).astype('uint8')
	
	#
	#  Plots each face into image
	#
	plot_face(img, y1, (0,0,255));
	plot_face(img, y2, (255,0,255));
	plot_face(img, y3, (255,0,0));
	plot_face(img, y4, (0,255,0));
	plot_face(img, y5, (0,255,255));
	plot_face(img, y6, (255,255,0));
	
	return img
	

#
#  Functions for updating camera parameters based on trackbar
#	
def change_f(val):
	global f
	f = val*10
	fn_callback()


def change_az(val):
    global az	
    az = (val - 180)*np.pi/180
    print(az)
    fn_callback()


def change_ax(val):
	global ax
	ax = (val - 180)*np.pi/180
	fn_callback()


def change_ay(val):
    global ay
    ay = (val - 180)*np.pi/180
    fn_callback()


#
#	Trackbar callback function
#	
def fn_callback():

#	cv2.namedWindow(title_window)
	
	#
	#  Defines all remaining camera parameters (that are not controlled by trackbars)
	#
	
	# pricipal point
	uc = 320; vc = 240;  
	
	# camera center in the WCS
	x0 = np.array( [[0, 0, 0]]).T; 
	
	# builds rotation matrices
	Rx = np.array([[ 1, 0, 0] , [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]]);
	Ry = np.array([ [cos(ay), 0, -sin(ay)], [0, 1,  0], [sin(ay), 0, cos(ay)] ]);
	Rz = np.array( [ [cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]]);
	R = Rx@Ry@Rz; # matrix products

	# intrinsics matrix
	K = np.array( [	[f, 0, uc], [0, f, vc], [0, 0, 1]] );
	T = -R@x0;
	
	# Builds 3 x 4 projection matrix
	P = K@np.hstack([R,T]);
	
	# Projects cube to image
	img = plot_cube(P, data) # plots cube
	
	cv2.putText(img, 'ax = %d, ay = %d, az = %d, f = %d' % (int(180*ax/np.pi), int(180*ay/np.pi), int(180*az/np.pi), f), (0,20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0,0,0)) 
	# Shows image
	cv2.imshow(title_window, img)	
	
	cv2.waitKey()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	#
	# Loads vertices of each face in a cube with side 20, centered at (0,0,110). 
	#
	global data, title_window
	data = loadmat('pontos_cubo.mat')

	#
	# Shows projected image
	#
	title_window = 'Projected cube' 
	cv2.namedWindow(title_window)
	cv2.createTrackbar('f (x 10)', title_window , 1, 100, change_f)
	cv2.createTrackbar('ax', title_window , -180, 360, change_ax)
	cv2.createTrackbar('ay', title_window ,  -180, 360, change_ay)
	cv2.createTrackbar('az', title_window ,  -180, 360, change_az)

	fn_callback()
	
	# pricipal point
	uc = 320; vc = 240; 
	
	# camera center in the WCS
	x0 = np.array( [[0, 0, 0]]).T; 
	
