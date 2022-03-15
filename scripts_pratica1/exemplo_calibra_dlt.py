# -*- coding: utf-8 -*-
"""
Exemplo simples de calibração completa usando o método DLT usando os vértices de
um cubo como padrao de calibracao
"""

import numpy as np
from scipy.io import loadmat
from math import cos, sin
import cv2
from scipy.linalg import rq


def extract_intrinsic_extrinsic(Pe):
	#
	# Extracts  intrinsics matrix K (3 x 3) e extrinsic parameters R (3 x
	# 3) and T (3 x 1) from a camera matrix (3 x 4)
	# Pe
	#
	
	#
	#  Gets first 3 x 3 block from Pe and computes RQ decomposition
	#
	
	Ke, Re = rq(Pe[0:3,0:3])
	
	#
	# Adjusts matrix Ke such that diagonal is positive
	#
	D = np.sign(Ke.diagonal())
	S = np.diag(D)
	

	Ke = Ke@S;
	Re = S@Re;
	
	#
	#  Gets estimated translation Te
	#
	Te = np.linalg.inv(Ke) @ Pe[:,3]	
	Te = Te.reshape(3,1)
	Ke=Ke/Ke[2,2]; # normalizes K so that last diagonal element equals 1
	
	
	return Ke, Re, Te


def calibra_camera_dlt(xc, xw):
	#
	# Estimates camera matrix Pe based on correspondences between  2D camera points xc 
	# and 3D world points wx. cam is a 2 x n matrix, wc is a 3 x n matrix (n >= 6)
	#
	n = xc.shape[1] # number of correspondences
	A = np.zeros((2*n, 12)) #matrix with correspondences
	for i in range(n):
		#
		# builds matrix A
		#
		A[2*i,:] = np.array( [xw[0,i], xw[1,i], xw[2,i], 1, 0, 0, 0, 0, -xc[0,i]*xw[0,i], -xc[0,i]*xw[1,i], -xc[0,i]*xw[2,i], -xc[0,i]]  )
		A[2*i + 1,:] = np.array( [0, 0, 0, 0, xw[0,i], xw[1,i], xw[2,i], 1, -xc[1,i]*xw[0,i], -xc[1,i]*xw[1,i], -xc[1,i]*xw[2,i], -xc[1,i]] )
		
	#
	#  Computes  SVD to find solution of homogeneous system
	#
	U,S,V = np.linalg.svd(A) # note that V is transposed
	
	# Reshapes least singular vector to matrix form
	Pe = np.reshape(V[-1,:], (3,4))
	return Pe


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
	out = dict()
	out['y1'] = project_polygon(data['x1'],P);
	out['y2'] = project_polygon(data['x2'],P);
	out['y3'] = project_polygon(data['x3'],P);
	out['y4'] = project_polygon(data['x4'],P);
	out['y5'] = project_polygon(data['x5'],P);
	out['y6'] = project_polygon(data['x6'],P);
	
	#
	# Creates synthetic image
	#
	
	img = 255*np.ones((480, 640, 3)).astype('uint8')
	
	#
	#  Plots each face into image
	#
	
	
	plot_face(img, out['y1'], (0,0,255));
	plot_face(img, out['y2'], (255,0,255));
	plot_face(img, out['y3'], (255,0,0));
	plot_face(img, out['y4'], (0,255,0));
	plot_face(img, out['y5'], (0,255,255));
	plot_face(img, out['y6'], (255,255,0));
	
	return img, out
	

if __name__ == '__main__':
	#
	# Loads vertices of each face in a cube with side 20, centered at (0,0,110). 
	#
	data = loadmat('pontos_cubo.mat')
	#
	#  Defines virtual camera
	#


	# rotation angles (in deg)
	ax = deg2rad(10); 
	ay = deg2rad(15);
	az = deg2rad(-30); 

	# focal distance 
	f = 500; 
	
	# pricipal point
	uc = 320; vc = 240; 
	
	# camera center in the WCS
	x0 = np.array( [[5, 0, 0]]).T; 
	
	# builds rotation matrices
	Rx = np.array([[ 1, 0, 0] , [0, cos(ax), -sin(ax)], [0, sin(ax), cos(ax)]]);
	Ry = np.array([ [cos(ay), 0, -sin(ay)], [0, 1,  0], [sin(ay), 0, cos(ay)] ]);
	Rz = np.array( [ [cos(az), -sin(az), 0], [sin(az), cos(az), 0], [0, 0, 1]]);
	R = Rx@Ry@Rz;

	# intrinsics matrix
	K = np.array( [	[f, 0, uc], [0, f, vc], [0, 0, 1]] );
	T = -R@x0;
	
	# Builds 3 x 4 projection matrix
	P0 = K@np.hstack([R,T]);
	img, out = plot_cube(P0, data)
	
	#
	#  Builds list of corresponce points 
	#
	dic1_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
	dic2_names = ['y1', 'y2', 'y3', 'y4', 'y5', 'y6']
	world_pts = np.empty((3,0))
	camera_pts = np.empty((2,0))
	for i in range(len(dic1_names)):
		world_pts = np.hstack((world_pts, data[dic1_names[i]]))
		camera_pts = np.hstack((camera_pts, out[dic2_names[i]]))
	
	
	#
	#  Adds noise to camera points
	#
	noise_level = 0.1
	camera_pts = camera_pts + np.random.normal(0, noise_level, camera_pts.shape)
	
	#
	#  Estimates camera matrix P based on correspondences (DLT)
	# 
	Pe = calibra_camera_dlt(camera_pts, world_pts) # note that Pe is invariant to scaling
	
	#
	#  Extracts intrinsics and extrinsics from Pe based on RQ decomposition
	#
	Ke, Re, Te = extract_intrinsic_extrinsic(Pe)
	
	#
	#  Note that P and -P generate the same projectio matrix. 
	#  Check if P0 and Pe have the "same sign", and possiblity adjust
	#  the sign of Re and Te
	#
	if np.sign(P0[1,1]) != np.sign(Pe[1,1]):
		Re = -Re
		Te = -Te

	
	#
	#  Shows results
	#
	print('Original matrices (K,R,T):')
	print(K,'\n')
	print(R,'\n')
	print(T,'\n')

	print('Estimated matrices (Ke,Re,Te):')
	print(Ke,'\n')
	print(Re,'\n')
	print(Te,'\n')
	
	img, pts = plot_cube(P0, data)
	img_est, pts_est = plot_cube(Pe, data)
	cv2.imshow('Projected GT cube', img)
	cv2.imshow('Projected estimated cube', img_est)
	cv2.waitKey()
	cv2.destroyAllWindows()
	
	
	
