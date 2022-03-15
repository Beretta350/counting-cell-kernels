# -*- coding: utf-8 -*-
"""
Example of 2D image filtering in the spatial domain (convolutions) and
in the frequency domain (using the Fourier Transform)
"""


import cv2
import numpy as np
from numpy.fft import fft2, ifft2
from scipy.signal import convolve2d
import time

if __name__ == '__main__':


	tipo = 'Lap' # filter type: 'LP' for average, anything else for Laplacian

	#
	# Filter size (ideally an odd number), only used for avrage filtering
	#
	N = 15
	
	#
	#  Read image, convert to gray and resize
	#
	print('Example of image filtering using either average filter or Laplacian')
	print('Filter size: %d x %d' % (N,N))
	x = cv2.cvtColor(cv2.imread('peppers-grey.bmp'),cv2.COLOR_BGR2GRAY)
	x = cv2.resize(x, (256, 256))

	#
	#  Define filter (average) or Laplacian
	#
	if tipo == 'LP':
		w = np.ones((N,N))/N**2
	else:
		w = np.ones((3,3))
		w[1,1] = -8

	#
	#  Direct filter (convolution) - runs 100 times to estimate running time
	# 
	t0 = time.time()
	for i in range(100):
		yc = convolve2d(x, w)
	etime = (time.time() - t0)/100
	print('Time for convolutions: %1.4f s'%etime)

	#
	#  Filter in frequency domain - again run 100 times to estimate running time
	#
	
	t0 = time.time()
	for i in range(100):


		#
		#  Build padded versions of image and maks
		#

		sx = x.shape
		sw = w.shape
		spad = (sx[0]+sw[0]-1, sx[1]+sw[1]-1) 
		
		xe = np.zeros(spad) 
		we = np.zeros(spad)
		
		xe[:sx[0],:sx[1]]=x
		we[:sw[0],:sw[1]]=w
		
		#
		#  Compute 2D FFTs
		#
		
		Xe = fft2(xe)
		We = fft2(we)
		
		#
		#  Direct product in the frequency domain
		#
		
		WX = Xe*We

		#
		#  filtered signal through the inverse 2D FFT
		#
		y = np.abs(ifft2(WX))
	etime = (time.time() - t0)/100
	print('Time for FFTs: %1.4f s'%etime)
	
	#
	#  Shows images 
	# 
	cv2.imshow('Original Image', x)
	cv2.imshow('FFT of the image (log)', np.fft.fftshift(np.log(abs(Xe)+1e-10)/np.log(np.max(np.abs(Xe)))))
	cv2.imshow('FFT of the filter', np.fft.fftshift(abs(We)/np.max(np.abs(We))))
	cv2.imshow('FFT of the product (log)', np.fft.fftshift(np.log(abs(WX)+1e-10)/np.log(np.max(np.abs(WX)))))
	cv2.imshow('Filtered image using FFT', y/255)
	cv2.imshow('Filtered image using convolution', yc/255)
	
	cv2.waitKey()
	cv2.destroyAllWindows()



