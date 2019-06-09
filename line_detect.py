# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 00:20:56 2019

@author: Ferdi
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

x_arr = np.load('x_arr_son_kare.npy')
y_arr = np.load('y_arr_son_kare.npy')
x2_arr = np.load('x2_arr_son_kare.npy')
y2_arr = np.load('y2_arr_son_kare.npy')

x_arr[239, 0] = np.nan
y_arr[239, 0] = np.nan
x2_arr[239, 0] = np.nan
y2_arr[239, 0] = np.nan

x_arr = x_arr[~np.isnan(x_arr)]
y_arr = y_arr[~np.isnan(y_arr)]
x2_arr = x2_arr[~np.isnan(x2_arr)]
y2_arr = y2_arr[~np.isnan(y2_arr)]


def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X)  # or len(Y)

    numer = sum([xi*yi for xi, yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b



a, b = best_fit(x_arr, y_arr)

slope1 = math.atan(b)
slope1 = math.degrees(slope1)
#slope2 = math.fabs(slope1)

# best fit line:
# y = a + b.x

c, d = best_fit(x2_arr, y2_arr)
slope2 = math.atan(d)
slope2 = math.degrees(slope2)
#slope2 = math.fabs(slope2)

slope = math.fabs(slope1) + math.fabs(slope2)
print('Angle between lines:', slope)
slope_middle = (slope2 + slope1)/2
slope_middle = math.radians(slope_middle)
f = math.tan(slope_middle)


x_intercept = (c-a)/(b-d)
y_intercept = a+b*x_intercept
e = y_intercept - f * x_intercept


# plot points and fit line
plt.figure()
plt.axis([100, 250, -75, 75])
plt.scatter(x_arr, y_arr)
plt.scatter(x2_arr, y2_arr)
yfit = [a + b * xi for xi in x_arr]
y_mid_fit = [e + f * xi for xi in x2_arr]
y2fit = [c + d * xi for xi in x2_arr]
d1 = 0
d2 = 0
for j in range(len(x_arr)):
    p1 = (x_arr[0], yfit[0])
    p2 = (x_arr[len(x_arr)-1], yfit[len(x_arr)-1])
    p3 = (x_arr[j], y_arr[j])
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    d1 = d1 + np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
for j in range(len(x2_arr)):
    p1 = (x2_arr[0], y2fit[0])
    p2 = (x2_arr[len(x2_arr)-1], y2fit[len(x2_arr)-1])
    p3 = (x2_arr[j], y2_arr[j])
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    d2 = d2 + np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1)
distance_error = d1 + d2
print('Deviation from the line:', distance_error)

if distance_error < 150:
    if slope < 75:
        detected_shape = 'Triangle'
    else:
        detected_shape = 'Square'
else:
    if slope < 75:
        detected_shape = 'Cylinder 5 cm'
    else:
        detected_shape = 'Cylinder 10 cm'

print('Detected Shape:', detected_shape)

if detected_shape == 'Triangle':
    length = 80/math.sqrt(3)
    x_var = x_intercept
    y_var = y_intercept
if detected_shape == 'Square':
    length = 70/math.sqrt(2)
    x_var = x_intercept
    y_var = y_intercept
if detected_shape == 'Cylinder 5 cm':
    length = 25
    x_var = np.nanmin(x2_arr)
    y_var = y_mid_fit[0]
if detected_shape == 'Cylinder 10 cm':
    length = 50
    x_var = np.nanmin(x2_arr)
    y_var = y_mid_fit[0]

coeff0 = math.pow(x_var, 2) + math.pow(e, 2) - 2*e*y_var + math.pow(y_var,2) - math.pow(length, 2)
coeff1 = -2*x_var + 2*e*f - 2*f*y_var
coeff2 = 1 + math.pow(f, 2)
coeff = [coeff2, coeff1, coeff0]
roots = np.roots(coeff)
x_center = max(roots[0], roots[1])
y_center = e + f*x_center



plt.plot(x_arr, yfit)
plt.plot(x2_arr, y2fit)
plt.text(x_center-15, y_center-10, 'center of gravity')
plt.text(x_center-15, y_center-30, detected_shape)
plt.text(x_center-15, y_center-23, 'Detected Shape:')
plt.scatter(x_center,y_center, lw=5)
plt.plot(x2_arr, y_mid_fit)
plt.show()
#plt.savefig('kare2_detected.png')
cv2.waitKey(0)
cv2.destroyAllWindows()
# plt.figure()
# plt.imshow()
'''
plt.plot(x_arr, yfit - np.std(yfit), c='y')
plt.plot(x2_arr, y2fit - np.std(y2fit), c='y')
plt.plot(x_arr, yfit + np.std(yfit), c='y')
plt.plot(x2_arr, y2fit + np.std(y2fit), c='y')
'''