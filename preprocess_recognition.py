import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches



x_arr = np.load('x_arr_silindir10cm.npy')
y_arr = np.load('y_arr_silindir10cm.npy')
x2_arr = np.load('x_arr_2_silindir10cm.npy')
y2_arr = np.load('y_arr_2_silindir10cm.npy')


kare_array = np.stack((x_arr, y_arr))
distance_array = np.empty([239, 1])
distance2_array = np.empty([239, 1])
plt.figure()
plt.axis([0, 300, -100, 200])
plt.plot(x_arr, y_arr, '.')
plt.plot(x2_arr, y2_arr, '.')
plt.savefig('kare2_son_original.png')

for j in range(0, 239):

    h = x_arr[j, 0]
    l = y_arr[j, 0]
    if h < 100:
        x_arr[j, 0] = np.nan
        y_arr[j, 0] = np.nan
    a = math.pow(h, 2)
    b = math.pow(l, 2)
    c = a + b
    c = math.sqrt(c)
    if c > 100 and h > 100:
        distance_array[j, 0] = c
    else:
        distance_array[j, 0] = np.nan

#    if c > 200:
#        x_arr[j, 0] = np.nan
#        y_arr[j, 0] = np.nan

    h = x2_arr[j, 0]
    l = y2_arr[j, 0]
    if h < 100:
        x2_arr[j, 0] = np.nan
        y2_arr[j, 0] = np.nan
    x = math.pow(h, 2)
    y = math.pow(l, 2)
    z = x + y
    z = math.sqrt(z)

    if z > 100 and h > 100:
        distance2_array[j, 0] = z
    else:
        distance2_array[j, 0] = np.nan
        x2_arr[j, 0] = np.nan
        y2_arr[j, 0] = np.nan


c_min = np.nanmin(distance_array)
c_min = np.where(distance_array == c_min)
c2_min = np.nanmin(distance2_array)
c2_min = np.where(distance2_array == c2_min)
c_max = np.nanmax(distance_array)
c_max = np.where(distance_array == c_max)

x_min = x_arr[c_min[0], 0]
y_min = y_arr[c_min[0], 0]
x2_min = x2_arr[c2_min[0], 0]
y2_min = y2_arr[c2_min[0], 0]
x_max = x_arr[c_max[0], 0]
y_max = y_arr[c_max[0], 0]
print(c_min)
print(c2_min)
print(x2_min)
print(y2_min)
print(x_min)
print(y_min)

for j in range(0, len(distance_array)):
    a = x_min[0]
    b = y_min[0]
    x = x_arr[j, 0]
    y = y_arr[j, 0]
    a = x - a
    b = y - b
    a = math.pow(a, 2)
    b = math.pow(b, 2)
    dist = math.sqrt(a + b)
    if dist > 65:
        x_arr[j, 0] = np.nan
        y_arr[j, 0] = np.nan

for j in range(0, len(distance2_array)):
    a = x2_min[0]
    b = y2_min[0]
    x = x2_arr[j, 0]
    y = y2_arr[j, 0]
    a = x - a
    b = y - b
    a = math.pow(a, 2)
    b = math.pow(b, 2)
    dist = math.sqrt(a + b)
    if dist > 65:
        x2_arr[j, 0] = np.nan
        y2_arr[j, 0] = np.nan

plt.figure()
plt.axis([100, 300, -100, 100])
plt.plot(x_arr, y_arr, '.')
plt.plot(x2_arr, y2_arr, '.')
plt.savefig('kare2_son_eliminated.png')
x_max_index = np.where(x_arr == x_max)
x_min_index = np.where(x_arr == x_min[0])
x2_min_index = np.where(x2_arr == x2_min[0])

print(x2_min_index[0][0])
print(x_min_index[0][0])
for k in range(0, 239):
    if k < x2_min_index[0][0]:
        x2_arr[k, 0] = np.nan
        y2_arr[k, 0] = np.nan
    if k > x_min_index[0][0]:
        x_arr[k, 0] = np.nan
        y_arr[k, 0] = np.nan

plt.figure()
plt.axis([100, 300, -100, 100])
plt.plot(x_arr, y_arr, lw=5)
plt.plot(x2_arr, y2_arr, lw=5)
plt.savefig('kare2_son_last_2.png')
#plt.show()

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

if distance_error < 200:
    if slope < 75:
        detected_shape = 'Triangle'
    elif slope < 150:
        detected_shape = 'Square'
    else:
        detected_shape = 'Shape not detected!'
elif distance_error < 1000:
    if slope < 75:
        detected_shape = 'Cylinder 5 cm'
    elif slope < 150:
        detected_shape = 'Cylinder 10 cm'
    else:
        detected_shape = 'Shape not detected!'
else:
    detected_shape = 'Shape not detected!'

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
plt.scatter(x_center, y_center, lw=5)
plt.plot(x2_arr, y_mid_fit)
if detected_shape == 'Cylinder 5 cm':
    circle = plt.Circle((x_center, y_center), 25, color='r', fill=False, lw=2)
    ax=plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')
elif detected_shape == 'Cylinder 10 cm':
    circle = plt.Circle((x_center, y_center), 50, color='g', fill=False, lw=2)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')
elif detected_shape == 'Square':
    slope_square = math.atan(f)
    slope_square = math.degrees(slope_square)
    slope3 = slope_square + 90
    slope4 = slope_square - 90
    slope5 = slope_square
    slope3 = math.radians(slope3)
    x_second = x_center + length * math.cos(slope3)
    y_second = y_center + length * math.sin(slope3)
    slope4 = math.radians(slope4)
    x_third = x_center + length * math.cos(slope4)
    y_third = y_center + length * math.sin(slope4)
    slope5 = math.radians(slope5)
    x_fourth = x_center + length * math.cos(slope5)
    y_fourth = y_center + length * math.sin(slope5)
    pt1 = [x_var, y_var]
    pt2 = [x_second, y_second]
    pt3 = [x_third, y_third]
    pt4 = [x_fourth, y_fourth]
    circle = plt.Polygon([pt1, pt2, pt4, pt3], color='b', fill=False, lw=2)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')
elif detected_shape == 'Triangle':
    slope_tri = math.atan(f)
    slope_tri = math.degrees(slope_tri)
    slope3 = slope_tri + 60
    slope4 = slope_tri - 60
    slope3 = math.radians(slope3)
    x_second = x_center + length * math.cos(slope3)
    y_second = y_center + length * math.sin(slope3)
    slope4 = math.radians(slope4)
    x_third = x_center + length * math.cos(slope4)
    y_third = y_center + length * math.sin(slope4)
    pt1 = [x_var, y_var]
    pt2 = [x_second, y_second]
    pt3 = [x_third, y_third]
    circle = plt.Polygon([pt1, pt2, pt3], color='purple', fill=False, lw=2)
    ax = plt.gca()
    ax.add_patch(circle)
    plt.axis('scaled')

plt.show()

#plt.savefig('kare2_detected.png')
# plt.figure()
# plt.imshow()



