import Max_entropy_method
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os

from sklearn.preprocessing import normalize
 
 
x = np.array([1, 2, 3, 4], dtype='float32').reshape(1,-1)
 
print("Before normalization: ", x)
 
options = ['l1', 'l2', 'max']
for opt in options:
    norm_x = normalize(x, norm=opt)
    print("After %s normalization: " % opt.capitalize(), norm_x)