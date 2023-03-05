import numpy as np
from tqdm import tqdm 
import math
import cv2
#使用原文中简化的公式进行
#Kittler_1986_Minimum error thresholding 公式(15)
#code
def get_hist(img_gray,l = 8):
    '''
    param : img_gray is a mat
            l is the bit of the image
    return : histogram of img_gray
    '''
    hist, bins = np.histogram(img_gray.ravel(), bins=2**l, range=(0, 2**l))
    p = hist/sum(hist)
    return p

def get_pixels(img_gray,l = 8):
    '''
    param : img_gray is a mat
            l is the bit of the image
    return : number for each gray
    '''
    hist, bins = np.histogram(img_gray.ravel(), bins=2**l, range=(0, 2**l))
    pixel = hist
    return pixel

def GetValues(p,A_star_list,Sub_A_star_list,l = 8):
    '''
    param:   A_star_list[k] is A*(k) 
        also Sub_A_star_list[k] is 1 - A*(k)
        p is histgram of img_gray
    return g1_list,g2_list,std_dev1_list,std_dev2_list
    '''
    g1_list       = [0 for k in range(2**l)]
    g2_list       = [0 for k in range(2**l)]
    std_dev1_list = [0 for k in range(2**l)]
    std_dev2_list = [0 for k in range(2**l)]
    for k in range(0,2**l):
        acc1 = 0
        if A_star_list[k] != 0:
            for g1 in range(k):
                acc1 += g1 * p[g1]/A_star_list[k] 
        g1_list[k] = acc1

        acc2 = 0
        if Sub_A_star_list[k] != 0:
            for g2 in range(k,2**l):
                acc2 += g2 * p[g2]/Sub_A_star_list[k]  
        g2_list[k] = acc2
        ### reference
        acc3 = 0
        if A_star_list[k] != 0:
            for g in range(k):
                acc3 += (g - g1_list[k])**2 * p[g]/A_star_list[k] 
        std_dev1_list[k] = acc3
        acc4 = 0
        if Sub_A_star_list[k] != 0:
            for g in range(k,2**l):
                acc4 += (g - g2_list[k])**2 * p[g]/Sub_A_star_list[k] 
        std_dev2_list[k] = acc4
        
    return std_dev1_list,std_dev2_list


def Get_G(u):
    '''
    '''
    if u == 1:
        res = 0
    else:
        res = -u * math.log(u) - (1 - u) * math.log(1 - u)
    return res

def Get_J_List(std_dev1_list,std_dev2_list,A_star_list,Sub_A_star_list,l):
    '''
    Reference paper[1] 30
    '''
    J_list = [0 for _ in range(2**l)]
    for k in range(2**l):
        J_list[k] = J(A_star_list[k],Sub_A_star_list[k],std_dev1_list[k],std_dev2_list[k])
    return J_list

def J(A_star,Sub_A_star,std_dev1,std_dev2):
    '''
    Get function J
    '''
    '''When the variance is zero, it will lead to the following log operation error.
    However, we notice that there is no optimal segmentation threshold above [0, k] 
    because the variance of gray level from zero to k is zero on the histogram 
    and the value of each point is equal to its average value.
    '''
    if std_dev1 == 0 or std_dev2 == 0:
        res = 1e10
    else:
        item1_2 = 0 if A_star == 0 else A_star * math.log(A_star) 
        item2_2 = 0 if A_star == 0 else Sub_A_star * math.log(Sub_A_star)
        
        res = 1 + 2 * (A_star * math.log(std_dev1 ** 0.5) + Sub_A_star * math.log(std_dev2 ** 0.5))\
                - 2 * (item1_2                            + item2_2)
    return res




def KI1986(img_gray,l = 8):
    '''
    method: Minimum_error_method0
    param : img_gray mat 
            l is the bit of the image 
    return: k(threshold) and binary_image_mat
    '''
    # Reference:paper [1] 
    # Begin  ###################################################
    # Get the histogram of img_gray
    p = get_hist(img_gray,l)
    pixels = get_pixels(img_gray,l)
    A_star_list     = [sum(pixels[0:k+1])/sum(pixels) for k in range(2**l)]

    Sub_A_star_list = [1 - A_star for A_star in A_star_list]
    
    std_dev1_list,std_dev2_list  = GetValues(p,A_star_list,Sub_A_star_list,l)
    J_list = Get_J_List(std_dev1_list,std_dev2_list,A_star_list,Sub_A_star_list,l)
    k = np.argmin(J_list)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"Sahli method\SS[9@OG%5WF@9B01N@AX(6P.png"
    img = cv2.imread(path)
    # img = cv2.resize(img,(400,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = KI1986(gray_img)
    cv2.imshow(f"2:{ret}",b)
    print(f"thresh:{ret}")
    cv2.imshow("o",img)
    cv2.waitKey(0)

'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images

#log: 

'''