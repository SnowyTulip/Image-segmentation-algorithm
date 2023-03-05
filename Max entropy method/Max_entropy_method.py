import numpy as np
import math
import cv2

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

def H_(k,p,A_star_list,Sub_A_star_list):
    '''
    param : "start" and "end" are the boundaries of the summation
            "p" is the histogram of the img_gray
            "A_star_list" is [A*(0),A*(1),A*(2),A*(3),...A*(255)]
    >>> get_entropy(1,p,A_star_list) means H1(k) 
    >>> get_entropy(2,p,Sub_star_list) means H2(k)
        H1(k) H2(K) :Reference [page 6] Equation(20) Equation(21)
    return H_list [H(0),H(1),H(2),...H(255)]
    '''
    bias = 1e-5
    H1 = 0
    if(A_star_list[k] > 0): # do nothing when A*(k) is zero
        for i in range(0,k+1): #sum from 0 to k
            probability =  p[i] / (A_star_list[k])
            if probability > bias:
                H1 += - probability * math.log(probability)
    H2 = 0
    if(Sub_A_star_list[k] > 0): # do nothing when A*(k) is zero
        for j in range(k+1,256):#sum from k+1 to 2^l-1
            probability =  p[j] / (Sub_A_star_list[k])
            if probability > bias:
                H1 += - probability * math.log(probability)
    return H1 + H2
    

def KSW1985(img_gray,l = 8):
    '''
    method: Max entropy method method (Tsai, 1985)
    param : img_gray mat 
            l is the bit of the image 
    return: k(threshold) and binary_image_mat
    '''
    # Reference:paper [1] 
    # Begin  ###################################################
    # Get the histogram of img_gray
    p = get_hist(img_gray,l)
    pixels = get_pixels(img_gray,l)
    # Get A*(k) List of img_gray
    # Then A_star_List means "[A*(0),A*(1),A*(2),A*(3),...A*(255)] "" 
    #  Sub_A_star_List means "[1 - A*(0),1 - A*(1),1 - A*(2),...1 - A*(255)] "" 
    # Reference:Equation(5) paper [1] 
    # A_star_list     = [sum(p[0:k+1]) for k in range(2**l)]
    # A_star_list     = [A_star if A_star <= 1 else 1 for A_star in A_star_list]
    # Sub_A_star_list = [1 - A_star for A_star in A_star_list]
    
    A_star_list     = [sum(pixels[0:k+1])/sum(pixels) for k in range(2**l)]
    Sub_A_star_list = [1 - A_star for A_star in A_star_list]


    # Reference:Equation(20)(21) paper [1] 
    # H1_list means "[H1(0),H1(1),H1(2),H1(3),...H1(255)] "" ,and H2_list too
    # Get the H(k) = H1(k) + H2(k) when it takes the maximum
    H_list = [H_(k,p,A_star_list,Sub_A_star_list) for k in range(2**l)]
    k = np.argmax(H_list)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"Sahli method\SS[9@OG%5WF@9B01N@AX(6P.png"
    img = cv2.imread(path)
    # img = cv2.resize(img,(700,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = KSW1985(gray_img)
    cv2.imshow(f"2:{ret}",b)
    cv2.imshow("o",img)
    cv2.waitKey(0)

'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images

#log: 
1、对A*换算法，使用数点的方式进行计算
@2023.2.12 已解决，使用pixel函数

2、想办法消除bias
'''