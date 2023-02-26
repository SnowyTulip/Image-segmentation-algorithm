import numpy as np
import math
import cv2

#code
#public function:
def get_hist(img_gray,l = 8):
    '''
    param : img_gray is a mat
            l is the bits of the image
    return : histogram of img_gray
    '''
    hist, bins = np.histogram(img_gray.ravel(), bins=2**l, range=(0, 2**l))
    p = hist/sum(hist)
    return p


def get_mi(img_gray,i:int,l = 8):
    '''
    Get  i order Moments of img_gray
    Reference formula (2) [page 2]
    '''
    ...
    #get the histogram of img_gray
    p = get_hist(img_gray,l)
    #calculate the mi of img_gray
    #Reference equation(2) paper [1]
    mi = 0
    for g in range(2**l):
        mi += g**i * p[g]
    return mi
#end ##########################

def Tsai1985(img_gray,l = 8):
    '''
    method: Moment-preserving method (Tsai, 1985)
    param : img_gray mat
    return: k(threshold) and binary_image_mat
    '''
    #Description:
    #Alogrithm : Tsai [2]
    #The following code are referenced from "Tsai_1985_Moment-preserving thresholding_A new approach"
    #It is to find the solution of equations (12), (13) and (14) in paper [1]
    #Begin  ###################################################
    #Get the i th moment of img_gray (i = 0,1,2,3)
    m0 = get_mi(img_gray,0,l) 
    m1 = get_mi(img_gray,1,l) 
    m2 = get_mi(img_gray,2,l) 
    m3 = get_mi(img_gray,3,l) 
    #Equation (i)   in Tsai [1]
    cd = m0 * m2 - m1 * m1
    c0 = 1/cd *( -m2 * m2 - (-m3 * m1) )
    c1 = 1/cd *( -m0 * m3 - (-m2 * m1) )

    #Equation (ii) [page 13] in Tsai [1]
    z0 = (1/2)*( -c1 - (c1**2 - 4 * c0)**(1/2) ) #"z0" is same as "g0" in paper
    z1 = (1/2)*( -c1 + (c1**2 - 4 * c0)**(1/2) ) #"z1" is same as "g1" in paper

    #Equation (iii) in Tsai [1]
    pd = 1 * z1 - 1 * z0
    p0 = (1/pd) * (1 * z1 - 1 * m1)              #"p0" is same as "A*(k)"   in paper
    # p1 = 1- p0                                   #"p1" is same as "1-A*(k)" in paper
    #End ###################################################

    #p-tile problem: Find k' to make "A*(k')" is closest to "p0" 
    #Reference equation(15) [1]
    #Begin
    #Get the histogram and p(g) in paper
    p = get_hist(img_gray)                       
    A_star = p0             # The A_k means  A*(k) in  paper [1]
    Err_value_list = []
    Sum_p_g = 0          # It means sum from p(0)  to p(g) 
    for kp in range(2**l):
        Sum_p_g += p[kp] 
        Err_value_list.append(math.fabs(A_star - Sum_p_g ) )
        
    k = np.argmin(Err_value_list)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary






if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"input\Li et al 2021\20190918-163253-925.bmp"
    img = cv2.imread(path)
    img = cv2.resize(img,(700,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = Tsai1985(gray_img)
    cv2.imshow(f"2:{ret}",b)
    cv2.imshow("o",img)
    cv2.waitKey(0)
'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images
[2] Tsai_1985_Moment-preserving thresholding_A new approach

#log: 
#Write the Get Histogram as a function. time[2023.2.11 11:18] code[6-14]
'''