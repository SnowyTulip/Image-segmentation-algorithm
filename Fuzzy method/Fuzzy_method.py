import numpy as np
from tqdm import tqdm 
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

def GetValues(p,A_star_list,Sub_A_star_list,l = 8):
    '''
    param:   A_star_list[k] is A*(k) 
        also Sub_A_star_list[k] is 1 - A*(k)
        p is histgram of img_gray
    '''
    g1_list = [0 for k in range(2**l)]
    g2_list = [0 for k in range(2**l)]
    C_list  = [0 for k in range(2**l)]
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
        C_list[k] = max( [g1_list[k],abs(k - g1_list[k]),abs(2**l-1 - g2_list[k]),abs(k-g2_list[k]) ] )
    return g1_list,g2_list,C_list

def U(i,j,k,img_gray,g1,g2,C):
    '''
    param :i, j is position of a pixel in the img_gray
            k : is Gray value
            img_gray is just ....
            g1,g2 is the Mean Gray value list
            C is Reference [1] (23)
    Reference [1] (23) is the function "U ij(k)"
    '''
    bias = 1e-5
    g = g1[k] if img_gray[i,j] <= k else g2[k]
    C_val = max(C)
    res = (1 - bias) / (1 + abs(img_gray[i][j] - g) / C_val)
    return res

def S(u):
    '''
    '''
    if u == 1:
        res = 0
    else:
        res = -u * math.log(u) - (1 - u) * math.log(1 - u)
    return res

def Get_E(img_gray,g1,g2,C,l = 8):
    '''
    '''
    E = [0 for k in range(2**l)]
    Nx,Ny = img_gray.shape
    #A three-tier python loop, just like a snail ....
    with tqdm(total=2**l*Nx*Ny,colour= "red") as pbar:
        pbar.set_description(f'Processing')
        for k in range(2**l):
            acc = 0
            for i in range(Nx):
                for j in range(Ny):
                    # print(U(i,j,k,img_gray,g1,g2,C))
                    acc += S(U(i,j,k,img_gray,g1,g2,C))
                    pbar.update(1)
            E[k] = 1 / (Nx * Ny * math.log(2)) * acc
        
    return E




def HW1995(img_gray,l = 8):
    '''
    method: Fuzzy method
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
    g1,g2,C  = GetValues(p,A_star_list,Sub_A_star_list,l)
    E_list = Get_E(img_gray,g1,g2,C,l)
    k = np.argmin(E_list)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary

    # return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"input\Mergel et al. 2018\Image_m=000.00_v=0.1_d=3.5_range=02N_000004.png"
    img = cv2.imread(path)
    img = cv2.resize(img,(100,100))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = HW1995(gray_img)
    cv2.imshow(f"2:{ret}",b)
    print(f"thresh:{ret}")
    cv2.imshow("o",img)
    cv2.waitKey(0)

'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images

#log: 

'''