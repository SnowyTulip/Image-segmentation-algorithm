import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm 
import math
import cv2
np.seterr(divide='ignore',invalid='ignore')
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


def G(u,std_dev,z):
    '''std_dev and u are constant'''
    res = 1 / (std_dev * 2**0.5 * np.pi) * np.exp(- (z - u)**2 / (2 * std_dev**2))
    return res

def F(a,b,c,d,z):
    '''a b c d are constant
    "a" is a normalization parameter
    '''
    res = a * np.exp( ((z - b) / c)**2 ) * np.log(1 + np.exp(0.1 *(z - d)))
    return res

def h(z, u,std_dev, a,b,c,d ,C1):
    '''z is var 
        Other variables are constants that need to be estimated.
        C1 is A*
        C2 is no need to estimate ,C2 is 1 - A*
    '''
    C2 = 1 - C1
    res = C1 * G(u,std_dev,z) + C2 * F(a,b,c,d,z)
    return res 



def Sahli_etal_2018(img_gray,l = 8):
    '''
    method: Minimum_error_method - Sahli_etal_2018
    param : img_gray mat 
            l is the bit of the image 
    return: k(threshold) and binary_image_mat
    '''
    # Reference:paper [1] 
    # Begin  ###################################################
    # Get the histogram of img_gray
    p = get_hist(img_gray,l)
    pixels = get_pixels(img_gray,l)
    # A_star_list     = [sum(pixels[0:k+1])/sum(pixels) for k in range(2**l)]
    # Sub_A_star_list = [1 - A_star for A_star in A_star_list]
    ydata = p.copy() # h(z) is ydata
    xdata = [i for i in range(len(ydata))]
    ''' u:[0,255], std_dev:[0:255],a:[0:np.inf],b:[0:255],c:[0:255],d:[-np.inf:+np.inf] ,C1:[0:1]'''
    param_bounds=([ 0,0,0,0,0,-np.inf,0],[255,255,np.inf,255,255,np.inf,1])

    popt, pcov = curve_fit(h, xdata, ydata,bounds=param_bounds,maxfev = 10000)
    
    for i,ch in enumerate( ['u','std_dev', 'a','b','c','d' ,'C1']) :
        print(f"{ch}: {popt[i]}")
    
    u,std_dev,a,b,c,d ,C1 = popt
    C2 = 1 - C1
    y1  = [h(z, *popt) for z in xdata]  #直方图拟合后的y值
    y_G = [G(u,std_dev,z)*C1 for z in xdata]  #前景的图线
    y_F = [F(a,b,c,d,z)*C2 for z in xdata]  #背景的图线
    plt.plot(xdata,y_G,"g")              #绘制前景图线
    plt.plot(xdata,y_F,"black")        #绘制背景图线
    plt.plot(xdata, y1, 'r')           #直方图拟合的图线
    plt.plot(xdata, ydata , "b")     #原始直方图
    plt.legend(labels=["target","background","histgram"],loc="lower right",fontsize=10)
    plt.grid()
    plt.show()
    
    
    k = 106
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    # binary = cv2.bitwise_not(binary)
    return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"Sahli method\asd.png"
    img = cv2.imread(path)
    # img = cv2.resize(img,(400,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = Sahli_etal_2018(gray_img)
    cv2.imshow(f"2:{ret}",b)
    print(f"thresh:{ret}")
    cv2.imshow("o",img)
    cv2.waitKey(0)

'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images

#log: 

'''