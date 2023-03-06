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

# def get_min_val(list1,list2):
#     '''获得list1与list2组成的图像交点的左边的交点'''
#     diff = list(np.array(list1) -np.array(list2)) 
#     val1 , val2 = sorted(diff)[:2]
#     return min(diff.index(val1), diff.index(val2))

def get_min_val(list1,list2):
    '''获得list1与list2组成的图像交点的左边的交点
    要求第一个参数必须是背景图像，第二个才是前景，顺序不能乱
    '''
    diff = list(np.array(list1) -np.array(list2)) 
    res = 0
    for index,val in enumerate(diff):
        if val >= 0:
            res = index
            break
    return res

def P1(u1,std_dev1,g):
    '''std_dev and u are constant'''
    res = 1 / (std_dev1 * 2**0.5 * np.pi ** 0.5) * np.exp(- (g - u1)**2 / (2 * std_dev1**2))
    return res

def P2(u2,std_dev2,d,g):
    '''a b c d are constant
    "a" is a normalization parameter
    '''
    # res = a * np.exp( ((z - b) / c)**2 ) * np.log(1 + np.exp(0.1 *(z - d)))
    res = 1 / (std_dev2 * 2**0.5 * np.pi ** 0.5) * np.exp(- (g - u2)**2 / (2 * std_dev2**2)) * np.log(1 + np.exp(0.1 * (g - d)))
    return res

def h(g,u1,std_dev1, u2,std_dev2,d,A_star):
    ''' g is var 
        Other variables are constants that need to be estimated.
        And a is 
    '''
    res = A_star * P1(u1,std_dev1,g) + (1 - A_star) * P2(u2,std_dev2,d,g)
    return res 



def Sahli_etal_2018(img_gray,l = 8,savePath=None):
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
    '''  u1:[0,255],std_dev1:[0,255],u2:[0:255],std_dev2:[0,255],d:[-np.inf:+np.inf],A_star:[0:1]'''
    param_bounds=([ 0,0,0,0,-np.inf,0],[255,255,255,255,np.inf,1])

    popt, pcov = curve_fit(h, xdata, ydata,bounds=param_bounds,maxfev = 10000)
    
    # for i,ch in enumerate( ['u1','std_dev1', 'u2','std_dev2','d','A_star']) :
    #     print(f"{ch}: {popt[i]}")
    
    u1,std_dev1,u2,std_dev2,d,A_star = popt
    y1  = [h(z, *popt) for z in xdata]  #直方图拟合后的y值
    y_p1 = [P1(u1,std_dev1,g) * A_star for g in xdata]  #前景的图线
    y_p2 = [P2(u2,std_dev2,d,g)*(1 - A_star) for g in xdata]  #背景的图线
    plt.plot(xdata,y_p1,"g")              #绘制前景图线
    plt.plot(xdata,y_p2,"black")        #绘制背景图线
    plt.plot(xdata, y1, 'r')           #直方图拟合的图线
    plt.plot(xdata, ydata , "b")     #原始直方图
    plt.legend(labels=["target","background","histgram","origin hist"],loc="lower right",fontsize=10)
    plt.grid()   
    #寻找最接近的P1和P2的点作为相交点
    k = get_min_val(y_p2,y_p1)
    plt.axvline(x = k)
    # plt.show() 
    # 保存一下绘制的直方图
    if savePath != None:
        plt.savefig(savePath)
    plt.close()
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    # binary = cv2.bitwise_not(binary)
    return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"Sahli method\test.png"
    img = cv2.imread(path)
    # img = cv2.resize(img,(400,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = Sahli_etal_2018(gray_img)
    cv2.imshow(f"thresh:{ret}",b)
    print(f"thresh:{ret}")
    cv2.imshow("o",img)
    cv2.waitKey(0)

'''
#References:
[1] A Revisit of Various Thresholding Methods for Segmenting the Contact Area Images

#log: 

'''