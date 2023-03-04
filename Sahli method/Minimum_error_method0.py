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

# def GetValues(p,A_star_list,Sub_A_star_list,l = 8):
#     '''
#     param:   A_star_list[k] is A*(k) 
#         also Sub_A_star_list[k] is 1 - A*(k)
#         p is histgram of img_gray
#     return g1_list,g2_list,std_dev1_list,std_dev2_list
#     '''
#     g1_list       = [0 for k in range(2**l)]
#     g2_list       = [0 for k in range(2**l)]
#     std_dev1_list = [0 for k in range(2**l)]
#     std_dev2_list = [0 for k in range(2**l)]
#     for k in range(0,2**l):
#         acc1 = 0
#         if A_star_list[k] != 0:
#             for g1 in range(k):
#                 acc1 += g1 * p[g1]/A_star_list[k] 
#         g1_list[k] = acc1

#         acc2 = 0
#         if Sub_A_star_list[k] != 0:
#             for g2 in range(k,2**l):
#                 acc2 += g2 * p[g2]/Sub_A_star_list[k]  
#         g2_list[k] = acc2
#         ### reference
#         acc3 = 0
#         if A_star_list[k] != 0:
#             for g in range(k):
#                 acc3 += (g - g1_list[k])**2 * p[g1]/A_star_list[k] 
#         std_dev1_list[k] = acc3
#         acc4 = 0
#         if Sub_A_star_list[k] != 0:
#             for g in range(k,2**l):
#                 acc4 += (g - g2_list[k])**2 * p[g2]/Sub_A_star_list[k] 
#         std_dev2_list[k] = acc4
#     return g1_list,g2_list,std_dev1_list,std_dev2_list


# def Get_G(u):
#     '''
#     '''
#     if u == 1:
#         res = 0
#     else:
#         res = -u * math.log(u) - (1 - u) * math.log(1 - u)
#     return res

# def Get_J_List(p,g1_list,g2_list,std_dev1_list,std_dev2_list,A_star_list,Sub_A_star_list,l):
#     '''
#     Reference paper[1] 30
#     '''
#     arg_pack = (g1_list,g2_list,std_dev1_list,std_dev2_list,A_star_list,Sub_A_star_list)
#     J_list = [0 for _ in range(2**l)]
#     for k in range(2**l):
#         acc = 0
#         for g in range(2**l):
#             acc += p[g] * E_g_k(*arg_pack,g,k)
#         J_list[k] = acc
#     return J_list






# def E_g_k(g1_list,g2_list,std_dev1_list,std_dev2_list,A_star_list,Sub_A_star_list,g,k):
#     '''
#     Reference paper[1] 29
#     Get E(g,k) in paper[1] 29
#     '''
#     g_mean   = g1_list[k]       if g <= k else g2_list[k]
#     std_dev  = std_dev1_list[k] if g <= k else std_dev2_list[k]
#     A_star   = A_star_list[k]   if g <= k else Sub_A_star_list[k]
#     if std_dev == 0:
#         res = 1e10
#     else:
#         res = (g - g_mean)**2/ std_dev - 2 * math.log(A_star) + 2 * math.log(std_dev ** 0.5)
#     return res

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

    popt, pcov = curve_fit(h, xdata, ydata,bounds=param_bounds)
    
    for i,ch in enumerate( ['u','std_dev', 'a','b','c','d' ,'C1']) :
        print(f"{ch}: {popt[i]}")
    # plt.plot(xdata, ydata , "b")       #直方图
    u,std_dev,a,b,c,d ,C1 = popt
    C2 = 1 - C1
    y1  = [h(z, *popt) for z in xdata]  #直方图拟合后的y值
    y_G = [G(u,std_dev,z)*C1 for z in xdata]  #前景的图线
    y_F = [F(a,b,c,d,z)*C2 for z in xdata]  #背景的图线
    plt.plot(xdata,y_G,"g")              #绘制前景图线
    plt.plot(xdata,y_F,"black")        #绘制背景图线
    plt.plot(xdata, y1, 'r')           #直方图拟合的图线
    plt.legend(labels=["target","background","histgram"],loc="lower right",fontsize=10)
    plt.grid()
    plt.show()
    
    
    k = 106
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary



if __name__ == "__main__":
    # img_concat_RGB_and_Binary(1)
    path = r"Sahli method\input\Sahli et a. 2018\sahli.png"
    img = cv2.imread(path)
    img = cv2.resize(img,(400,400))
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