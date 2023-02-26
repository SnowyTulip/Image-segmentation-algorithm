import numpy as np
import cv2

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

def E_(k_,g1_mean_list, g2_mean_list,l = 8):

    E1 = 0
    for g1 in range(0,k_):    # 0-k
        E1 += (g1-g1_mean_list[k_])**2
    E2 = 0
    for g2 in range(k_ ,2**l): # k-2**l-1
        E2 += (g2-g2_mean_list[k_])**2
    return E1 + E2

def K_means(img_gray,l = 8):
    p = get_hist(img_gray,l)
    pixels = get_pixels(img_gray,l)
    #使用计数方法求A*(k)
    A_star_list     = [sum(pixels[0:k+1])/sum(pixels) for k in range(2**l)]
    Sub_A_star_list = [1 - A_star for A_star in A_star_list]

    g1_k = [0 for i in range(256)]
    g2_k = [0 for i in range(256)]
    
    for k in range(0,2**l):
        t1 = 0
        if A_star_list[k] != 0:
            for g1 in range(k):
                t1 += g1 * p[g1]/A_star_list[k] 
        g1_k[k] = t1

        t2 = 0
        if Sub_A_star_list[k] != 0:
            for g2 in range(k,2**l):
                t2 += g2 * p[g2]/Sub_A_star_list[k]  
        g2_k[k] = t2
    # g1_mean_list = [np.sum(k*p[k])/sum(p[0:k+1]) for k in range(2**l)]
    # g2_mean_list = [np.sum(k*p[k])/Sub_A_star_list[k] for k in range(2**l)]

    E_list = [E_(k_,g1_k, g2_k) for k_ in range(0, 2**l)]
    k = np.argmin(E_list)
    _ , binary = cv2.threshold(img_gray, k, 2**l - 1, cv2.THRESH_BINARY)
    return k,binary

if __name__ == "__main__":
    path = r"input\Liang\2021_October_31 20.48.50.559.png"
    img = cv2.imread(path)
    img = cv2.resize(img,(400,400))
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,b  = K_means(gray_img)
    cv2.imshow(f"threshold:{ret}",b)
    cv2.imshow("orignal",img)
    cv2.waitKey(0)