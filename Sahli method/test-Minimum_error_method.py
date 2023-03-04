from sklearn.preprocessing import normalize
import Minimum_error_method1
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
import os


def img_concat_RGB_and_Binary(img,binary):
    dst = cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
    img_=cv2.hconcat([img,dst])
    return img_
    
def save_hist(img_gray,k,save_dir):
    '''to return a hist of img_gray and tick k in that'''
    plt.hist(img_gray.ravel(), 256, [0, 256])
    hist, bins = np.histogram(img_gray.ravel(), bins=256, range=(0, 256))
    # hist = [math.log(b) for b in hist]                #使用log放缩直方图
    #正则化 begin
    #options = ['l1', 'l2', 'max'] 正则化的选项
    # hist = normalize(hist.reshape(1,-1) , norm = 'l2').reshape((-1))
    ##end
    # plt.plot(hist)
    plt.plot([k],[hist[k]], color='r',marker='v',linewidth = 10)
    plt.title("img_histgram")
    plt.xlabel("gray")
    plt.ylabel("pixs")
    plt.savefig(save_dir)
    plt.close()

if __name__ == "__main__":
    path_input  = "Minimum error method/input"
    path_output = "Minimum error method/output"
    input_folders = os.listdir(path_input)
    threshold_dict = {}
    for folder_name in input_folders:
        if not os.path.exists(path_output + "/" + folder_name):
            os.makedirs(path_output + "/" + folder_name)
        for file_name in os.listdir(path_input + "/" + folder_name):
            # print(file_name)
            img = cv2.imread(path_input + "/" + folder_name + "/" + file_name)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # img_gray = cv2.resize(img_gray,(100,100))
            # img = cv2.resize(img, (100,100))
            k , binary = Minimum_error_method1.KI1986(img_gray)
            img_contact = img_concat_RGB_and_Binary(img,binary)
            cv2.imwrite(path_output + "/" + folder_name + "/" + file_name,img_contact)
            #get the hist
            save_hist(img_gray,k,path_output + "/" + folder_name + "/" + f"hist_{k}_" + file_name + ".jpg")
            #save the threshold
            threshold_dict[folder_name] = k
    #output 
    for key in threshold_dict:
        print('{0: <15}'.format(key),end= "   ")
    print()
    for key in threshold_dict:
        print('{0: ^15}'.format(threshold_dict[key]),end= "   ")



