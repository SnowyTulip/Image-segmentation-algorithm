import Sahli_etal_2018
import Sahli_etal_2018_without_a
import matplotlib.pyplot as plt
import numpy as np
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
    plt.plot([k],[hist[k]], color='r',marker='v',linewidth = 10)
    plt.title("img_histgram")
    plt.xlabel("gray")
    plt.ylabel("pixs")
    plt.savefig(save_dir)
    plt.close()

if __name__ == "__main__":
    path_input  = "Sahli method/input"
    path_output = "Sahli method/output"
    input_folders = os.listdir(path_input)
    threshold_dict = {}
    for folder_name in input_folders:
        threshold_list = []
        if not os.path.exists(path_output + "/" + folder_name):
            os.makedirs(path_output + "/" + folder_name)
        for file_name in os.listdir(path_input + "/" + folder_name):
            print(file_name)
            img = cv2.imread(path_input + "/" + folder_name + "/" + file_name)
            img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #Sahli 原文中的处理方法
            k , binary = Sahli_etal_2018.Sahli_etal_2018(img_gray,savePath = path_output + "/" + folder_name + "/" + f"hist_" + file_name + ".jpg")
            #paper 中的改进的处理方法 ，其中不包含待拟合参数 a
            # k , binary = Sahli_etal_2018_without_a.Sahli_etal_2018(img_gray,savePath = path_output + "/" + folder_name + "/" + f"hist_" + file_name + ".jpg")
            img_contact = img_concat_RGB_and_Binary(img,binary)
            cv2.imwrite(path_output + "/" + folder_name + "/" + file_name,img_contact)
            threshold_dict[folder_name] = k
            #将本次求得阈值加入
            threshold_list.append(k)
        with open(path_output + "/" + folder_name + f"/{folder_name}.txt",'w',encoding = "utf-8") as f:
            f.write(f"阈值  :{str(threshold_list)}\n")
            f.write(f"阈值均值:{np.mean(threshold_list)}\n")
            f.write(f"阈值方差:{np.var (threshold_list)}\n")
    #output 
    for key in threshold_dict:
        print('{0: <15}'.format(key),end= "   ")
    print()
    for key in threshold_dict:
        print('{0: ^15}'.format(threshold_dict[key]),end= "   ")



