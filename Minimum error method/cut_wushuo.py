import numpy as np
import cv2
#用来切割.\Minimum error method\input\WuShuo_cut 的照片
if __name__ == "__main__":
    img = cv2.imread(r"Minimum error method\input\WuShuo\img_20N_05.jpg")
    print(img.shape)
    img_cr = img[0:300,:,:,] #保留上半部分
    img_cr = img[300:550,60:600]
    cv2.imwrite(r"Minimum error method\input\WuShuo_cut\img_20N_05_cut.jpg",img_cr)
    cv2.imshow("r",img_cr)
    cv2.waitKey(0)