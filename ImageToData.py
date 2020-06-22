#!/usr/bin/env python
# -*-coding:utf-8 -*-
import FourierDescriptor as fd
import cv2
import numpy as np

path = './' + 'features' + '/'
path_img = './' + 'images' + '/'

if __name__ == "__main__":
    for i in range(1, 11):
        for j in range(1, 501):
            roi = cv2.imread(path_img + str(i) + '_' + str(j) + '.png')
            descirptor_in_use = abs(fd.FourierDescriptor(roi)[1])
            fd_name = path + str(i) + '_' + str(j) + '.txt'
            with open(fd_name, 'w', encoding='utf-8') as f:
                temp = descirptor_in_use[1]
                for k in range(1, len(descirptor_in_use)):
                    x_record = int(100 * descirptor_in_use[k] / temp)
                    f.write(str(x_record))
                    f.write(' ')
                f.write('\n')
            print(i, '_', j, 'Finished')