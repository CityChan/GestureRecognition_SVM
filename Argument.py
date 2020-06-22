import random
import cv2
path = './' + 'images' + '/'


#rotate
def rotate(image, scale=0.9):
    angle = random.randrange(-90, 90) # random degree
    w = image.shape[1]
    h = image.shape[0]
    #rotate matrix
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, scale)
    #rotate
    image = cv2.warpAffine(image,M,(w,h))
    return image


if __name__ == "__main__":
    for i in range(10):
        cnt = 21  # count
        for j in range(1, 21):
            roi = cv2.imread(path + str(i) + '_' + str(j) + '.png')
            for k in range(12):
                img_rotation = rotate(roi)  # rotating
                cv2.imwrite(path + str(i) + '_' + str(cnt) + '.png', img_rotation)
                cnt += 1
                img_flip = cv2.flip(img_rotation, 1)  # fliping
                cv2.imwrite(path + str(i) + '_' + str(cnt) + '.png', img_flip)
                cnt += 1
            print(i, '_', j, 'Finished')
