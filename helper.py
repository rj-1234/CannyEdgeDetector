import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import listdir

""" 
Specify the Image directory(should only contain the images 
and one other directory named 'output' to save output images) 
"""
image_dir = ".\\images\\" 

def read_images(source_path):
    '''
    read the image in a specific directory
    :param source_path: path to the directory contains images
    :return: a list of image objects
    '''
    imgs = []
    files = []
    for file in listdir(source_path):
        print(file)
        if 'output' in file:
            pass
        else:
            img = cv2.imread(source_path + file)
            imgs.append(img)
            files.append(file)
    print( files)
    return imgs, files

def convolution(image, kernel):
    image_height = image.shape[0]
    image_width  = image.shape[1]

    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    height = (kernel_height - 1) / 2
    width = (kernel_width - 1) / 2

    output = np.zeros((image_height, image_width))

    for i in np.arange(height, image_height - height):
        for j in np.arange(width, image_width - width):
            sum = 0
            for k in np.arange(-height, height+1):
                for l in np.arange(-width, width+1):
                    a = image[i+k, j+l]
                    p = kernel[height+k, width+l]
                    sum += (p * a)
            output[i, j] = sum
    
    return output




def gaussian_filter(img):
    """
    Returns a smoothed image with a 7 x 7 Gaussian filter for the input image
    """
    gaussian_kernel = (1.0/140) * np.array(
                        [[1 ,1 ,2 ,2 ,2 ,1 ,1],
                         [1 ,2 ,2 ,4 ,2 ,2 ,1],
                         [2 ,2 ,4 ,8 ,4 ,2 ,2],
                         [2 ,4 ,8 ,16,8 ,4 ,2],
                         [2 ,2 ,4 ,8 ,4 ,2 ,2],
                         [1 ,2 ,2 ,4 ,2 ,2 ,1],
                         [1 ,1 ,2 ,2 ,2 ,1 ,1]])

    s = sum(sum(gaussian_kernel))
    print(s)
    print("Running Smoothing operation using a 7 x 7 Gaussian Kernel")

    after7x7Smoothning = np.copy(img)

    for i in range(3,len(img)-3):
        for j in range(3,len(img[i])-3):
            after7x7Smoothning[i][j] = (sum(map(sum, (gaussian_kernel * after7x7Smoothning[i-3:i+4,j-3:j+4]))))
    
    print("DONE")
    return after7x7Smoothning

def greyscale_Image(rgb):
    """
    Returns a gray scale version of the input image
    """
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def gradient(img):
    """
    Returns the normalized horizonal and vertical gradient for the input image
    """
    prewitt_hor = np.array([[-1 ,0 ,1],
                            [-1 ,0 ,1],
                            [-1 ,0 ,1]])

    prewitt_ver = np.array([[-1 ,-1 ,-1],
                            [ 0 , 0 , 0],
                            [ 1 , 1 , 1]])

    hor_gradient = np.copy(img)
    print("Running horizonal gradient operation using prewitt_hor Kernel")
    hor_gradient = convolution(hor_gradient, prewitt_hor)
    print("DONE")

    ver_gradient = np.copy(img)
    print("Running vertical gradient operation using prewitt_ver Kernel")
    ver_gradient = convolution(ver_gradient, prewitt_ver)
    print("DONE")

    return hor_gradient, ver_gradient

def NMsuppression(img, grad_h, grad_v, grad_m):

    height = img.shape[0]
    width = img.shape[1]

    angle = np.arctan2(grad_v, grad_h)

    quantized_angle = (np.round(angle * (5.0 / np.pi) + 5 )) % 5 
    supressed_grad_m = grad_m.copy()

    print("Running Non Maxima Suppression operation on gradient magnitude")
    for i in range(height):
        for j in range(width):
            if i == 0 or i == height - 1 or j == 0 or j == width - 1:
                supressed_grad_m[i, j] = 0

            qa = quantized_angle[i, j] % 4

            if qa == 0: # 0 -> E-W (horizontal)
                if supressed_grad_m[i, j] <= supressed_grad_m[i, j-1] or supressed_grad_m[i, j] <= supressed_grad_m[i, j+1]:
                    supressed_grad_m[i, j] = 0
            
            if qa == 1: # 1 -> NE SW
                if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j+1] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j-1]:
                    supressed_grad_m[i, j] = 0
            
            if qa == 2: # 2 -> N-S (vertical)
                if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j]:
                    supressed_grad_m[i, j] = 0
            
            if qa == 3: # 3 -> NW SE
                if supressed_grad_m[i, j] <= supressed_grad_m[i-1, j-1] or supressed_grad_m[i, j] <= supressed_grad_m[i+1, j+1]:
                    supressed_grad_m[i, j] = 0
    print("DONE")
    return supressed_grad_m
            
def thresholding(img, thresholdValue):
    img_copy = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img_copy[i, j] < thresholdValue: img_copy[i, j] = 0
    return img_copy

def ptile(img, p):
    height = img.shape[0]
    width = img.shape[1]

    gray_val = []

    for i in range(height):
        for j in range(width):
            if img[i, j] > 0.0:
                gray_val.append(img[i, j])
    print ("Length of Gray value list : "+str(len(gray_val)))
    gray_val.sort(reverse=True)
    # print(gray_val[:100])
    idx = int((p/float(100)) * len(gray_val))
    print ("Chosen index from gray value list : "+str(idx))
    print ("Total no. of edges detected for p = "+str(p)+" : "+str(len(gray_val[:idx+1])))
    print ("Chosen gray level value for this p value : "+str(gray_val[idx]))
    return gray_val[idx]

def save(images, names):
    """
    Takes 2 lists
        images : list of images as numpy arrays
        names  : list of names as a string
    """
    for i in range(len(names)):
        # print("saving : "+names[i])
        print(names[i])

        cv2.imwrite(image_dir+'output/'+str(names[i])+' image.png', images[i])

def show(images, names):
    """
    Takes 2 lists
        images : list of images as numpy arrays
        names  : list of names as a string
    """
    for i in range(len(names)):
        if str(images[i]) == 'gray_img':
            plt.imshow(images[i], cmap = plt.get_cmap('gray'))
        else:
            plt.imshow(images[i])
        print("showing : "+str(names[i]))        
        plt.show()
