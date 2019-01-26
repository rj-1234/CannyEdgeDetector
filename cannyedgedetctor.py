import cv2
import numpy as np
from helper import *


if __name__== "__main__":
    """ 
    Specify the Image directory(should only contain the images 
    and one other directory named 'output' to save output images) 
    """
    image_dir = ".\\images\\"  

    """ Read the Image """
    images, filename = read_images(image_dir)
    for i in range(len(images)): 
        img = images[i]
        img = np.array(img, dtype=float)
        name = filename[i]
        print("\n################### "+name+" ###################\n")

        """ Conver to grayscale """
        gray_img = greyscale_Image(img)

        """ Gaussian filtering/smoothing operation """
        smooth_img = gaussian_filter(gray_img)

        # cv2 version for comparison
        # cv2_smooth_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

        """ Computing horizontal and vertical gradients for the smoothed image """
        horizontal_gradient_img, vertical_gradient_img = gradient(smooth_img)

        """ Compute the gradient magnitude """
        gradient_magnitude_img = np.sqrt(np.power(horizontal_gradient_img, 2)/6.0 + np.power(vertical_gradient_img, 2)/6.0)
        
        """ Non Maxima Suppression """
        NNMsuppressed_img = NMsuppression(smooth_img, horizontal_gradient_img, vertical_gradient_img , gradient_magnitude_img)
       
        """ Normalizing for the range -> 0 - 255 """
        normalized_gradient_magnitude_img =(gradient_magnitude_img / np.max(gradient_magnitude_img)) * 255
        normalized_horizontal_gradient_img, normalized_vertical_gradient_img = horizontal_gradient_img/6.0, vertical_gradient_img/6.0
        normalized_NNMsuppressed_img = (NNMsuppressed_img / np.max(NNMsuppressed_img)) * 255

        """ Save Images """
        save([img, gray_img, 
                smooth_img, 
                normalized_horizontal_gradient_img, 
                normalized_vertical_gradient_img,
                normalized_gradient_magnitude_img,
                normalized_NNMsuppressed_img], 
                ["Original "+name+" ", 
                "Gray "+name+" ", 
                "Smoothed "+name+" ", 
                "Horizontal Gradient "+name+" ", 
                "Vertical Gradient "+name+" ",
                "Gradient Magnitude "+name+" ",
                "Non-Maxima Suppressed "+name+" "])
        
        # show([img, gray_img, 
        #         smooth_img, 
        #         normalized_horizontal_gradient_img, 
        #         normalized_vertical_gradient_img,
        #         normalized_gradient_magnitude_img,
        #         normalized_NNMsuppressed_img, 
        #         threshold_img], 
        #         ["Original "+name+" ", 
        #         "Gray "+name+" ", 
        #         "Smoothed "+name+" ", 
        #         "Horizontal Gradient "+name+" ", 
        #         "Vertical Gradient "+name+" ",
        #         "Gradient Magnitude "+name+" ",
        #         "Non-Maxima Suppressed "+name+" ",
        #         "Threshold_"+str(p)+" "+name+" "])

        """ Thresholding """
        for p in [10, 30, 50]:
            print("P = "+str(p))
            T = ptile(normalized_NNMsuppressed_img, p)
            threshold_img = thresholding(normalized_NNMsuppressed_img, T)
            """ Show/Save Images """
            save([threshold_img], 
                ["Threshold_"+str(p)+" "+name+" "])

            