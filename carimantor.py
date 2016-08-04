import cv2
import numpy as np
import sys


def cal_scale_size(face , caricature_head_height):
    face_height, face_width = face.shape[0:2]
    height_scale_size = float(caricature_head_height) / float(face_height)
    return height_scale_size

def scale_face(face , caricature_head_height):
    scale_size = cal_scale_size(face , caricature_head_height)
    scaled_face = cv2.resize(face , None , fx=scale_size , fy=scale_size)
    return scaled_face.copy()

def ellipse_face(face):
    mask = np.zeros_like(face)
    rows , cols = mask.shape[0:2]
    cv2.ellipse(mask , center=(rows/2 , cols/2) , axes=(60,70) , angle=0 , startAngle=0,
                endAngle=360 , color=(255,255,255) , thickness=-1)
    return cv2.bitwise_and(face , mask)

def find_face(img , img_gray):
    face_cascade = cv2.CascadeClassifier('haar_cascade_files/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_gray, 1.05, 10, 0)
    (x , y , w , h) = faces[0]
    return img[y:y+h ,  x:x+w]

def dodgeNaive(image, mask):
    return cv2.divide(image, 255-mask, scale=256)

def filter_image_RGB(image):
    image_3 = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    image_4 = cv2.medianBlur(image_3 , 3)
    image_5 = cv2.adaptiveThreshold(image_4 , 255 , cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV , blockSize=9 , C=2)
    image_5 = cv2.cvtColor(image_5 , cv2.COLOR_GRAY2RGB)
    return image_5

def filter_image_GRAY(image_gray):
    image_4 = cv2.medianBlur(image_gray , 7)
    image_5 = cv2.adaptiveThreshold(image_4 , 255 , cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY , blockSize=9 , C=2)
    return image_5

def remove_pepper_noise(mask):
    rows , cols = mask.shape[0:2]
    y = 2
    while( y < rows-2 ):
        pThis = y
        pUp1 = y - 1
        pUp2 = y - 2
        pDown1 = y + 1
        pDown2 = y + 2
        x = 2
        while( x < cols-2 ):
            if ( mask[pThis , x] == 0):
                allAbove = ( (mask[pUp2 , x - 2]) and (mask[pUp2 , x - 1]) and (mask[pUp2 , x]) and (mask[pUp2 , x+1]) and (mask[pUp2 , x+2]) )
                allLeft = ( (mask[pUp1, x - 2]) and (mask[pThis, x - 2]) and (mask[pDown1, x-2]) )
                allBelow = ( (mask[pDown2, x - 2]) and (mask[pDown2, x - 1]) and (mask[pDown2, x]) and (mask[pDown2, x+1]) and (mask[pDown2, x+2]) )
                allRight = ((mask[pUp1, x + 2]) and (mask[pThis, x + 2]) and (mask[pDown1, x + 2]))
                surroundings = allAbove and allLeft and allBelow and allRight
                if( surroundings == True):
                    mask[pUp1, x - 1] = 255
                    mask[pUp1, x] = 255
                    mask[pUp1, x + 1] = 255
                    mask[pThis, x - 1] = 255
                    mask[pThis, x] = 255
                    mask[pThis, x + 1] = 255
                    mask[pDown1, x - 1] = 255
                    mask[pDown1, x] = 255
                    mask[pDown1, x + 1] = 255
            x += 1
        y += 1
    return mask

def sketch(image):
    image_3 = cv2.cvtColor(image , cv2.COLOR_RGB2GRAY)
    image_4 = cv2.medianBlur(image_3 , 3)
    image_5 = cv2.adaptiveThreshold(image_4 , 255 , cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY , blockSize=9 , C=2)
    image_5 = cv2.cvtColor(image_5 , cv2.COLOR_GRAY2RGB)
    return image_5

def cartoonify(image):
    rows , cols = image.shape[0:2]

    img_gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    image_median_blur = cv2.medianBlur(img_gray , 7)

    #mask = np.zeros_like(image)
    mask = np.zeros((rows , cols , 3) , dtype=np.uint8)
    #edges = np.zeros_like(image)
    edges = np.zeros((rows , cols , 3) , dtype=np.uint8)

    edges = cv2.Laplacian(image_median_blur , cv2.CV_8U , 5)
    ret , mask = cv2.threshold(edges , 4 , 255 , cv2.THRESH_BINARY_INV)
    #mask = remove_pepper_noise(mask)
    sketch = cv2.cvtColor(mask , cv2.COLOR_GRAY2BGR)
    repitition = 7
    cpy_image = image.copy()
    for i in range(repitition):
        size = 17
        sigmacolor = 9
        sigmaspace = 7
        tmp = cv2.bilateralFilter(cpy_image, size, sigmaColor=sigmacolor, sigmaSpace=sigmaspace)
        cpy_image = cv2.bilateralFilter(tmp, size, sigmaColor=sigmacolor, sigmaSpace=sigmaspace)
    return cpy_image

def remove_white_pixel(image):
    img_copy = image.copy()
    rows , cols = img_copy.shape[0:2]
    i = 0
    while( i < rows ):
        j = 0
        while( j < cols ):
            first_val = img_copy.item(i , j)
            if( first_val >= 240):
                img_copy[i , j] = 0
            j += 1
        i += 1
    return img_copy

def black_to_RGB(image , color):
    img = image.copy()
    rows , cols = img.shape[0:2]
    i = 0
    while( i < rows ):
        j = 0
        while( j < cols ):
            fir_val = img.item(i , j , 0)
            sec_val = img.item(i , j , 1)
            thr_val = img.item(i , j , 2)
            if (fir_val == 0) and (sec_val == 0) and (thr_val == 0):
                img[i , j] = color
            j += 1
        i += 1
    return img

def black_to_GRAY(image , color):
    img = image.copy()
    rows , cols = img.shape[0:2]
    i = 0
    while( i < rows ):
        j = 0
        while( j < cols ):
            fir_val = img.item(i , j , 0)
            sec_val = img.item(i , j , 1)
            thr_val = img.item(i , j , 2)
            if (fir_val == 0) and (sec_val == 0) and (thr_val == 0):
                img[i , j] = color
            j += 1
        i += 1
    return img

def write_mat_to_file(mat , file_name):
    f = open(file_name , 'w')
    rows, cols = mat.shape[0:2]
    i = 0
    while (i < rows):
        j = 0
        while (j < cols):
            fir_val = mat.item(i, j, 0)
            sec_val = mat.item(i, j, 1)
            thr_val = mat.item(i, j, 2)
            f.write('[' + str(fir_val) + ',' + str(sec_val) + ',' + str(thr_val) + ']')
            f.write('\t')
            j += 1
        f.write('\n')
        i += 1
    return

def create_caricature_RGB(face , caricature , transition_y , transition_x):
    # Scaled Face information
    face_height, face_width = face.shape[0:2]
    # ROI
    roi = caricature[transition_y:transition_y+face_height ,transition_x:transition_x+face_width]
    # Scaled Face Gray
    scaled_face_gray = cv2.cvtColor(face , cv2.COLOR_RGB2GRAY)
    # Ret and Mask
    ret , mask = cv2.threshold(scaled_face_gray , 0 , 255 , cv2.THRESH_BINARY)
    # Mask inverse
    mask_inv = cv2.bitwise_not(mask)
    # caricature background
    caricature_bg = cv2.bitwise_and(roi , roi , mask=mask_inv)
    # Scaled Face foreground
    face_fg = cv2.bitwise_and(face , face , mask=mask_inv)
    # Do Mix
    dst = cv2.add(caricature_bg , face_fg)
    dst = black_to_RGB(dst , [102 , 132 , 150] )
    caricature[transition_y:transition_y+face_height ,transition_x:transition_x+face_width ] = dst
    return caricature

def create_caricature_GRAY(face_gray , caricature , transition_y , transition_x):
    # Scaled Face information
    caricature = cv2.cvtColor(caricature , cv2.COLOR_BGR2GRAY)
    face_height, face_width = face_gray.shape[0:2]
    # ROI
    roi = caricature[transition_y:transition_y+face_height ,transition_x:transition_x+face_width]
    # Ret and Mask
    ret , mask = cv2.threshold(face_gray , 0 , 255 , cv2.THRESH_BINARY)
    # Mask inverse
    mask_inv = cv2.bitwise_not(mask)
    # caricature background
    caricature_bg = cv2.bitwise_and(roi , roi , mask=mask_inv)
    # Scaled Face foreground
    scaled_face_fg = cv2.bitwise_and(face_gray , face_gray , mask=mask)
    # Do Mix
    dst = cv2.add(caricature_bg , scaled_face_fg)
    caricature[transition_y:transition_y+face_height ,transition_x:transition_x+face_width ] = dst
    return caricature

def show_and_destroy(image):
    cv2.imshow('Hi', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()

if( __name__ == '__main__'):
    # img = cv2.imread(sys.argv[1])
    # caricature = cv2.imread(sys.argv[2])

    img = cv2.imread('pics/me.png')
    caricature = cv2.imread('caricature/man_2.png')

    img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)

    face = find_face(img , img_gray)
    # Gray Face
    face_gray = cv2.cvtColor(face , cv2.COLOR_BGR2GRAY)
    # Invert Gray Face
    invert_gray_face = 255 - face_gray
    # Invert Gray Gaussian Blur Face
    invert_gray_gaussian_blur_face = cv2.GaussianBlur(invert_gray_face , (121,121) , 0)

    invert_gray_gaussian_blur_dodge_face = dodgeNaive(face_gray , invert_gray_gaussian_blur_face)

    scaled_face = scale_face(invert_gray_gaussian_blur_dodge_face , 160)

    #filtered_scaled_face = filter_image_GRAY(scaled_face)
    ellipsed_filtered_scaled_face = ellipse_face(scaled_face)
    ellipsed_filtered_scaled_face = remove_white_pixel(ellipsed_filtered_scaled_face)
    final_caricature = create_caricature_GRAY(ellipsed_filtered_scaled_face , caricature , 50 , 100 )

    show_and_destroy(final_caricature)

# az bala payin yekam boride beshe. hamin juri.
# markaz beyzi az damagh age peyda shod.