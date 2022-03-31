import cv2
import numpy as np

def import_org(image_dir):
    image=cv2.imread(image_dir)
    cv2.imshow('org',image)
    return image

def to_gray(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray',gray)
    return gray

#otsu
def to_otsu(gray):
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imshow('thresh',thresh)
    return thresh

#laplacian
def to_laplacian(gray):
    laplacian = cv2.Laplacian(gray,cv2.CV_32F)
    cv2.imshow('laplacian',laplacian)
    return laplacian

#sobel
def to_sobel(gray):
    x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)# 轉回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imshow('Sobel',dst)
    return absX,absY,dst

#denoise
def to_denoise_gray(gray):
    denoise=cv2.fastNlMeansDenoising(gray)
    cv2.imshow('denoise',denoise)
    return denoise

#build kernel
def build_kernel(k=3):
    kernel = np.ones((k,k),np.uint8)
    return kernel

class to_blur():
    def average(img):
        blur = cv2.blur(img,(5,5))
        return blur
    def gaussian(img):
        blur = cv2.GaussianBlur(img,(5,5),0)
        return blur
    def median(img):
        median = cv2.medianBlur(img,5)
        return median
    def bilate(img):
        blur = cv2.bilateralFilter(img,9,75,75)
        return blur

def to_open(img,kernel):
    open=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel, iterations = 2)
    cv2.imshow("open",open)
    return open

def to_clahe(gray):
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    dst = clahe.apply(gray)
    cv2.imshow('dst',dst)
    return dst

def unevenLightCompensate(gray, blockSize=16):
    average = np.mean(gray)
    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))
    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]
            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver
    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    return dst

def custom_blur_demo(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #銳化
    dst = cv2.filter2D(image, -1, kernel=kernel)
    cv2.imshow("custom_blur_demo", dst)
    return dst

def adp_morpho(gray):
    adp=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,25,6)
    adp=cv2.bitwise_not(adp)
    kernel = np.ones((2,2),np.uint8)
    opening = cv2.morphologyEx(adp,cv2.MORPH_OPEN,kernel, iterations = 1)
    opening = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 3)
    cv2.imshow("adp_morpho", opening)
    return opening

def gamma(gray):
    img1 = np.power(gray/float(np.max(gray)), 1/1.5)
    img2 = np.power(gray/float(np.max(gray)), 1.5)

    cv2.imshow('gamma=1/1.5',img1)
    cv2.imshow('gamma=1.5',img2)

if __name__ == "__main__":
    image_dir="C:\\Users\\tiger\\Desktop\\defects detection\\L01_back-3_20200801060150.jpg"#"C:\\Users\\tiger\\Desktop\\mid_iron\\2020_08_31_23_02_56.bmp"
    org=import_org(image_dir)
    # gray=to_gray(org)
    custom_blur_demo(org)
    g=to_gray(org)
    # z=adp_morpho(g)
    gamma(g)
    
    # denoise=to_denoise_gray(gray)
    # lapla=to_laplacian(denoise)
    # k=build_kernel(500)
    # open=to_open(denoise,k)
    # blur=to_blur.bilate(lapla)
    # cv2.imshow("bb",blur)
    # to_clahe(gray)
    

cv2.waitKey(0)
cv2.destroyAllWindows()