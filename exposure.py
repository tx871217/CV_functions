import numpy as np
import cv2
from matplotlib import pyplot as plt

def import_org(image_dir):
    image=cv2.imread(image_dir)
    # cv2.imshow('org',image)
    return image

def Show_origin_and_output(origin, I):
   """
   Show final result.
   """
   plt.figure(figsize=(12, 6))
   plt.subplots_adjust(left=0,right=1,bottom=0,top=1, wspace=0.005, hspace=0)

   plt.subplot(121),plt.imshow(np.flip(origin, 2)),plt.title('Origin')
   plt.axis('off')
   plt.subplot(122),plt.imshow(np.flip(I, 2)),plt.title('Fake HDR')
   plt.axis('off')
#    plt.savefig('compare.png', bbox_inches='tight', pad_inches=0)
   plt.show()

# x=np.array([[1.0,0.9,0.8],
#             [0.5,0.4,0.3],
#             [0.0,0.1,0.2]])
# print(x*0.7)

img_dir="C:\\Users\\tiger\\Desktop\\mid_iron\\2020_08_31_23_56_24.bmp"
img=import_org(img_dir)
cv2.imshow('org',result)

cv2.waitKey(0)
cv2.destroyAllWindows()