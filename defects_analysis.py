import cv2
import numpy as np
import requests, json
from pydantic import BaseModel
from pydantic import ValidationError
from typing import List,Tuple
# import os
# import time
# from tqdm import tqdm

class Annotation(BaseModel):
    box: List[float]=[]
    label: str
    score: float
    cv_bbox : List[float]=[]
    cv_size : Tuple[float] = ()

class DefectsAnalysis():
    def get_annotations_from_server(file,url):
        resp = requests.post(url, files={'file': (file, open(file, 'rb'))})
        annotations=json.loads(resp.text)["annotations"]
        print(annotations)
        return annotations

    def adp_morpho(gray):
        gray = (np.power(gray/float(np.max(gray).astype(np.float32)), 1.5).astype(np.float32)*255).astype(np.uint8)
        adp=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,55,7)
        adp=cv2.bitwise_not(adp)
        kernel = np.ones((2,2),np.uint8)
        opening = cv2.morphologyEx(adp,cv2.MORPH_OPEN,kernel, iterations = 1)
        opening = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel, iterations = 3)
        return opening

    def measure_height_width(file,annotations):
        results_list=[]
        image=cv2.imread(file)
        copy=image.copy()
        print(len(annotations))
        if len(annotations)>0:
            for df_num in range(len(annotations)):
                print(annotations[df_num])
                try:
                    annotation=Annotation(**annotations[df_num])
                    print(annotation.json())
                    rx1,ry1,rx2,ry2=annotation.box
                    # rx1,ry1,rx2,ry2=annotations[df_num]['box']
                    x1,x2,y1,y2=int(rx1*image.shape[1]),int(rx2*image.shape[1])+15,int(ry1*image.shape[0]),int(ry2*image.shape[0])
                    crop=image[y1:y2,x1:x2]
                    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                    adp=DefectsAnalysis.adp_morpho(gray)
                    contours = cv2.findContours(adp, 1, 2)[0]
                    points=np.array([[0,0]])
                    for i in contours:
                        i=i.reshape(i.shape[0],2)
                        points=np.concatenate((points, i), axis=0)
                    x,y,w,h = cv2.boundingRect(points[1:])
                    cv2.rectangle(crop,(x,y),(x+w,y+h),(0,0,255),1)
                    mask=np.zeros((crop.shape), dtype=np.uint8)
                    for i in range(adp.shape[0]):
                        for j in range(adp.shape[1]):
                            if adp[i][j]!=0:
                                mask[i][j]=(0,0,255)
                    mask_img = cv2.addWeighted(crop, 1, mask, 0.3, 0)
                    copy[y1:y2,x1:x2]=mask_img
                    cv2.rectangle(copy,(x1,y1),(x2,y2),(0,255,0),1)
                    # annotations[df_num]['cv_bbox']=[(x+x1)/image.shape[1],(y+y1)/image.shape[0],(x+x1+w)/image.shape[1],(y+y1+h)/image.shape[0]]
                    # annotations[df_num]['cv_size']=(w/image.shape[1],h/image.shape[0])
                    annotation.cv_bbox=[(x+x1)/image.shape[1],(y+y1)/image.shape[0],(x+x1+w)/image.shape[1],(y+y1+h)/image.shape[0]]
                    annotation.cv_size=(w/image.shape[1],h/image.shape[0])
                    results_list.append(annotation.dict())
                except ValidationError as e:
                    print(e.json()) 
            return results_list
        else:
            return None

# file="C:\\Users\\tiger\\Desktop\\2020-08-01\\L01_back-1_20200801050030.jpg" # 0
file="C:\\Users\\tiger\\Desktop\\2020-08-01\\L37_back-2_20200801235950.jpg" #2
print(DefectsAnalysis.measure_height_width(file,DefectsAnalysis.get_annotations_from_server(file,"http://192.168.1.119:5000/api/detect")))
# print(DefectsAnalysis.get_annotations_from_server(file,"http://192.168.1.119:5000/api/detect"))