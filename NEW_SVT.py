#For data analysis
import numpy as np
import pandas as pd
import sys
from scipy.stats import norm

#For navigating directories
import os

#For graphing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.mlab as mlab


#Image processing
from skimage.measure import regionprops,regionprops_table, label
from skimage import morphology
from skimage.segmentation import expand_labels,watershed
from skimage.feature import peak_local_max
from skimage import exposure
from scipy.ndimage import distance_transform_edt
from PIL import Image as im
# import cv2 as cv
import seaborn as sns 

#Statistics
from skimage.color import label2rgb
from skimage.feature import blob_log



    
def detect_LoG(img_laplace,img_intensity):   

    img_pre = label(img_laplace)
    img_label = morphology.remove_small_objects(img_pre, 4)
    
    
    
    laplace_rprops = pd.DataFrame(regionprops_table(label_image=img_label,
                                                    intensity_image= img_laplace,
                                                    properties=('label','mean_intensity','area')))
    intensity_rprops = pd.DataFrame(regionprops_table(label_image=img_label,
                                                      intensity_image= img_intensity,
                                                      properties=('label','mean_intensity','area')))
    
    return img_label


class DIA:
    def __init__(self,mask_raw,target_raw,mask_lap,target_lap): 



        mask_label = detect_LoG(mask_lap, mask_raw) 
        target_label = detect_LoG(target_lap, target_raw) 
        

        target_binary = target_label.copy()
        target_binary[target_binary>0]=1
        target_binary[target_binary<0]=0
        


        data = pd.DataFrame(regionprops_table(label_image=mask_label,
                                              intensity_image= mask_raw,
                                              properties=('label','area','mean_intensity','weighted_centroid')))
        data = data.rename(columns={"mean_intensity": "mask_intensity"})
        

        data["target_intensity"] = (regionprops_table(label_image=mask_label,
                                                      intensity_image= target_raw,
                                                      properties=('label','mean_intensity')))['mean_intensity']


        data["size_fraction"] = regionprops_table(label_image=mask_label,
                                                          intensity_image= target_binary,
                                                          properties=('label','mean_intensity'))["mean_intensity"]
    
 

        #data ["target_area"] = regionprops_table(label_image=target_label,
        #                                         intensity_image = target_binary,
        #                                         properties = ('label'
       
        data["expressed"] = data.size_fraction.values > 0
        
        data["DF_expressed"] = (data["target_intensity"] *data["expressed"])
        
        data["DF_unexpressed"] = (data["target_intensity"]) - (data["target_intensity"] * (data["expressed"]))
        
        data["log10_DF_Intensity"] = np.log10(data["target_intensity"])
        
        n_vesicles = data.shape[0]
        n_markers = data.expressed.values.sum() 
        Fexp = data.expressed.values.mean()
        Fsize = data.size_fraction.values.mean()
        log10intensity = np.log10(data.target_intensity.values).mean()
        log2SBR = np.log2(data.target_intensity.values).mean()
        

        results = pd.DataFrame(np.array([n_vesicles,n_markers,Fexp,
                                         Fsize,log10intensity,log2SBR])).T
        
        results.columns = ['n_vesicles','n_markers','Fexp',
                           'Fsize','log10intensity','log2SBR']
        
        self.mask_label = mask_label
        self.target_label = target_label
        self.data = data
        self.results = results
        self.mask_lap = mask_lap
        self.target_lap = target_lap
        self.mask_raw = mask_raw
        self.target_raw = target_raw


