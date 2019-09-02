# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 15:06:46 2018

@author: Rodrigo Doering Neves
"""


# Importando Bibliotecas


import cv2
import numpy as np
import pandas as pd
from skimage.measure import compare_ssim


# Métodos para detecção de movimentos

 
class Camera: 
    
    """    
    ABORDAGEM COM FRAME REFERENCIAL ATUALIZÁVEL
    
    """
    
    def __init__(self, name, video_id):        
        self.video_name = name
        self.cap = cv2.VideoCapture(self.video_name)
        self.path_out = "C:/Users/User/appcamera/busted"
        self.videoId = video_id

    def process_video(self):
        
        while True:
            self.cap.read()
            bol, frame = self.cap.retrieve(4)
            self.ref_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if True:                
                break
        
        curr_frame = None
        prev_frame = None
        first_frame = True
        self.video_fps = int(self.cap.get(5))
        self.video_range = int(self.cap.get(7))
        contador1 = 0
        contador2 = 0
        row_ssim = []
        row_mse = []
        row_movement = []
        row_update = []
        tempoIndex = []
        
        while True:           
            contador1 = contador1 + 1
            prev_frame = curr_frame
            _ , curr_frame = self.cap.read()
            
            if contador1 == self.video_range:                
                print('End of Video')
                break
            
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            if first_frame:                
                prev_frame = curr_frame
                first_frame = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break
            
            else:
                
                if contador1 % 15 == 0:                    
                    ssim_result = compare_ssim(self.ref_frame, curr_frame, data_range=curr_frame.max() - curr_frame.min())
                    mse_result = ((self.ref_frame - curr_frame) ** 2).mean()
                    calc1 = contador1 / self.video_fps
                    estimated_time = round(calc1, 2)
                    movement = None
                    update = False
                    row_ssim.append(ssim_result)
                    row_mse.append(mse_result)
                    row_movement.append(movement)
                    row_update.append(update)
                    tempoIndex.append(estimated_time)
                    
                    if ssim_result <= 0.9:                       
                        img_name = '/frame_{}_{}.jpg'.format(round(calc1,1), self.videoId)
                        cv2.imwrite(str(self.path_out + img_name), curr_frame)
                        row_movement.pop()
                        row_movement.append(True)
                        contador2 = contador2 + 1
                
                if contador2 == 7:                   
                    row_update.pop()
                    row_update.append(True)
                    self.ref_frame = curr_frame
                    contador2 = 0
                    
        df_dados = {'SSIM':row_ssim, 'MSE':row_mse, 'Movement':row_movement, 'Update_Ref':row_update}
        dataframe = pd.DataFrame(df_dados, index=tempoIndex)           
        return dataframe
                        

class Camera_v2:
    
    """
    ABORDAGEM FRAME A FRAME
    
    """
    
    def __init__(self, name, video_id, path_out):        
        self.video_name = name
        self.cap = cv2.VideoCapture(self.video_name)
        self.path_out = path_out
        self.videoId = video_id

    def process_video(self):        
        curr_frame = None
        prev_frame = None
        first_frame = True
        self.video_fps = int(self.cap.get(5))
        self.video_range = int(self.cap.get(7))
        contador1 = 0
        contador2 = 0
        row_ssim = []
        row_mse = []
        row_movement = []
        row_update = []
        tempoIndex = []
        
        while True:            
            contador1 = contador1 + 1
            prev_frame = curr_frame
            _ , curr_frame = self.cap.read()
            
            if contador1 == self.video_range:                
                print('FUCK YEAH')
                break
            
            curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            if first_frame:                
                prev_frame = curr_frame
                first_frame = False
            
            if cv2.waitKey(1) & 0xFF == ord('q'):                
                break
            
            else:
                
                if contador1 % 15 == 0:                    
                    ssim_result = compare_ssim(prev_frame, curr_frame, data_range=curr_frame.max() - curr_frame.min())
                    mse_result = ((prev_frame - curr_frame) ** 2).mean()
                    calc1 = contador1 / self.video_fps
                    estimated_time = round(calc1, 2)
                    movement = None
                    update = False
                    row_ssim.append(ssim_result)
                    row_mse.append(mse_result)
                    row_movement.append(movement)
                    tempoIndex.append(estimated_time)
                    
                    if mse_result >= 4 and calc1 > 0.5:                        
                        img_name = '/frame_{}_{}.jpg'.format(round(calc1,1), self.videoId)
                        cv2.imwrite(str(self.path_out + img_name), curr_frame)
                        row_movement.pop()
                        row_movement.append(True)
                        contador2 = contador2 + 1
                

                    
        df_dados = {'SSIM':row_ssim, 'MSE':row_mse, 'Movement':row_movement}
        dataframe = pd.DataFrame(df_dados, index=tempoIndex)           
        return dataframe 
    
