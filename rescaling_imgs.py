# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 16:04:34 2018

@author: rodri
"""

import PIL
import os
from PIL import Image

# Ignorar por enquanto
#os.chdir('C:\imagens_site')
#os.chdir('C:\imagens_novo')

refHeight = 768
refWidth = 1024

ImageString = ['ap-1.jpg','ap-2.jpg','ap-3.jpg','ap-4.jpg','ap-5.jpg','ap-6.jpg','ap-7.jpg',
               'bg-1.jpg','bg-2.jpg','bg-3.jpg','bg-4.jpg','bg-5.jpg','bg5.jpg','bg-6.jpg',
               'emp-1.jpg','emp-2.jpg','emp-3.jpg','emp-4.jpg','new-1.jpg','new-2.jpg','new-4.jpg',
               'contact-3.png']



def resize_and_convert(imgList, imgFolder, imgOut, width):
     
     refWidth = width
     
     for i in range(len(imgList)):
         
         os.chdir(imgFolder)
         img = Image.open(imgList[i])
         wpercent = (refWidth / float(img.size[0]))
         heightSize = int((float(img.size[1]) * float(wpercent)))
         os.chdir(imgOut)
         new_img = img.resize((refWidth, heightSize), PIL.Image.ANTIALIAS)
         new_img.convert('RGB').save(imgList[i][:-4] + '.png', "PNG")
         print('{} re-scaled to {} x {} and saved to {}'.format(imgList[i], str(refWidth), str(heightSize), imgOut))
         

new_3 = ['contato-mobile.png']
resize_and_convert(new_3, 'C:\\Users\\rodri\\Desktop\\fotos site\\fotos editadas\\public_html\\images\\slider', 'C:\imagens_novo', 1024 )
