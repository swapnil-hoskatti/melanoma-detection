# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:25:09 2018

@author: Gavin
"""
import os
import matplotlib.pyplot as plt
import pickle

def segreg():
    #img_list = os.listdir(r".\all")
    
    csv_file = open('godfather.csv').read().split('\n')
    
    for line in csv_file[1:-1]:
        img_name, MEL, NV, BCC, AKIEC, BKL, DF, VASC = line.split(',')
        img_name = img_name + '.jpg'
        if MEL == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\MEL\{}".format(img_name))
        elif NV == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\NV\{}".format(img_name))    
        elif BCC == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\BCC\{}".format(img_name)) 
        elif AKIEC == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\AKIEC\{}".format(img_name))
        elif BKL == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\BKL\{}".format(img_name))
        elif DF == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\DF\{}".format(img_name))
        elif VASC == "1.0":
            os.rename(r".\all\{}".format(img_name), r".\all\VASC\{}".format(img_name))
            
def pickleem():
    folders = [x for x in os.listdir(r".\all") if f"{x}.pickle" not in os.listdir()]
    for folder in folders:
        temp = [plt.imread(r".\all\{}\{}".format(folder,x)) for x in os.listdir(r".\all\{}".format(folder))]
        pickle.dump(temp, open(f'{folder}.pickle', 'wb'))
        print(f"{folder} done")
        
    
pickleem()