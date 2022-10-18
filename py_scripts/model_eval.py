import glob
import os
import sys
import csv

import random
from tkinter import W
from turtle import pos
import numpy as np
import math
import time
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.image as mpllimg
from matplotlib.pyplot import figure
from Image_Sampler import Sampler
import cv2

import torch


class Evaluater:

    def __init__(self, model, device, path):
        self.model = model
        self.model.to(device)
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        self.device = device
        if self.device == "cpu": print("!= Warning: Device == CPU --> slow runtime =!")

    # input image size: w,h,3
    def predict(self, in_image):
        with torch.no_grad():
            img_tensor = np.transpose(in_image, (2,1,0))
            img_tensor = torch.as_tensor(np.array([img_tensor]))
            img_tensor = img_tensor.to(self.device)

            out = self.model(img_tensor)
            out = out[0].cpu().numpy()
            out = np.transpose(out, (2,1,0))

        return in_image, out

# ==============================================================================
# -- Visualizations ------------------------------------------------------------
# ==============================================================================

    def predictToHeatMap(self, input, output):
        #1 grayscale
        g_input = np.dot(input[...,:3], [0.2989, 0.5870, 0.1140])
        g_output = np.dot(output[...,:3], [0.2989, 0.5870, 0.1140]) # dims are now: w,h
        errormap = abs(g_input-g_output)
        
        #2 heatmap
        hm = np.zeros((input.shape[0],input.shape[1],3))
        hm = hm.astype("float32")
        for x in range(len(hm)):
            for y in range(len(hm[x])):
                if errormap[x][y] < 0.5:
                    hm[x][y][2] = 1
                    hm[x][y][1] = 2*errormap[x][y]
                    hm[x][y][0] = 2*errormap[x][y]
                else:
                    hm[x][y][0] = 1
                    hm[x][y][1] = 1 - 2*(errormap[x][y] - 0.5)
                    hm[x][y][2] = 1 - 2*(errormap[x][y] - 0.5)

        return hm, errormap #shape: hm=(w,h,3), errormap=(w,h)

    # returns the map of anomaly pixels
    def heatToDetection(self, errormap):
        
        # errormap = np.transpose(errormap, (2,1,0)) # new shape: (3,w,h)
        # errormap = errormap.mean(0) #avg of the 3 pixel values | new shape: (1,w,h)
        # errormap = errormap.squeeze()

        # mu_err = np.average(errormap)
        # flatten = errormap.reshape(-1)
        # median, lowerSplit, upperSplit = Evaluater.get_median_set(flatten)
        # median_Upper, _, upperSplit = Evaluater.get_median_set(upperSplit)
        # median_Lower, _, upperSplit = Evaluater.get_median_set(lowerSplit)
        # rangeIQR = median_Upper - median_Lower

        # baseline = 5.5 * rangeIQR + median_Upper

        baseline = 0.5 # best for a MAE


        errormap[errormap > baseline] = 1.0
        errormap[errormap <= baseline] = 0.0

        return errormap # shape: w,h

    # colors the pixels that are anomalys
    def colorDetectionMap(self, dtMap):
        detectionMap = np.zeros((3, dtMap.shape[0], dtMap.shape[1])).astype("float32")
        detectionMap[0][dtMap == 1.0] = 0.98
        detectionMap[1][dtMap == 1.0] = 0.55
        detectionMap[2][dtMap == 1.0] = 0.32

        return np.transpose(detectionMap, (1,2,0)) #shape: (w,h,3)
    
    def getColoredDetectionMap(self, input):
        dtMap = self.getDetectionMap(input)
        return self.colorDetectionMap(dtMap)

    def getHeatMap(self, input):
        _, output = self.predict(input)
        hm, errormap = self.predictToHeatMap(input, output)
        return hm
    
    def getDetectionMap(self, input):
        _, output = self.predict(input)
        hm, errormap = self.predictToHeatMap(input, output)
        return self.heatToDetection(errormap)

# ==============================================================================
# -- Distance measures ---------------------------------------------------------
# ==============================================================================

    def getAvgError(self, images, maxImages=None):
        if type(images) == str:
            images = Sampler.load_Images(images, size=maxImages).astype("float32") / 255
        
        mse_list = []
        mae_list = []
        for image in images:
            mse, mae = self.evalSinglePrediction(image)
            mse_list.append(mse)
            mae_list.append(mae)
        
        mse = np.average(np.array(mse_list))
        mae = np.average(np.array(mae_list))

        print(f"Average values among {len(mse_list)} images: MSE = {mse} | MAE = {mae}")
        return mse, mae
    
    def getBoxPlotOutlierBorder(self, images, maxImages=None, metric="mse"):
        if type(images) == str:
            images = Sampler.load_Images(images, size=maxImages).astype("float32") / 255
        
        error_list = []
        for image in images:
            mse, mae = self.evalSinglePrediction(image)
            if metric == "mse":
                error_list.append(mse)
            elif metric == "mae":
                error_list.append(mae)
        
        median, median_Lower, median_Upper, rangeIQR = Evaluater.generateBoxPlot(error_list)
        upperBorder = 1.5 * rangeIQR + median_Upper
        upperBorder = int(upperBorder * 100000)/ 100000.0
        print(f"Upper Border among {len(error_list)} images: {upperBorder} | Metric == {metric}")
        
        return upperBorder

    def getZScoreBorder(self, images, maxImages=None, metric="mse", border=0.999):
        if type(images) == str:
            images = Sampler.load_Images(images, size=maxImages).astype("float32") / 255
        
        error_list = []
        for image in images:
            mse, mae = self.evalSinglePrediction(image)
            if metric == "mse":
                error_list.append(mse)
            elif metric == "mae":
                error_list.append(mae)
        
        border = Evaluater.get_Z_score(error_list, border=border)
        border = int(border * 100000)/ 100000.0
        print(f"Upper Border among {len(error_list)} images: {border} | Metric == {metric}")

        return border

    def getMaxError(self, images, maxImages=None, metric="mse"):
        if type(images) == str:
            images = Sampler.load_Images(images, size=maxImages).astype("float32") / 255
        
        error_list = []
        for image in images:
            mse, mae = self.evalSinglePrediction(image)
            if metric == "mse":
                error_list.append(mse)
            elif metric == "mae":
                error_list.append(mae)
        
        error_list = np.array(error_list)
        maxError = max(error_list)
        print(f"Max error among {len(error_list)} images: {maxError} | Metric == {metric}")
        return maxError

    # returns mse and mae
    def evalSinglePrediction(self, in_image):
        input, output = self.predict(in_image)
        mse = self.getMSE(input, output)
        mae = self.getMAE(input, output)
        return mse, mae

    def getMSE(self, input, output):
        errorMatrix = (input-output) ** 2
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        errorAvg = int(errorAvg * 100000)/ 100000.0
        return errorAvg
    
    def getMAE(self, input, output):
        errorMatrix = abs(input-output)
        errorAvg = np.sum(errorMatrix) / (errorMatrix.shape[0] * errorMatrix.shape[1] * errorMatrix.shape[2])
        errorAvg = int(errorAvg * 100000)/ 100000.0
        return errorAvg


# ==============================================================================
# -- Static methods ------------------------------------------------------------
# ==============================================================================


    @staticmethod
    def generateBoxPlot(values):
        median, lowerSplit, upperSplit = Evaluater.get_median_set(values)
        median_Upper, _, upperSplit = Evaluater.get_median_set(upperSplit)
        median_Lower, _, upperSplit = Evaluater.get_median_set(lowerSplit)
        rangeIQR = median_Upper - median_Lower

        return median, median_Lower, median_Upper, rangeIQR

    # returns median and the smaller, bigger array
    @staticmethod
    def get_median_set(values):
        median = np.median(values)
        values = list(values)
        upperSplit = []
        lowerSplit = []
        for value in values:
            if value > median:
                upperSplit.append(value)
            else:
                lowerSplit.append(value)
        
        return median, np.array(lowerSplit), np.array(upperSplit)

    
    @staticmethod
    def get_Z_score(values, border=0.99):
        z = np.abs(stats.zscore(values))
        z = np.sort(z)
        x = int((1-border) * len(z))
        if x == 0: x = 1 # prevent empty output
        border = z[len(z) - x]

        return border


