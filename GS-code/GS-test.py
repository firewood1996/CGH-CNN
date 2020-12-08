import math
import numpy as np
from numpy import *
from scipy.fftpack import *
import matplotlib.pyplot as plt
import cv2 as cv
import time
import numpy as np
from skimage import metrics
import math
import numpy as np


def Ger_Sax_algo(img, max_iter):
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        pm_s = np.angle(signal_s)
        signal_s = am_s*np.exp(pm_s * 1j)

    pm =pm_s
    return pm

def GS_2(img, max_iter):
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f = am_s*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm

def GS_22(img, max_iter,pm_s):
    h, w = img.shape
    
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f = am_s*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm

def GS_3(img, max_iter):
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.ifft2(signal_s)
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = np.fft.fft2(signal_f)
        pm_s = np.angle(signal_s)
        pm_ss=pm_s
        signal_s = am_s*np.exp(pm_s * 1j)

    pm =pm_f
    return pm

def GS_en(img, max_iter):
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    pm_M2=pm_f*(1/30)*pi
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        signal_f=signal_f*np.exp(pm_M2* 1j)
        signal_f=np.fft.fft2(signal_f)
        pm_f = np.angle(signal_f)
        signal_f = am_s*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        signal_s=signal_s*np.exp(pm_M2* 1j*(-1))
        signal_s=np.fft.ifft2(signal_s)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm



def GS_en_t(img, max_iter,pm_s):
    h, w = img.shape
    pm_f = np.ones((h, w))
    pm_M2=pm_f*(1/30)*pi
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        signal_f=signal_f*np.exp(pm_M2* 1j)
        signal_f=np.fft.fft2(signal_f)
        pm_f = np.angle(signal_f)
        signal_f = am_s*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f)
        signal_s=signal_s*np.exp(pm_M2* 1j*(-1))
        signal_s=np.fft.ifft2(signal_s)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm



def GS_en_t2(img, max_iter,pm_s):
    h, w = img.shape
    pm_f = np.ones((h, w))
    pm_M2=pm_f*(1/30)*pi
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f = np.fft.fft2(signal_s)
        signal_f=signal_f*np.exp(pm_M2* 1j)
        signal_f=np.fft.fft2(signal_f)
        pm_f = np.angle(signal_f)
        signal_f2 = am_s*np.exp(pm_f * 1j)
        signal_s = np.fft.ifft2(signal_f2)
        signal_s=signal_s*np.exp(pm_M2* 1j*(-1))
        signal_s=np.fft.ifft2(signal_s)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =signal_f2
    return pm
