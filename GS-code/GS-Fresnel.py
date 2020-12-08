from numpy import *
from scipy.fftpack import *
import matplotlib.pyplot as plt
import cv2 as cv
import time
import numpy as np
from skimage import metrics
import math
import numpy as np

##-------#注意逆菲涅尔衍射的k是负的

def ssim_F(img_mat_1, img_mat_2):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma=1.5
    gaussian_kernel_width=3
    gaussian_kernel=np.zeros((gaussian_kernel_width,gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j]=\
            (1/(2*pi*(gaussian_kernel_sigma**2)))*\
            exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))
 
    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1=img_mat_1.astype(np.float)
    img_mat_2=img_mat_2.astype(np.float)
    
    #Squares of input matrices
    img_mat_1_sq=img_mat_1**2
    img_mat_2_sq=img_mat_2**2
    img_mat_12=img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1=scipy.ndimage.filters.convolve(img_mat_1,gaussian_kernel)
    img_mat_mu_2=scipy.ndimage.filters.convolve(img_mat_2,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq=img_mat_mu_1**2
    img_mat_mu_2_sq=img_mat_mu_2**2
    img_mat_mu_12=img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq=scipy.ndimage.filters.convolve(img_mat_1_sq,gaussian_kernel)
    img_mat_sigma_2_sq=scipy.ndimage.filters.convolve(img_mat_2_sq,gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12=scipy.ndimage.filters.convolve(img_mat_12,gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq=img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq=img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12=img_mat_sigma_12-img_mat_mu_12;
    
    #c1/c2 constants
    #First use: manual fitting
    c_1=6.5025
    c_2=58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on L (width of color map)
    l=255
    k_1=0.01
    c_1=(k_1*l)**2
    k_2=0.03
    c_2=(k_2*l)**2
    
    #Numerator of SSIM
    num_ssim=(2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim=(img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map=num_ssim/den_ssim
    index=np.average(ssim_map)

 
    return index

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

    pm =pm_f
    return pm

def nor_0_1(x):
    """
    归一化函数
    :param u2_ang: 输入数据
    :return: 归一化数据
    """
    u_max = amax(x)
    u_min = amin(x)
    u = (x - u_min) / (u_max - u_min)

    return u
def DFFT(Ui, k, z,wave):
    '''
    菲涅耳衍射
    Ui:图片
    k:2*pi/波长
    z:透镜和衍射屏距离
    dxo&dyo:空间调制器的参数
    '''
    dxo = 8e-6
    dyo = 8e-6
    M, N = Ui.shape
    U_obj = zeros([M, M], dtype=complex)
    U_obj = Ui
    Lx = dxo * M
    Ly = dyo * N
    x = linspace(-Lx / 2, Lx / 2, M)
    y = linspace(-Ly / 2, Ly / 2, N)
    x, y = np.meshgrid(x, y)
    fx = x / (wave * z)
    fy = y / (wave * z)
    H = np.exp(1j * k * z * (1 - wave ** 2 / 2 * (fx ** 2 + fy ** 2)))
    U = fftshift(fft2(fftshift(U_obj))) * H
    U = ifftshift(ifft2(ifftshift(U)))
    U_crop=U
    return U

def FrT2(Ui, k, z,wave):
    '''
    菲涅耳衍射
    Ui:图片
    k:2*pi/波长
    z:透镜和衍射屏距离
    dxo&dyo:空间调制器的参数
    '''
    dxo = 8e-6
    dyo = 8e-6
    M, N = Ui.shape
    U_obj = zeros([M, M], dtype=complex)
    U_obj = Ui
    Lx = dxo * M
    Ly = dyo * N
    x = linspace(-Lx / 2, Lx / 2, M)
    y = linspace(-Ly / 2, Ly / 2, N)
    x, y = np.meshgrid(x, y)
    fx = x / (wave * z)
    fy = y / (wave * z)
    H = exp(1j * k * z * (1 - wave ** 2 / 2 * (fx ** 2 + fy ** 2)))
    U =np.fft.fft2(U_obj) * H
    U = np.fft.ifft2(U)
    U_crop=U
    return U


def IDFFT(Ui, k, z,wave):
    '''
    逆菲涅耳衍射
    Ui:图片
    k:2*pi/波长
    z:透镜和衍射屏距离
    dxo&dyo:空间调制器的参数
    '''
    dxo = 8e-6
    dyo = 8e-6
    M, N = Ui.shape
    U_obj = zeros([M, M], dtype=complex)
    #U_obj[int(M / 4):int(M / 4 * 3), int(N / 4):int(N / 4 * 3)] = Ui
    U_obj = Ui
    Lx = dxo * M
    Ly = dyo * N
    x = linspace(-Lx / 2, Lx / 2, M)
    y = linspace(-Ly / 2, Ly / 2, N)
    x, y = np.meshgrid(x, y)
    fx = x / (wave * z)
    fy = y / (wave * z)
    H = np.exp(1j * k * z * (1 - wave ** 2 / 2 * (fx ** 2 + fy ** 2)))
    U = fftshift(fft2(fftshift(U_obj))) * H
    U = ifftshift(ifft2(ifftshift(U)))
    U_crop=U
    return U_crop


def IFrT2(Ui, k, z,wave):
    '''
    逆菲涅耳衍射
    Ui:图片
    k:2*pi/波长
    z:透镜和衍射屏距离
    dxo&dyo:空间调制器的参数
    '''
    dxo = 8e-6
    dyo = 8e-6
    M, N = Ui.shape
    U_obj = zeros([M, M], dtype=complex)
    #U_obj[int(M / 4):int(M / 4 * 3), int(N / 4):int(N / 4 * 3)] = Ui
    U_obj = Ui
    Lx = dxo * M
    Ly = dyo * N
    x = linspace(-Lx / 2, Lx / 2, M)
    y = linspace(-Ly / 2, Ly / 2, N)
    x, y = np.meshgrid(x, y)
    fx = x / (wave * z)
    fy = y / (wave * z)
    H = exp(1j * k * z * (1 - wave ** 2 / 2 * (fx ** 2 + fy ** 2)))
    U =np.fft.fft2(U_obj) * H
    U = np.fft.ifft2(U)
    U_crop=U
    return U_crop



def GS_FrT(img, max_iter,k,z,wave):##不包含密钥，只进行一次菲涅耳衍射，相位首先和原图相乘
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))

    signal_s = am_s*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f =IDFFT(signal_s,-k,z,wave)
        pm_f = np.angle(signal_f)
        signal_f = am_f*np.exp(pm_f * 1j)
        signal_s = DFFT(signal_f,k,z,wave)
        pm_s = np.angle(signal_s)
        pm_ss=pm_s
        signal_s = am_s*np.exp(pm_s * 1j)

    pm =pm_f
    return pm

def GS_FrT_2(img, max_iter,k,z,wave):##不包含密钥，只进行一次菲涅耳衍射,相位首先和全1矩阵相乘
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    pm_f = np.ones((h, w))
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
   
    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        pm_f = np.angle(signal_f)
        signal_f = am_s*np.exp(pm_f * 1j)
        signal_s = IDFFT(signal_f,-k,z,wave)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm

def GS_FrT_En(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M2=np.exp(pm_M2 * 1j)
    M2_c=M2.conjugate()
    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        pm_fn=signal_f*M2
        signal_fn=DFFT(pm_fn,k,z,wave)
        pm_fn_u=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_fn_u * 1j)
        pm_fn = IDFFT(signal_fn_u,-k,z,wave)
        signal_f=pm_fn*M2_c
        signal_s = IDFFT(signal_f,-k,z,wave)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm




def GS_FrT_En_2(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #菲涅耳衍射之后，做一个fft再乘M2加密相位
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M2=np.exp(pm_M2 * 1j)
    M2_c=M2.conjugate()
    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        signal_f_fft=np.fft.fft2(signal_f)
        pm_fn_fft=signal_f*M2
        pm_fn=np.fft.ifft2(pm_fn_fft)
        signal_fn=DFFT(pm_fn,k,z,wave)
        pm_fn_u=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_fn_u * 1j)
        pm_fn = IDFFT(signal_fn_u,-k,z,wave)
        pm_fn_fft=np.fft.fft2(pm_fn)
        signal_f_fft=pm_fn_fft*M2_c
        signal_f=np.fft.ifft2(signal_f_fft)
        signal_s = IDFFT(signal_f,-k,z,wave)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return pm


def GS_FrT_En_3(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #菲涅耳衍射之后，做一个fft再乘M2加密相位
                                        #M2换成随机的
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2=np.random.rand(h, w)
    M2=np.exp(pm_M2 * 1j)
    M2_c=M2.conjugate()
    signal_s = am_f*np.exp(pm_s * 1j)

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        signal_f_fft=np.fft.fft2(signal_f)
        pm_fn_fft=signal_f*M2
        pm_fn=np.fft.ifft2(pm_fn_fft)
        signal_fn=DFFT(pm_fn,k,z,wave)
        pm_fn_u=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_fn_u * 1j)
        pm_fn = IDFFT(signal_fn_u,-k,z,wave)
        pm_fn_fft=np.fft.fft2(pm_fn)
        signal_f_fft=pm_fn_fft*M2_c
        signal_f=np.fft.ifft2(signal_f_fft)
        signal_s = IDFFT(signal_f,-k,z,wave)
        pm_s = np.angle(signal_s)
        signal_s = am_f*np.exp(pm_s * 1j)

    pm =pm_s
    return 
    

def GS_FrT_En_4(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #分开取模
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        pm_fn=signal_f*M2
        signal_fn=DFFT(pm_fn,k,z,wave)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = IDFFT(signal_fn_u,-k,z,wave)

        signal_f=pm_fn*M2_c
        s_f_p=signal_f/abs(signal_f)

        s_s = IDFFT(s_f_p,-k,z,wave)
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm


def GS_FrT_En_5(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        pm_fn=signal_f*M2
        signal_fn=DFFT(pm_fn,k,z,wave)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = IDFFT(signal_fn_u,-k,z,wave)

        signal_f=pm_fn*M2_c


        s_s = IDFFT(signal_f,-k,z,wave)
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    
    return pm


def GS_FrT_En_6(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
                                        #消除M2的影响
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =FrT2(signal_s,k,z,wave)
        pm_fn=signal_f*M2
        signal_fn=FrT2(pm_fn,k,z,wave)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = IFrT2(signal_fn_u,-k,z,wave)

        signal_f=pm_fn*M2_c
        signal_f=signal_f/abs(M2_c)

        s_s = IFrT2(signal_f,-k,z,wave)
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm


#pm_M2=np.random.random(size=(h, w))
def GS_FrT_En_7(img, max_iter,k,z,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z,wave)
        pm_fn=signal_f*M2
        signal_fn=DFFT(pm_fn,k,z,wave)

        ps1=np.angle(signal_fn)
        s_fn_p=np.exp(ps1 * 1j)


        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = IDFFT(signal_fn_u,-k,z,wave)

        signal_f=pm_fn*M2_c


        s_s = IDFFT(signal_f,-k,z,wave)
        pM1=np.angle(s_s)
        M1=np.exp(pM1 * 1j)
        signal_s = am_f*M1

    pm =M1
    
    return pm



#M2采用随机相位，不再是一个固定的值
#用两个不同的距离
def GS_FrT_En_8(img, max_iter,k,z1,z2,wave):##包含密钥，进行两次菲涅耳衍射,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    #pm_M2 = np.ones((h, w))
    #pm_M2=pm_M2*(1/3)*pi
    pm_M2=np.random.random(size=(h, w))
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =DFFT(signal_s,k,z1,wave)
        pm_fn=signal_f*M2
        signal_fn=DFFT(pm_fn,k,z2,wave)

        ps1=np.angle(signal_fn)
        s_fn_p=np.exp(ps1 * 1j)


        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = IDFFT(signal_fn_u,-k,z2,wave)

        signal_f=pm_fn*M2_c


        s_s = IDFFT(signal_f,-k,z1,wave)
        pM1=np.angle(s_s)
        M1=np.exp(pM1 * 1j)
        signal_s = am_f*M1

    pm =M1
    
    return pm



def GS_fft_En(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
                                        #消除M2的影响
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =np.fft.fft2(signal_s)
        pm_fn=signal_f*M2
        signal_fn=np.fft.fft2(pm_fn)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn =  np.fft.ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c
        signal_f=signal_f/abs(M2_c)

        s_s =  np.fft.ifft2(signal_f )
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm


def GS_fft_En_2(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
                                        #消除M2的影响
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =np.fft.fft2(signal_s)
        pm_fn=signal_f*M2
        signal_fn=np.fft.fft2(pm_fn)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn =  np.fft.ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c


        s_s =  np.fft.ifft2(signal_f )
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm



def GS_fft_En_3(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #不取模取相位
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =np.fft.fft2(signal_s)
        En_fn=signal_f*M2
        signal_fn=np.fft.fft2(En_fn)
        pm_f=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_f * 1j)#与图片幅值相乘

        pm_fn =  np.fft.ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c


        s_s =  np.fft.ifft2(signal_f )
        pm_M1=np.angle(s_s)
        M1=np.exp(pm_M1* 1j)
        signal_s = am_f*M1

    pm =M1
    return pm



def GS_fft_En_4(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #不取模取相位
                                        #fft2
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =fft2(signal_s)
        En_fn=signal_f*M2
        signal_fn=fft2(En_fn)
        pm_f=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_f * 1j)#与图片幅值相乘

        pm_fn = ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c


        s_s =ifft2(signal_f )
        pm_M1=np.angle(s_s)
        M1=np.exp(pm_M1* 1j)
        M11=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm



def GS_fft_En_5(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #按步骤取模
                                        #消除M2的影响
                                        #fft2
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =fft2(signal_s)
        pm_fn=signal_f*M2
        signal_fn=fft2(pm_fn)
        s_fn_p=signal_fn/abs(signal_fn)
        signal_fn_u = am_s*s_fn_p#与图片幅值相乘

        pm_fn = ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c


        s_s = ifft2(signal_f )
        M1=s_s/abs(s_s)
        signal_s = am_f*M1

    pm =M1
    return pm


def GS_fft_En_6(img, max_iter):##包含密钥，进行两次fft,相位首先和全1矩阵相乘
                                        #不把相位单独提出，参照论文的方法
                                        #不取模取相位
    h, w = img.shape
    pm_s = np.random.rand(h, w)
    
    am_s = np.sqrt(img)
    am_f = np.ones((h, w))
    pm_M2 = np.ones((h, w))
    pm_M2=pm_M2*(1/3)*pi
    M1=np.exp(pm_s * 1j)#用作迭代
    M2=np.exp(pm_M2 * 1j)#用作加密
    M2_c=M2.conjugate()#用作解密
    signal_s = am_f*M1

    for iter in range(max_iter):
        signal_f =np.fft.fft2(signal_s)
        En_fn=signal_f*M2
        signal_fn=np.fft.fft2(En_fn)
        pm_f=np.angle(signal_fn)
        signal_fn_u = am_s*np.exp(pm_f * 1j)#与图片幅值相乘

        pm_fn =  np.fft.ifft2(signal_fn_u)

        signal_f=pm_fn*M2_c


        s_s =  np.fft.ifft2(signal_f )
        pm_M1=np.angle(s_s)
        M1=np.exp(pm_M1* 1j)
        signal_s = am_f*M1

    pm =signal_fn
    return pm
