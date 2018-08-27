''' Script to identify, convolve, and mask the tapebump issue
Some remarks: 
- I wll use the high sky brightness sample, as this are my work cases
- The ,ask will be dilated, to not only comprise the tight region
'''

import os
import glob
import time
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, ZScaleInterval)
import skimage.filters
import skimage.exposure
from skimage import measure
import multiprocessing as mp

def gauss_filter(arr, sigma, order=0, mode='constant', cval=0):
    ''' Covolves with a Gaussian, with the given parameters
    - order: 0 for Gaussian; 1, 2, 3 for Gaussian derivates
    - mode: options are reflect, constatn, nearest, mirror, wrap
    - cval: value for pading when mode=constant is selected
    '''
    gf = scipy.ndimage.filters.gaussian_filter(
        arr, sigma, order=order, mode=mode, cval=cval,
    )
    return gf

def connected_reg(arr, threshold=0):
    ''' Simple but useful method to get connected regions. Get all regions
    and the subset above some threshold
    '''
    all_labels = measure.label(arr)
    sub_labels = measure.label(blobs, background=threshold)
    return all_labels, sub_labels

def otsu_threshold(arr):
    ''' Calculates the Otsu threshold value 
    '''
    return skimage.filters.threshold_otsu(arr)

def bin_dilation(mask, niter=1):
    ''' Binary dilation
    '''
    return scipy.ndimage.binary_dilation(mask, iterations=niter)

def stamp_stats():
    ''' Plot the simple stats generated from each one of the median images
    '''
    return

def gen_mask(fnm_arr, pix_border=10, sigma1=1, sigma2=10, dilat_n=10):
    ''' Method to contruct the mask for the stamps. Check what happens when
    no mask is created (good sections)
    Inputs
    - arr 
    - pix_border: how many pixels to discard at each stamp border, to avoid 
    strong slope
    - sigma1, sigma2
    - dilat_n: number of iterations for the dilation
    '''
    arr = np.load(fnm_arr)
    # Discard the borders where some edge effects are presents
    arr = arr[pix_border : -(pix_border - 1), pix_border : -(pix_border - 1)]
    # Apply a combined Gaussian kernel
    s1, s2 = 1, 10
    karr = (gauss_filter(arr, s1, order=0, mode='constant', cval=0) *
            gauss_filter(arr, s2, order=0, mode='constant', cval=0))
    # Over the kernelized image apply the Otsu threshold  
    val_otsu = otsu_threshold(karr)
    msk = karr > val_otsu
    print('Otsu={0:.3f} Std={1:.3f}'.format(val_otsu, np.std(karr)))
    # Dilate the mask 
    d_msk = bin_dilation(msk, niter=10)
    # After create the masks, check is need to discard small satellite masks
    #
    # Checking cases
    #
    if (np.abs(val_otsu) > 1):
        fig, ax = plt.subplots(1, 2)
        im_norm = ImageNormalize(karr, 
                                  interval=ZScaleInterval(),
                                  stretch=SqrtStretch(),)
        kw1 = {
            'norm' : im_norm,
            'origin' : 'lower',
            'cmap' : 'gray_r',
        }
        ax[0].imshow(karr, **kw1)
        ax[1].imshow(d_msk, origin='lower')
        ax[1].imshow(msk, origin='lower', alpha=0.5)
        plt.show()
    #
    #
    return karr, d_msk

def aux_main():
    path1 = 'medImg_a01/'
    path2 = 'medImg_a01_stat/'
    # Multiprocessing
    Px = mp.Pool(processes=mp.cpu_count())
    # Iteratively work by band
    for b in ['g', 'r', 'i', 'z', 'Y']:
        #for ccd in np.r_[[1], np.arange(2, 60 + 1), [62]]:
        
        t0 = time.time() 
        aux_path = os.path.join(path1, '{0}/c*/*npy'.format(b))
        fnm = glob.glob(aux_path, recursive=True)
        if False:
            # Get the masks for each one of them
            res = Px.map_async(gen_mask, fnm)
            # Control of the pool
            res.wait()
            res.successful()
            res = res.get()
        else:
            for f in fnm:
                ccd, s = 
                print('i')
                m = gen_mask(f)
        t1 = time.time()
        print('{0}-band, {1:.2f} min'.format(b, (t1 - t0) / 60))
    print ('---------')    

if __name__ == '__main__':
    
    aux_main()
    
    """
    fnm = 'medImg_a01/g/c19//medImg_y4e1_a01b_c19_s1_g.npy'
    arr = np.load(fnm)
    # Discard the borders where some edge effects are presents
    mx = arr[10 : -9, 10 : -9]

    # Image segmentation
    # IDEAS:
    # - denoising filter
    # - otsu thresholding
    # - watershed and random walker
    # - clustering, spectral, dbscan
    # medimg[np.where(medimg < 0.02)] = 0

    #
    # Do one method, one plot schema
    # ==============================
    #

    # Setup plot
    fig, ax = plt.subplots(1, 6, figsize=(12, 3))

    #
    # Sigma for the plots
    #
    sig1, sig2 = 5, 1
    sig3, sig4 = 10, 1

    # Plot 1
    mx1 = meth01(mx, sig1, sig2)
    im_norm1 = ImageNormalize(mx1, 
                             interval=ZScaleInterval(),
                             stretch=SqrtStretch(),)
    kw1 = {
        'norm' : im_norm1,
        'origin' : 'lower',
        'cmap' : 'gray_r',
    }
    im1 = ax[0].imshow(mx1, **kw1)
    # To locate colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    cbar = fig.colorbar(im1, cax=cax)
    cbar.ax.tick_params(labelsize=9)
    cax.yaxis.set_ticks_position('right')
    
    # Plot 2
    mx2 = meth01(mx, sig3, sig4)
    im_norm2 = ImageNormalize(mx2, 
                             interval=ZScaleInterval(),
                             stretch=SqrtStretch(),)
    kw2 = {
        'norm' : im_norm2,
        'origin' : 'lower',
        'cmap' : 'gray_r',
    }
    im2 = ax[1].imshow(mx2, **kw2)

    # Plot 3
    mx3, val3 = meth02(mx, sig1, sig2)
    kw3 = {
        'origin' : 'lower',
        'cmap' : 'viridis',
    }
    ax[2].imshow(mx3, **kw3)
    # Dilate the mask
    niter = 10
    d_mx3 = scipy.ndimage.binary_dilation(mx3, iterations=niter)
    ax[2].imshow(d_mx3, alpha=0.5, **kw3)
    
    # Plot 4
    mx4, val4 = meth02(mx, sig3, sig4)
    kw4 = {
        'origin' : 'lower',
        'cmap' : 'viridis',
    }
    ax[3].imshow(mx4, **kw4)
    # Dilate the mask
    niter = 10
    d_mx4 = scipy.ndimage.binary_dilation(mx4, iterations=niter)
    ax[3].imshow(d_mx4, alpha=0.5, **kw4)

    # Plot 5 
    # Histogram of the stamp
    we = np.ones_like(mx.flatten()) / mx.flatten().size
    ax[4].hist(mx.flatten(), color='b', histtype='step', 
                bins=100)
    ax[4].axvline(val3, lw=2, color='navy')
    ax[4].axvline(val4, lw=2, color='green')

    # Plot 6
    # Original stamp
    arr_norm = ImageNormalize(arr, 
                             interval=ZScaleInterval(),
                             stretch=SqrtStretch(),)
    kw1 = {
        'norm' : arr_norm,
        'origin' : 'lower',
        'cmap' : 'gray_r',
    }
    im1 = ax[5].imshow(arr, **kw1)


    plt.show()
    """
