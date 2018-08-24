''' Script to identify, convolve, and mask the tapebump issue
Some remarks: 
- I wll use the high sky brightness sample, as this are my work cases
- The ,ask will be dilated, to not only comprise the tight region
'''

import os
import numpy as np
import scipy
import scipy.ndimage
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, ZScaleInterval)
import skimage.filters
import skimage.exposure


def gaussian(mu, sigma, x):
    return np.exp(np.power(-(x - mu), 2) / (2 * np.power(sigma, 2))) / (sigma * np.sqrt(2 * np.pi))

def gauss_filter(arr, sigma, order=0, mode='constant', cval=0):
    '''
    - order: 0 for Gaussian; 1, 2, 3 for Gaussian derivates
    - mode: options are reflect, constatn, nearest, mirror, wrap
    - cval: value for pading when mode=constant is selected
    '''
    gf = scipy.ndimage.filters.gaussian_filter(
        arr, sigma, order=order, mode=mode, cval=cval,
    )
    return gf

def dilate():
    ''' Dilation 
    '''
    from skimage import morphology
    morphology.binary_dilation(a, morphology.diamond(1)).astype(np.uint8)
    return

def connected_reg():
    ''' Get connected regions
    '''
    from skimage import measure
    all_labels = measure.label(blobs)
    blobs_labels = measure.label(blobs, background=0)

def meth01(mx):
    mx1 = gauss_filter(mx, 5, order=0) * gauss_filter(mx, 1, order=0)
    return mx1

def meth02(x):
    # The Otsu thresholding work well with the image when: 
    # K(sigma=5) * K(sigma=1)
    # Try with other tapebumps to see if the good behaviour persists
    aux_x = meth01(x)
    val_x = skimage.filters.threshold_otsu(aux_x)
    return aux_x < val_x

if __name__ == '__main__':
    fnm = 'brute_analysis/medImg/medImg_y4e1_a01b_c19_s1_g.npy'
    mx = np.load(fnm)
    # Discard the borders where some edge effects are presents
    mx = mx[10 : -9, 10 : -9]

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
    fig, ax = plt.subplots(1, 5, figsize=(12, 3))

    # Plot 1
    mx1 = meth01(mx)
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
    #
    # Plot 2
    mx2 = meth02(mx)
    kw2 = {
        'origin' : 'lower',
        'cmap' : 'viridis',
    }
    ax[1].imshow(mx2, **kw2)

    # Plot 3 
    # Histogram of the stamp
    hist, bins_cntr = skimage.exposure.histogram(mx)
    ax[2].plot(bins_cntr, hist, lw=0.8, c='goldenrod')
    
    plt.show()

