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
from skimage import measure

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

def connected_reg(arr, background=0):
    ''' Simple but useful method to get connected regions. 
    '''
    all_labels = measure.label(arr)
    sub_labels = measure.label(blobs, background=0)

def meth01(mx, sigma1, sigma2):
    mx1 = gauss_filter(mx, sigma1, order=0) * gauss_filter(mx, sigma2, order=0)
    return mx1

def meth02(x, sigma1, sigma2):
    # The Otsu thresholding work well with the image when: 
    # K(sigma=5) * K(sigma=1)
    # Try with other tapebumps to see if the good behaviour persists
    aux_x = meth01(x, sigma1, sigma2)
    val_x = skimage.filters.threshold_otsu(aux_x)
    return aux_x > val_x, val_x
    
def meth04(mx, ):
    from sklearn.feature_extraction import image
    from sklearn.cluster import spectral_clustering
    # Instead of decrease resolution, use the kernelized image
    mx3 = gauss_filter(mx, 3, order=0)
    mx3 = np.copy(mx3)[:100, :100]
    # mx3 = mx3[::2, ::2] + mx3[1::2, ::2] + mx3[::2, 1::2] + mx3[1::2, 1::2]
    # mx3 = mx3[::2, ::2] + mx3[1::2, ::2] + mx3[::2, 1::2] + mx3[1::2, 1::2]
    # Convert the image into a graph with the value of the gradient on the
    # edges.
    graph = image.img_to_graph(mx3)
    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # actual image. For beta=1, the segmentation is close to a voronoi
    beta = 5 #1
    eps = 1e-6
    graph.data = np.exp(-beta * graph.data / mx3.std()) + eps
    # Apply spectral clustering (this step goes much faster if you have pyamg
    # installed)
    N_REGIONS = 3
    # Image normalization
    im_norm3 = ImageNormalize(mx3, 
                              interval=ZScaleInterval(),
                              stretch=SqrtStretch(),)
    for assign_labels in ('kmeans', 'discretize'):
        labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                     assign_labels=assign_labels,
                                     random_state=1)
        labels = labels.reshape(mx3.shape)                                               
        plt.imshow(mx3, cmap=plt.cm.gray, origin='lower', norm=im_norm3)
        for l in range(N_REGIONS):
            plt.contour(labels == l, contours=1,
                        colors=[plt.cm.spectral(l / float(N_REGIONS)), ])

if __name__ == '__main__':
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

