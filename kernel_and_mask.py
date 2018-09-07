''' Script to identify, convolve, and mask the tapebump issue
Some remarks:
- I wll use the high sky brightness sample, as this are my work cases
- The ,ask will be dilated, to not only comprise the tight region

BPM definition of bits
BPMDEF_FLAT_MIN     1
BPMDEF_FLAT_MAX     2
BPMDEF_FLAT_MASK    4
BPMDEF_BIAS_HOT     8
BPMDEF_BIAS_WARM   16
BPMDEF_BIAS_MASK   32
BPMDEF_BIAS_COL    64
BPMDEF_EDGE       128
BPMDEF_CORR       256
BPMDEF_SUSPECT    512
BPMDEF_FUNKY_COL 1024
BPMDEF_WACKY_PIX 2048
BPMDEF_BADAMP    4096
BPMDEF_NEAREDGE  8192
BPMDEF_TAPEBUMP 16384
'''

import os
import glob
import time
import logging
import numpy as np
import pandas as pd
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

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

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

def gen_mask(arr, mask, band=None, ccd=None, section=None,
             min_otsu=None,
             max_area=0.7,
             pix_border=10, sigma1=1, sigma2=10, dilat_n=10,
             do_plot=False):
    ''' Method to contruct the mask for the stamps. Check what happens when
    no mask is created (good sections)
    Inputs
    - arr: filename or array object
    - mask: mask derived from immask image
    - min_otsu: minimum threshold to be used for the Otsu value
    - pix_border: how many pixels to discard at each stamp border, to avoid
    strong slope
    - sigma1, sigma2
    - dilat_n: number of iterations for the dilation
    '''
    # To mange both filename or array as input
    if (type(arr) == np.ndarray):
        pass
    elif isinstance(arr, str):
        arr = np.load(arr)
    # Apply the mask
    arr = np.ma.masked_where(mask, arr)
    # Discard the borders where some edge effects are presents
    # arr = arr[pix_border : -(pix_border - 1), pix_border : -(pix_border - 1)]
    #
    # NOTE: the Gaussian filter does not work with masked arrays. I need to
    # make zero the masked regions. NaN doesn't get a good result
    arr[np.ma.getmask(arr)] = 0
    # Apply a combined Gaussian kernel
    karr = (gauss_filter(arr, sigma1, order=0, mode='constant', cval=0) *
            gauss_filter(arr, sigma2, order=0, mode='constant', cval=0))
    # Over the kernelized image apply the Otsu threshold
    val_otsu = otsu_threshold(karr)
    #
    # Try different thresholds
    #
    rms = (np.sqrt(np.mean(np.square(karr.flatten()))))
    mad = np.median(np.abs(karr-np.median(karr)))
    msk_otsu1 = karr > 0.9 * val_otsu
    #
    # TEST FOR R-BAND
    # After get a first threshold, get a second one, below the initial,
    # to separate 'the holes' in the image, whicha re aldo part of the
    # tapebumps
    if False:
        karr_tmp = np.copy(karr)
        karr_tmp[msk_otsu] = val_otsu
        val_otsu_tmp = otsu_threshold(karr_tmp)
        print('Apply the mask for above and below both threshold values')
        msk_otsu_tmp = karr_tmp < val_otsu_tmp
        msk_otsu_dual = np.logical_or(msk_otsu, msk_otsu_tmp)
        msk_otsu = msk_otsu_dual
        #
        #
        #
    msk_otsu2 = karr > 0.8 * val_otsu
    msk_mad = karr > mad
    msk_rms = karr > 3 * rms
    #
    print('Otsu={0:.3f} Std={1:.3f}'.format(val_otsu, np.std(karr)))
    # Dilate the mask
    d_msk_otsu1 = bin_dilation(msk_otsu1, niter=dilat_n)
    d_msk_otsu2 = bin_dilation(msk_otsu2, niter=dilat_n)
    d_msk_mad = bin_dilation(msk_mad, niter=dilat_n)
    d_msk_rms = bin_dilation(msk_rms, niter=dilat_n)
    # After create the masks, check is need to discard small satellite masks
    #
    # Divide in 2 groups: easy and difficult
    #

    #
    # Check sets of median images
    #
    cont = True
    area = d_msk_otsu1[np.where(d_msk_otsu1)].size / karr.size
    if (0.9 * val_otsu >= min_otsu) and (area <= max_area) and (do_plot):
        #True: #((np.abs(val_otsu) <= 1) or force_plot):
        tmp_karr_otsu1 = np.ma.masked_where(d_msk_otsu1, karr)
        tmp_karr_otsu2 = np.ma.masked_where(d_msk_otsu2, karr)
        tmp_karr_mad = np.ma.masked_where(d_msk_mad, karr)
        tmp_karr_rms = np.ma.masked_where(d_msk_rms, karr)
        #
        plt.close('all')
        fig, ax = plt.subplots(3, 3)
        #
        im_norm = ImageNormalize(karr,
                                 interval=ZScaleInterval(),
                                 stretch=SqrtStretch(),)
        # Different cuts
        # x times Otsu
        kw1 = {
            'norm': im_norm,
            'origin': 'lower',
            'cmap': 'gray_r',
        }
        ax[0, 0].imshow(karr, **kw1)
        ax[0, 1].imshow(d_msk_otsu1, origin='lower')
        ax[0, 1].imshow(msk_otsu1, origin='lower', alpha=0.5, cmap='gray')
        ax[0, 2].imshow(tmp_karr_otsu1, **kw1)
        ax[0, 0].set_title('0.9 Otsu={0:.4f}'.format(val_otsu))
        # x times Otsu
        im_norm2 = ImageNormalize(karr,
                                  interval=ZScaleInterval(),
                                  stretch=SqrtStretch(),)
        kw2 = {
            'norm': im_norm2,
            'origin': 'lower',
            'cmap': 'gray_r',
        }
        ax[1, 0].imshow(karr, **kw2)
        ax[1, 1].imshow(d_msk_otsu2, origin='lower')
        ax[1, 1].imshow(msk_otsu2, origin='lower', alpha=0.5, cmap='gray')
        ax[1, 2].imshow(tmp_karr_otsu2, **kw2)
        ax[1, 0].set_title('0.8 Otsu={0:.4f}'.format(3 * val_otsu))
        # RMS
        im_norm3 = ImageNormalize(karr,
                                  interval=ZScaleInterval(),
                                  stretch=SqrtStretch(),)
        kw3 = {
            'norm' : im_norm3,
            'origin' : 'lower',
            'cmap' : 'gray_r',
        }
        ax[2, 0].imshow(karr, **kw3)
        ax[2, 1].imshow(d_msk_rms, origin='lower')
        ax[2, 1].imshow(msk_rms, origin='lower', alpha=0.5, cmap='gray')
        ax[2, 2].imshow(tmp_karr_rms, **kw3)
        ax[2, 0].set_title('3 RMS={0:.5f}'.format(3 * rms))
        #
        plt.subplots_adjust(hspace=0.5)
        txt1 = '{0}-band, CCD {1:02}, {2},'.format(band, ccd, section)
        txt1 += ' area: {0:.2f}, niter={1}'.format(area, dilat_n)
        plt.suptitle(txt1, color='blue')
        #
        outdir = os.path.join('plot_a01_masks', band)
        if not os.path.exists(outdir):
            os.makedirs(outdir)
            logging.warning('Creating directory: {0}'.format(outdir))
        outnm = 'mask_{0}_c{1:02}_{2}_09otsu.pdf'.format(band, ccd, section)
        outnm = os.path.join(outdir, outnm)
        try:
            plt.savefig(outnm, dpi=300, format='pdf')
        except:
            raise
        # plt.show()
    return d_msk_otsu1, band, ccd, section, val_otsu, area, rms, mad

def bit2decompose(k_int):
    ''' Method receives an integer and split it in its base-2 composing bits
    '''
    base2 = []
    z = 1
    while (z <= k_int):
        if (z & k_int):
            base2.append(z)
        # Shift bits to the left 1 place
        z = z << 1
    return base2

def aux_main(write_tab=False):
    path1 = 'medImg_a01/'
    path2 = 'medImg_a01_stat/'
    path3 = 'mask_stamps/'
    path4 = 'mask_products/'
    # Multiprocessing
    # Px = mp.Pool(processes=mp.cpu_count())
    #
    # Iteratively work by the triplet of band-ccd-tape
    bcs = [[x, y, z] for x in ['g', 'r', 'i', 'z', 'Y']
           for y in np.r_[[1], np.arange(3, 60 + 1), [62]]
           for z in ['s1', 's2', 's3', 's4', 's5', 's6']]
    # Remove CCD31 tapebumps: s2, s3, s6
    for b in ['g', 'r', 'i', 'z', 'Y']:
        bcs.remove([b, 31, 's2'])
        bcs.remove([b, 31, 's3'])
        bcs.remove([b, 31, 's6'])
    # Levels for each band
    b_otsu = {'g': 0.003, 'r': 0.05, 'i': 0.07, 'z': 0.09, 'Y': 0.09}
    b_area = {'g': 0.9, 'r': 0.7, 'i': 0.7, 'z': 0.6, 'Y': 0.6}
    msk_info = []
    for tri in bcs:
        # [['r', 50, 's5'], ['r', 51, 's3'], ['r', 60, 's5']]:
        # [['g', 1, 's2'], ['g', 14, 's4'], ['g', 40, 's3'], ['g', 40, 's6']]:
        b, ccd, sx = tri
        if True: #b == 'i':
            # b, ccd, sx = 'i', 40, 's3'
            #
            # Construct the regex to look for immask masks, then get the unique
            # element
            regx3 = '{0}/c{1:02}/*_{2}_*'.format(b, ccd, sx)
            aux_path3 = os.path.join(path3, regx3)
            logging.info('Regex to look for immask-mask: {0}'.format(aux_path3))
            fnm3 = glob.glob(aux_path3, recursive=True)
            if ((len(fnm3) > 1) or (len(fnm3) == 0)):
                logging.error('Not an unique immask-mask was found')
            else:
                fnm3 = fnm3[0]
            # Construct the regex to look for the median images
            regx1 = '{0}/c{1:02}/*a01b_c{1:02}_{2}_*'.format(b, ccd, sx)
            aux_path1 = os.path.join(path1, regx1)
            logging.info('Regex to look for med images: {0}'.format(aux_path1))
            fnm1 = glob.glob(aux_path1, recursive=True)
            if (len(fnm1) == 0):
                logging.error('No med images found using {0}'.format(aux_path1))
            # Open the immask-mask and identify the problematic bits. Map them
            # and pass the mask to the mask generation method
            immask_msk = np.load(fnm3)
            #
            # Unique bits, just for comparison
            ubit = map(bit2decompose, np.unique(immask_msk.flatten()))
            ubit = list(ubit)
            #
            dbits = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048]
            # [1, 2, 512, 1024, 2048]
            # [1, 2, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
            aux_mask = np.zeros_like(immask_msk).astype('bool')
            for bit in dbits:
                aux_mask[np.where(bit & immask_msk)] = True
            # Call all the median images iteratively, with an unique immask-msk
            # for each set of band-ccd-tape
            for filex in fnm1:
                # This call can be parallelizable once works without problems
                res_aux = gen_mask(
                    filex, aux_mask,
                    band=b, ccd=ccd, section=sx,
                    sigma1=1.5, sigma2=6,
                    max_area=b_area[b],
                    min_otsu=b_otsu[b],
                    dilat_n=10,
                    do_plot=True,
                )
                # Split the results in the generated mask and the stats
                msk_info.append(res_aux[1:])
                # If selected, write out the mask in numpy binary format
                area_x = res_aux[5]
                otsu_x = res_aux[4]
                criteria1 = area_x < b_area[b]
                criteria2 = 0.9 * otsu_x > b_otsu[b]
                if (criteria1 and criteria2):
                    # Outname
                    o = 'a01b_{0}_c{1:02}_{2}.npy'.format(b, ccd, sx)
                    o = os.path.join(path4, o)
                    if not os.path.exists(path4):
                        raise
                    # Write out the numpy mask
                    x_msk = res_aux[0]
                    np.save(o, x_msk)
                    #
    # Write out the results from the masking, one table per band
    # Each entry has: band, ccd, section, val_otsu, area, rms, mad
    df = pd.DataFrame(
        msk_info, 
        columns=['band', 'ccd', 'section', 'otsu', 'area', 'rms', 'mad'],
    )
    # tmp_band = list(zip(*msk_info))
    # df = pd.DataFrame({'band': tmp_band[0], 'ccd': tmp_band[1],
    #                    'section': tmp_band[2], 'otsu': tmp_band[3],
    #                    'area': tmp_band[4], 'rms': tmp_band[5],
    #                    'mad': tmp_band[6]})
    if write_tab:
        outres = 'otsu_masking.csv'
        df.to_csv(outres, index=False, header=True)

    if False:
        #
        # Test on a special bad case: i-band, ccd 4, s3. Done! worked well
        #
        # First of all, check the mask and the image. Remember is one mask per
        # ccd-band, based on the first exposure of the queue used to generate
        # the median image
        b, ccd, sx= 'i', 4, 's3'
        aux_path3 = os.path.join(path3,
                                 '{0}/c{1:02}/*_{2}_*'.format(b, ccd, sx))
        #
        fnm1 = 'medImg_a01/i/c04/medImg_y4e1_a01b_c04_s3_i.npy'
        fnm3 = glob.glob(aux_path3, recursive=True)
        fnm3 = fnm3[0]
        #
        # Open median image
        mi_x = np.load(fnm1)
        #
        # Open the mask and identify the different bits
        msk_x = np.load(fnm3)
        aux_msk_x = map(bit2decompose, np.unique(msk_x.flatten()))
        aux_msk_x = list(aux_msk_x)
        # Dangerous bits
        # Get the positions to be masked
        dbits = [1, 128, 1024, 2048]
        aux_msk = np.zeros_like(mi_x).astype('bool')
        for bit in dbits:
            aux_msk[np.where(bit & msk_x)] = True
        # Create the masked version of the median image
        # Works well when plotted
        mi_masked = np.ma.masked_where(aux_msk, mi_x)
        # Feed the Otsu method with the masked array
        tmp = gen_mask(mi_masked)
        #
        exit()
        # Flatten the 2 levels list
        f_flat = lambda x: [i for sublist in x for i in sublist]
        unique_msk_x = f_flat(aux_msk_x)
        # Remove duplicates
        unique_msk_x = np.unique(np.array(unique_msk_x))
        exit()

    if False:
        exit()
        # First of all, check the mask and the image. Remember is one mask per
        # ccd-band, based on the first exposure of the queue used to generate
        # the median image
        b, ccd, sx= 'i', 4, 's3'
        aux_path3 = os.path.join(path3, '{0}/c{1:02}/*_{2}_*'.format(b, ccd, sx))

        fnm1 = 'medImg_a01/i/c04/medImg_y4e1_a01b_c04_s3_i.npy'
        fnm3 = glob.glob(aux_path3, recursive=True)
        fnm3 = fnm3[0]

        # Open median image
        mi_x = np.load(fnm1)

        # Open the mask and identify the different bits
        msk_x = np.load(fnm3)
        aux_msk_x = map(bit2decompose, np.unique(msk_x.flatten()))
        aux_msk_x = list(aux_msk_x)
        # Dangerous bits
        # Get the positions to be masked
        dbits = [1, 128, 1024, 2048]
        aux_msk = np.zeros_like(mi_x).astype('bool')
        for bit in dbits:
            aux_msk[np.where(bit & msk_x)] = True
        # Create the masked version of the median image
        # Works well when plotted
        mi_masked = np.ma.masked_where(aux_msk, mi_x)

        # Feed the Otsu method with the masked array
        tmp = gen_mask(mi_masked)

        exit()

        # for ccd in np.r_[[1], np.arange(2, 60 + 1), [62]]:
        t0 = time.time()
        aux_path = os.path.join(path1, '{0}/c*/*npy'.format(b))
        fnm = glob.glob(aux_path, recursive=True)
        if False:
            print('Multiprocessing')
            # Get the masks for each one of them
            res = Px.map_async(gen_mask, fnm)
            # Control of the pool
            res.wait()
            res.successful()
            res = res.get()
        else:
            for f in fnm:
                # ccd, s =
                # print('i')
                m = gen_mask(f)
        t1 = time.time()
        print('{0}-band, {1:.2f} min'.format(b, (t1 - t0) / 60))
        print ('---------')


if __name__ == '__main__':

    aux_main()
