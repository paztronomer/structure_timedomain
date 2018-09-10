''' Code for add a mask, per CCD, to the original Bad Pixel Masks 
Note the setup of this code to recognize filenames is VERY local to the actual
setup
Function-driven instead of Class, for simpler paralellisnm
'''

import os
import sys
import glob
import time
import logging
import argparse
import copy
import uuid
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
try:
    import matplotlib.pyplot as plt
except:
    pass
import fitsio

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

# Global variable, to load the set of bit definitions
BITDEF = None

def open_fits(fnm):
    ''' Open the FITS, read the data and store it on a list
    Inputs
    - fnm: filename
    Returns
    - data, header: copy of data and header
    '''
    tmp = fitsio.FITS(fnm)
    data, header = [], []
    for extension in tmp:
        ext_tmp = np.copy(extension.read())
        data.append(ext_tmp)
        header_tmp = copy.deepcopy(extension.read_header())
        header.append(header_tmp)
    tmp.close()
    if (len(data) == 0):
        logging.error('No extensions were found on {0}'.format(fnm))
        exit(1)
    elif (len(data) >= 1):
        return data, header

def load_bitdef(d1=None):
    ''' Load table of bit definitions and pass them to a global dictionary
    Inputs
    - d1: filename
    Returns
    - Boolean true
    '''
    global BITDEF
    try:
        kw = {
            'sep' : None,
            'comment' : '#', 
            'names' : ['def', 'bit'], 
            'engine' : 'python',
        }
        t1 = pd.read_table(d1, **kw)
    except:
        t_i = 'pandas{0} doesn\'t support guess sep'.format(pd.__version__) 
        logging.info(t_i)
        kw.update({'sep' : '\s+',})
        t1 = pd.read_table(d1, **kw)
    # Construct the dictionaries
    BITDEF = dict(zip(t1['def'], t1['bit']))
    # Change data type to unsigned integers, to match the dtype of the BPMs
    for k in BITDEF:
        BITDEF[k] = np.uint(BITDEF[k])
    return True 

def bit_count(int_type):
    ''' Function to count the amount of bits composing a number, that is the
    number of base 2 components on which it can be separated, and then
    reconstructed by simply sum them. Idea from Wiki Python.
    Brian Kernighan's way for counting bits. Thismethod was in fact
    discovered by Wegner and then Lehmer in the 60s
    This method counts bit-wise. Each iteration is not simply a step of 1.
    Example: iter1: (2066, 2065), iter2: (2064, 2063), iter3: (2048, 2047)
    Inputs
    - int_type: integer
    Output
    - counter with the number of base-2 numbers needed for the decomposition
    '''
    counter = 0
    while int_type:
        int_type &= int_type - 1
        counter += 1
    return counter

def bit_decompose(int_x):
    ''' Function to decompose a number in base-2 numbers. This is performed by
    two binary operators. Idea from Stackoverflow.
        x << y
    Returns x with the bits shifted to the left by y places (and new bits on
    the right-hand-side are zeros). This is the same as multiplying x by 2**y.
        x & y
    Does a "bitwise and". Each bit of the output is 1 if the corresponding bit
    of x AND of y is 1, otherwise it's 0.
    Inputs
    - int_x: integer
    Returns
    - list of base-2 values from which adding them, the input integer can be
    recovered
    '''
    base2 = []
    i = 1
    while (i <= int_x):
        if (i & int_x):
            base2.append(i)
        i <<= 1
    return base2

def flatten_list(list_2levels):
    ''' Function to flatten a list of lists, generating an output with
    all elements in a single level. No duplicate drop neither sort are
    performed
    Inputs
    - list_2levels: nested list to be flatten
    Returns
    - res: list of unordered items contained in the initial nested list
    '''
    f = lambda x: [item for sublist in x for item in sublist]
    res = f(list_2levels)
    return res

def ingest_mask(input_x):
    ''' Method to map back the mask into the CCD, adding the bits without 
    duplicate them
    '''
    # NOTE: Remember table coordinates starts in 1. Shape is (4096, 2048)
    # NOTE: is an unique mask for all the CCD, so don't expect to mask by 
    # section. The masc is a binary 1/0 array
    #
    # Require the BPM, mask, header of the BPM, map of coordinates, bit to use
    # Using variable name 'masc' for the mask
    bpm, masc, header, bit2use = input_x
    ccdnum = int(header['CCDNUM'])
    # Using the mask to identify the region on the BPM we want to mask
    # If the bit we want to use is already in some pixels of the region, we 
    # don't want to overwrite it
    # For a more efficient implementation, first see if the bit to be used is
    # in any of the target pixels. If so, then go one by one. If not, then
    # do all of them at once
    #
    bit_a = bpm[np.where(masc)]
    bit_a = np.unique(bit_a)
    bit_a = list( map(bit_decompose, bit_a) )
    bit_a = flatten_list(bit_a)
    bit_a = list( set(bit_a) )
    if bit2use in bit_a:
        # Bit to be use for masking is between the ones composing the pixels 
        # of the interest area
        nz = np.nonzero(masc)
        for idx in range(nz[0].size):
            row, col = nz[0][idx], nz[1][idx]
            # Go pixel by pixel detecting where the bit to be used lives
            bit_pix = bit_decompose(bpm[row, col])
            if (bit2use in bit_pix):
                # Do not duplicate the bit, because it will be then assumed
                # as 2*bit, thus leading to confussion
                t_w = 'Bit showing in: ({0},{1}). CCD {2}'.format(row, col, c)
                logging.warning(t_w)
            else: 
                bpm[row, col] += bit2use
    else:
        bpm[np.where(masc)] += bit2use
    #
    # Diagnosis plot
    # tmp = np.copy(bpm)
    # tmp[np.where(masc)] = -1000     
    # plt.imshow(tmp, origin='lower')
    # plt.show()
    #
    return bpm
    
def aux_main(fnm_pos='Tapebump_Sections.txt', path_msk='mask_products/',
             path_bpm = 'bpm_Y4E1/', fix_dtype = True,
             use_bit = 1, prefix = 'masked')
    # NOTE: the filename for each BPM is defined lines below, only changing
    # the CCD number
    # Warning message
    logging.warning('Setup needs to be updated/revised for new runs')
    #
    # Load coordinates corresponding to each tapebump section
    # Columns: ccd,x0,x1,y1,y2,tape
    pos = pd.read_csv(fnm_pos)     
    #
    # Create the mask by CCD
    ccdlist = np.r_[[1], np.arange(3, 30 + 1), np.arange(32, 60 + 1), [62]]
    band = 'z'
    # Locate tapebumps masks and construct the total mask, using filename to
    # identify the section
    regex1 = 'a01b_{0}_c*_*.npy'.format(band)
    regex1 = os.path.join(path_msk, regex1)
    ls1 = glob.glob(regex1, recursive=False)
    # Get list of CCDs for which masks are available  
    m_ccd = [int(x[x.find('_c') + 2 : x.find('_c') + 4]) for x in ls1] 
    m_ccd = list(set(m_ccd))
    # Construct the whole CCD mask from files named as a01b_z_c50_s4.npy
    for c in m_ccd:
        f1 = filter(lambda x: '_c{0:02}_'.format(c) in x, ls1) 
        # Get the sections, and load the arrays
        f2 = map(
            lambda y: [np.load(y), y[y.find('.npy') - 2 : y.find('.npy')]], f1
        )
        # Empty host array
        e = np.zeros((4096, 2048)).astype('bool')
        # Apply each masked region
        for i in f2:
            # Get mask (boolean array) and section (string)
            m, s = i 
            # From dataframe get 0-based indices
            row = pos.loc[(pos['tape'] == s) & (pos['ccd'] == c)]
            r0, r1 = row['y1'].iloc[0] - 1, row['y2'].iloc[0] - 1
            c0, c1 = row['x1'].iloc[0] - 1, row['x2'].iloc[0] - 1
            # Ingest into the empty array
            e[r0:r1 + 1 , c0:c1 + 1] = m
        #
        # At this point we have the mask for the nth-CCD 
        #
        # Load the nth-CCD mask
        bpm_file = 'D_n20160921t1003_c{0:02}_r2901p02_bpm.fits'.format(c)
        bpm_file = os.path.join(path_bpm, bpm_file)
        x, hdr = open_fits(bpm_file)
        # We expect the FITS file to have only one extension
        if ((len(x) > 1) or (len(hdr) > 1)):
            t_w = 'FITS file {0} has more than 1 extension'.format(bpm_file)
            t_w += ' Using only first element'
            logging.warning(t_w)
        # Get data from FITS, only first element
        x = x[0]
        hdr = hdr[0]
        # Some BPMs were written with a bug: even when data is integer,
        # the datatype was not. I need to fix it before do comparison
        # against other integers (masks)
        if fix_dtype:
            x = x.astype('uint16') 
        logging.info('CCD {0}'.format(int(hdr['CCDNUM'])))
        #
        # At this point we have loaded the BPM, and the mask for the CCD
        #
        xmod = ingest_mask([x, e, hdr, use_bit]) 
        #
        # Save the modified masks
        #
        outnm = prefix + '_c{0:02}.fits'.format(c)
        try:
            fits = fitsio.FITS(outnm, 'rw')
            fits.write(xmod, header=hdr)
            # Add comment
            comm = 'fpazch added mask to regions in tapebumps.'
            comm += ' Original file {0}'.format(os.path.basename(bpm_file))
            hlist = [{
                'name': 'comment',
                'value': comm,
            }]
            fits[-1].write_keys(hlist)
            fits.close()
            logging.info('Masked BPM written: {0}'.format(outnm))
        except:
            t_e = sys.exc_info()[0]
            logging.error(t_e)
        #
        # Eval plot
        # plt.imshow(e, origin='lower')
        # plt.show()
    return True

if __name__ == '__main__':
    t0 = time.time()
    NPROC = mp.cpu_count() - 1
    aux_main()
