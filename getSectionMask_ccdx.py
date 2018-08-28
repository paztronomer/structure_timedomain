''' Code to get basic statistics for a section of a list of exposures 
'''

import os
import sys
import uuid
import time
import numpy as np
import pandas as pd
import fitsio
import pickle
import logging
import argparse
try:
    import partial
except:
    pass
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)
# Dictionary taken from pixcorrect/decaminfo.py
ccdnums = {
    'S29':  1,
    'S30':  2,
    'S31':  3,
    'S25':  4,
    'S26':  5,
    'S27':  6,
    'S28':  7,
    'S20':  8,
    'S21':  9,
    'S22':  10,
    'S23':  11,
    'S24':  12,
    'S14':  13,
    'S15':  14,
    'S16':  15,
    'S17':  16,
    'S18':  17,
    'S19':  18,
    'S8':  19,
    'S9':  20,
    'S10':  21,
    'S11':  22,
    'S12':  23,
    'S13':  24,
    'S1':  25,
    'S2':  26,
    'S3':  27,
    'S4':  28,
    'S5':  29,
    'S6':  30,
    'S7':  31,
    'N1':  32,
    'N2':  33,
    'N3':  34,
    'N4':  35,
    'N5':  36,
    'N6':  37,
    'N7':  38,
    'N8':  39,
    'N9':  40,
    'N10':  41,
    'N11':  42,
    'N12':  43,
    'N13':  44,
    'N14':  45,
    'N15':  46,
    'N16':  47,
    'N17':  48,
    'N18':  49,
    'N19':  50,
    'N20':  51,
    'N21':  52,
    'N22':  53,
    'N23':  54,
    'N24':  55,
    'N25':  56,
    'N26':  57,
    'N27':  58,
    'N28':  59,
    'N29':  60,
    'N30':  61,
    'N31':  62
}

def main_aux(pathlist=None, ccd=None, coo=None, raw=None, extens=None, 
             label=None, suffix=None):
    if (pathlist is not None):
        # Load the table, with no constraint on filetype, in case path
        # is extremely large
        dt = [('nite', 'i8'), ('expnum', 'i8'), ('band', '|S10'), 
              ('exptime', 'f8'), ('ccdnum' , 'i8'), ('path','|S200'),]
        tab = np.genfromtxt(
            pathlist,
            dtype=dt,
            comments='#',
            delimiter=',',
            missing_values=np.nan,
        )
        #
        # Filter by CCD. It already happened to me that the input list 
        # contained more than the target CCD, and the generated results/stamps
        # were erroneous
        tab = tab[np.where(tab['ccdnum'] == ccd)]
        #
    # Remove duplicates
    uarr, uidx, uinverse, ucounts= np.unique(
        tab['path'],
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    if (uidx.size != tab.size):
        ndup = tab.size - uidx.size
        logging.info('Removing {0} duplicates'.format(ndup))
        tab = tab[uidx]
    # Get CCD name
    def f(dic, x):
        for a, b in dic.iteritems():
            if b == x:
                return a
    ccdname = f(ccdnums, ccd)
    # If raw, extension is going to be the ccd name
    if raw:
        extens = ccdname
    # 
    uband = np.unique(tab['band'])
    aux_call = []
    for b in uband:
        s = tab[np.where(tab['band'] == b)]
        # Will use first element to generate the mask per band
        z = (s['path'][0], coo, extens, raw, 
             s['nite'][0], s['expnum'][0], s['exptime'][0], b,
             ccd, suffix)
        # sel_section(z)
        aux_call.append(z)
    if True:
        # Setup parallel call     
        nproc = len(aux_call) #mp.cpu_count()
        logging.info('Launch {0} parallel processes'.format(nproc))
        P1 = mp.Pool(processes=nproc)
        #
        tmpS = P1.map_async(sel_section, aux_call)
        # To wait and check for the process
        tmpS.wait()
        tmpS.successful()
        tmpS = tmpS.get()
        # Close the pool
        P1.close()
        # For partial usage:
        # partial_aux = partial(stat_section, *list_of_nonvarying_variables)
        # P1.map_async(stat_section, fits_filename)
    return True

def sel_section(y_list):
    ''' Function to open the FITS file, cut the region and save mask
    Inputs 
    - y_list contains the following
    - p: path to the FITS file, to be passed to fits_section()
    - coo: coordinates to be passed to fits_section()
    - ext: extension, to be passed to fits_section() 
    - raw: flag, to be passed to fits_section()
    - nite: observation day for the exposure
    - expnum: exposure number for the image
    - exptime: exposure time in seconds
    - band: filter used for the exposure
    Returns
    - a tuple of the results, to be used in constructing a structured array. 
    '''
    path, coo, ext, raw, nite, expnum, exptime, band, ccd, suf = y_list 
    m, header = fits_section(path, coo, ext, raw)
    # Do not normalize by exptime, as these are masks!
    out_dir = 'mask_stamps/{0}/c{1:02}'.format(band, ccd)
    out_npy = 'msk_{0}_{1}_D{2:08}.npy'.format(suf, band, expnum)
    out_npy = os.path.join(out_dir, out_npy)
    if (not os.path.exists(out_dir)):
        logging.error('Directory {0} does not exists'.format(out_dir))
        logging.info('Creating directory {0}'.format(out_dir))
        os.makedirs(out_dir)
    if os.path.exists(out_npy):
        logging.warning('File {0} exists'.format(out_npy))
        out_npy = out_npy.replace(suf, str(uuid.uuid4()))
        logging.warning('New output name {0}'.format(out_npy))
    try:
        np.save(out_npy, m)
        logging.info('Saving MSK {0} without normalization'.format(out_npy))
    except:
        logging.info('File {0} could not be saved'.format(out_npy))
        os.remove(out_npy)
    """
    # Function to get mean, median, stdev, min, max, MAD, RMS
    f1 = lambda x: [ np.mean(x), np.median(x), np.std(x), np.min(x), 
                     np.max(x), np.median(np.abs( x - np.median(x) )), 
                     np.sqrt(np.mean(np.square( x.ravel() ))) ]
    aux = f1(m) + [nite, expnum, exptime, band, header['MJD-OBS']]
    return tuple(aux)
    """
    return True

def fits_section(fname, coo, ext, raw):
    ''' Function to read section of the CCD, based in input coordinates. For
    raw images, use DATASEC to determine the offset to use
    Inputs
    - fname: filename of the FITS file
    - coo: list or tuple. Coordinates for the left-lower and right-upper 
    of the box
    - ext: FITS file extension to read
    - raw: boolean indicating if FITS is raw, having overscan still in the 
    image
    Returns
    - ccd section array
    '''
    # fname, coo, ext, raw = x_list
    # x0, y0, x1, y1 = coo
    x0, x1, y0, y1 = coo
    x0 -= 1
    x1 -= 1
    y0 -= 1
    y1 -= 1
    # logging.info('Transforming position to indices (starting at zero)')
    if os.path.exists(fname):
        fits = fitsio.FITS(fname)
        # sq = np.copy(fits[ext][y0:y1 , x0:x1])
        # fits.close()
        # return sq
    else:
        logging.error('File {0} does not exists'.format(fname))
        return False
    # For raw exposures, determine the offset
    if raw:
        # Read the header info
        hdr = fitsio.read_header(fname, ext)
        dsec = hdr['DATASEC'].strip().strip('[').strip(']').replace(':', ',')
        dsec = map(int, dsec.split(','))
        if (dsec[2] > dsec[3]) and (dsec[0] < dsec[1]):
            logging.info('{0} inverse y-reading direction'.format(fname))
            sec = np.copy(fits[ext][y0 + dsec[3] : y1 + 1, 
                                    x0 + dsec[0] : x1 + 1]) 
        elif (dsec[2] > dsec[3]) and (dsec[0] > dsec[1]):
            logging.info('{0} inverse y and x-reading direction'.format(fname))
            sec = np.copy(fits[ext][y0 + dsec[3] : y1 + 1,
                                    x0 + dsec[1] : x1 + 1])
        elif (dsec[2] < dsec[3]) and (dsec[0] > dsec[1]):
            logging.info('{0} inverse x-reading direction'.format(fname))
            sec = np.copy(fits[ext][y0 + dsec[2] : y1 + 1,
                                    x0 + dsec[1] : x1 + 1])
        elif (dsec[2] < dsec[3]) and (dsec[0] < dsec[1]):
            # logging.info('{0} increasing reading direction'.format(fname))
            sec = np.copy(fits[ext][y0 + dsec[2] : y1 + 1,
                                    x0 + dsec[0] : x1 + 1])
        fits.close()
        return sec
    else:
        sec = np.copy(fits[ext][y0 : y1 + 1, x0 : x1 + 1])
        hdr = fits[ext].read_header()
        fits.close()
        return [sec, hdr]

if __name__ == '__main__':
    
    hgral = 'Script to write out the mask of a immask image. Uses ONLY'
    hgral += ' the first element of a given list per band, as the BPM are'
    hgral += ' supossed to be constant along same band exposures'
    epil = 'MODIFIED VERSION, USES CCD COLUMN'
    ecl = argparse.ArgumentParser(description=hgral, epilog=epil)
    h0 = 'Table of night, expnum, band, exptime, and path to images'
    h0 += ' for which stats should be calculated. Please use column names:'
    h0 += ' nite, expnum, band, exptime, ccdnum, path.'
    h0 += 'Format is CSV'
    ecl.add_argument('path', help=h0)
    h1 = 'CCD number on which operate'
    ecl.add_argument('--ccd', help=h1, type=int)
    h2 = 'Coordinates of the section to cut and save.'
    h2 += ' Coordinates values starting at 1, not at 0.'
    h2 += ' Format: x0 x1 y0 y1. It includes the ending position.'
    h2 += ' A unit is subtracted from the coords to convert them to indices'
    ecl.add_argument('--coo', help=h2, type=int, nargs=4)
    h3 = 'Flag. Use if inputs are raw image with overscan'
    ecl.add_argument('--raw', help=h2, action='store_true')
    h4 = 'Label to be used for output files. Default id PID'
    def_label = 'PID{0}'.format(os.getpid())
    ecl.add_argument('--label', help=h4, metavar='', default=def_label)
    h6 = 'Suffix to be used on the stamps'
    ecl.add_argument('--suffix', help=h6, metavar='')
    aux_ext = 'MSK'
    h7 = 'Extension to read from the FITS. Default : {0}'.format(aux_ext)
    ecl.add_argument('--ext', help=h7, metavar='extension', default=aux_ext)
    # Parser
    ecl = ecl.parse_args()
    #
    logging.warning('Only generating the mask, one per set')
    #
    main_aux(pathlist=ecl.path, ccd=ecl.ccd, coo=ecl.coo, raw=ecl.raw, 
             label=ecl.label, suffix=ecl.suffix, 
             extens=ecl.ext)

