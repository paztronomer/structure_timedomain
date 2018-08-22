''' Code to generate median image and other analysis, from a set of tapebumps
Steps
=====
1) get DB assessment of the exposures
2) divide the expnums in subgroups for the epoch
3) create the median image per band, per ccd, per tapebump  
4) get the statistics of the region of the tapebump of interest
'''

import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import multiprocessing as mp
import easyaccess as ea

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
)

def stat_cube(x3d, func):
    ''' Function to calculate the median image per pixel, using an input 
    3dimensional array containing the stamps on which to work.
    Uses numpy iteration tools
    Example for easy usage:
    x3d = np.dstack(array_i)
    x_median = stat_cube(x3d, (lambda: np.median)())
    '''
    out = np.zeros_like(x3d[:, :, 0])
    it = np.nditer(x3d[:, :, 0], flags=['multi_index'])
    while not it.finished:
        i1, i2 = it.multi_index
        out[i1, i2] = func(x3d[i1, i2, :])
        it.iternext()
    return out

def db_query(q):
    connection = ea.connect('desoper')
    df_qa = connection.query_to_pandas(q)
    return df_qa

def stamp_load(args):
    filename = args
    try:
        return np.load(filename)
    except:
        return np.nan

def read_stamps(epoch, ccd, tape, b, sublist_expnum):
    pattern = 'y4a1_{0}_a01_c{1:02}_{2}_{3}_*'.format(epoch, ccd, tape, b)
    t0 = time.time()
    # Iterator to walk through the directory, once used it cannot be reused
    iterat_aux = glob.glob(pattern, recursive=False)
    # If sublist_expnum, then only use these exposure numbers
    iterat = [x for x in iterat_aux for e in sublist_expnum if e in x]
    P2 = mp.Pool(processes=mp.cpu_count())
    res_v2 = P2.map_async(np.load, iterat)
    res_v2.wait()
    res_v2 = res_v2.get()
    x_v2 = np.dstack(res_v2)
    t1 = time.time()
    print(x_v2.shape)
    print('Elapsed time: {0:.2f}'.format((t1 - t0) / 60))
    return x_v2

def stamp_medImg(df, label_epoch, label_set):
    ''' Generates median image, write it out, and return simple statistics 
    of each created image
    '''
    # Receives the dataframe. Divide by band, then by tapebump, and then 
    # by ccd
    # Path were stamps are located
    p = '/work/devel/fpazch/calib_space/Y5A1_tapebumps_check/stamps/'
    df['fname'] = p + df['fname']
    # Initialize pool
    P1 = mp.Pool(processes=mp.cpu_count())
    for b in ['g', 'r', 'i', 'z', 'Y']: #df['band'].unique():
        # List to save statistics
        aux_stat = []
        for tape in df['s'].unique():
            logging.info('{0} {1} {2}'.format(b, tape, time.ctime()))
            #
            # Change it for running all
            # for c in df['ccd'].unique():
            #
            # Running for ccds 1 to 25
            #
            for c in df['ccd'].unique():
                dfx = df.loc[(df['band'] == b) & (df['s'] == tape) &
                             (df['ccd'] == c)]

                res = P1.map_async(stamp_load, dfx['fname'].values, 
                                   chunksize=20)
                res.wait()
                res.successful()
                res = res.get()
                # Drop NaN in case some numpy error loading
                res = [x for x in res if ~np.all(np.isnan(x))]
                # Stack into a 3D array
                x3d_tmp = np.dstack(res)
                # Create the median image
                med_tmp = stat_cube(x3d_tmp, (lambda: np.median)()) 
                # Write out median images
                out = 'medImg_{0}_{1}_c{2:02}'.format(label_epoch, label_set, c)
                out += '_{0}_{1}.npy'.format(tape, b)
                out =os.path.join('medImg/', out)
                np.save(out, med_tmp)
                # Save statistics of the stamp
                med_tmp = med_tmp.flatten()
                aux_stat.append(
                    [label_set, c, tape, b, 
                    np.mean(med_tmp), 
                    np.median(med_tmp),
                    np.median(np.abs( med_tmp - np.median(med_tmp) )),
                    np.sqrt( np.mean(np.square(med_tmp)) ),
                    np.std(med_tmp),
                    np.ptp(med_tmp),]
                )
        # Write out stats in CSV, per band
        fnm_out = 'medImg_stat_{0}_{1}_{2}.csv'.format(label_epoch, 
                                                       label_set, 
                                                       b)
        dt = [('set', '|S15'), ('ccd', 'i8'), ('tape', '|S10'), ('band', '|S5'),
              ('mean', 'f8'), ('median', 'f8'), ('mad', 'f8'), ('rms', 'f8'),
              ('std', 'f8'), ('range', 'f8')]
        arr = np.rec.array(aux_stat, dtype=dt)
        np.save(fnm_out.replace('.csv', '.npy'), arr)
        try:
            L = list(zip(*aux_stat))
            df_out = pd.DataFrame({'set' : L[0], 'ccd' : L[1],
                                   'tape' : L[2], 'band' : L[3],
                                   'mean' : L[4], 'median' : L[5],
                                   'mad' : L[6], 'rms' : L[7],
                                   'std' : L[8], 'range' : L[9],})
            # df_out = pd.DataFrame.from_records(arr)
            df_out.to_csv(fnm_out, index=False, header=True)
        except: 
            pass
    P1.close()
    return aux_stat

if __name__ == '__main__':
    # NOTE: the main is going to be verbose, keep it simple
    #
    # First, get DB info for Y4E1 and Y4X1
    query_db = False
    if query_db:
        q1 = 'select qa.expnum, e.nite, qa.t_eff, e.band, qa.skybrightness,'
        q1 += ' qa.psf_fwhm, qa.f_eff, qa.c_eff, qa.b_eff, qa.skytilt'
        q1 += ' from exposure e, qa_summary qa, proctag pt'
        q1 += ' where pt.tag=\'Y4A1_FINALCUT\''
        q1 += ' and pt.pfw_attempt_id=qa.pfw_attempt_id'
        q1 += ' and qa.expnum=e.expnum'
        q1 += ' and e.nite between 20160813 and 20161208'
        q1 += ' order by qa.expnum'
        #
        q2 = 'select qa.expnum, e.nite, qa.t_eff, e.band, qa.skybrightness,'
        q2 += ' qa.psf_fwhm, qa.f_eff, qa.c_eff, qa.b_eff, qa.skytilt'
        q2 += ' from exposure e, qa_summary qa, proctag pt'
        q2 += ' where pt.tag=\'Y4A1_FINALCUT\''
        q2 += ' and pt.pfw_attempt_id=qa.pfw_attempt_id'
        q2 += ' and qa.expnum=e.expnum'
        q2 += ' and e.nite between 20161209 and 20170102'
        q2 += ' order by qa.expnum'
        #
        db_y4e1 = db_query(q1)
        db_y4x1 = db_query(q2)
        db_y4e1.to_pickle('dbinfo_y4e1.pickle')
        db_y4x1.to_pickle('dbinfo_y4x1.pickle')
    else:
        db_y4e1 = pd.read_pickle('dbinfo_y4e1.pickle')
        db_y4x1 = pd.read_pickle('dbinfo_y4x1.pickle')
    # List of CCDs to be used, avoid CCDs 2 and 61
    CCD = np.r_[[1], np.arange(3, 61), [62]]
    #
    # Y4E1 > 20161018
    #
    # Set a01: Y4E1 after oct 18, to the end of the season. Cuts in
    # t-eff by band.
    a01 = db_y4e1.loc[db_y4e1['NITE'] >= 20161018]
    #
    # Discard bands not being used in coadds: u
    # Cutvals for t_eff: 0.2 for g, Y; 0.3 for g, r, i, z
    band = a01['BAND'].unique()
    band = band[np.where(band != 'u')]
    for b in a01['BAND'].unique():
        if (b in ['g', 'Y']):
            # Apply cut in t_eff. Define th2 2 subsets: above and below the 
            # cut value
            a01a = a01.loc[a01['T_EFF'] > 0.2]
            a01b = a01.loc[a01['T_EFF'] <= 0.2]
        elif (b in ['r', 'i', 'z']):
            a01a = a01.loc[a01['T_EFF'] > 0.3]
            a01b = a01.loc[a01['T_EFF'] <= 0.3]
    # Re create the set of filenames?
    create_filename_a01 = True
    if create_filename_a01:
        logging.info('Creating df_a01a.pickle, df_a01b.pickle')
        # For sets of band-exposures, go through the individuals ccds, an inside 
        # the ccds go tapebump by tapebump
        a01a_fnm, a01b_fnm = [], []
        a01a_b, a01b_b = [], []
        a01a_ccd, a01b_ccd = [], []
        a01a_s, a01b_s = [], []
        for b in a01['BAND'].unique():
            aux_a01a = a01a.loc[a01a['BAND'] == b]
            aux_a01b = a01b.loc[a01b['BAND'] == b]
            # Create list of expected filenames
            # y4a1_y4e1_c59_s3_Y_D00573344.npy
            # Set above the cut level
            for idx, row in aux_a01a.iterrows():
                for s in ['s1', 's2', 's3', 's4', 's5', 's6']:
                    for ccd in CCD:
                        f = 'y4a1_y4e1_a01_c{0:02}_{1}_{2}_D{3:08}.npy'.format(
                            ccd,
                            s,
                            b,
                            row['EXPNUM'],
                        )
                        a01a_fnm.append(f)
                        a01a_b.append(b)
                        a01a_ccd.append(ccd)
                        a01a_s.append(s)
            # Set below the cut level
            for idx, row in aux_a01b.iterrows():
                for s in ['s1', 's2', 's3', 's4', 's5', 's6']:
                    for ccd in CCD:
                        f = 'y4a1_y4e1_a01_c{0:02}_{1}_{2}_D{3:08}.npy'.format(
                            ccd,
                            s,
                            b,
                            row['EXPNUM'],
                        )
                        a01b_fnm.append(f)
                        a01b_b.append(b)
                        a01b_ccd.append(ccd)
                        a01b_s.append(s)
        # Save into dataframes for easy of use and load
        df_a01a = pd.DataFrame({'fname' : a01a_fnm, 'band' : a01a_b, 
                                'ccd' : a01a_ccd, 's' : a01a_s})
        df_a01b = pd.DataFrame({'fname' : a01b_fnm, 'band' : a01b_b, 
                                'ccd' : a01b_ccd, 's' : a01b_s})
        # Pickle to files
        df_a01a.to_pickle('df_a01a.pickle')
        df_a01b.to_pickle('df_a01b.pickle')
    else:
        logging.info('Loading df_a01a.pickle, df_a01b.pickle')
        df_a01a = pd.read_pickle('df_a01a.pickle')
        df_a01b = pd.read_pickle('df_a01b.pickle')
    #
    # Read the stamps and create the median image. Also save some basic stats
    a01b_medImg = stamp_medImg(df_a01b, 'y4e1', 'a01b')
    a01a_medImg = stamp_medImg(df_a01a, 'y4e1', 'a01a')
    
    
    logging.info(time.ctime())
    
    #
    # *) SSIM 
    # *) Flat fields are supossed to be changing 
    # *)  
