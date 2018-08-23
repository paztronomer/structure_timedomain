''' Script to create sub-directoris, based on some rigid criteria
'''

import os
import sys
import numpy as np
import logging

if __name__ == '__main__':
    # Read table of filenames
    # stamps/bands/ccd/tape
    d = np.loadtxt('list_stamp', dtype=[('path', 'S100'),], 
                   converters={0 : lambda s: os.path.join('stamps/', s)})
    # Add the location
    # d['path'] = os.path.join('stamps/', d['path'])
    # Create the structure
    for b in ['g', 'r', 'i', 'z', 'Y']:
        for c in np.r_[[1], np.arange(3, 61), [62]]:
            for t in ['s{0}'.format(pos) for pos in range(1, 7)]:
                folder = os.path.join('stamps_a01', b, 'c{0:02}'.format(c), t)
                # The os.makedirs is recursive and creates all the upper
                # directories up to the leaf
                try:
                    # os.removedirs(folder)
                    os.makedirs(folder)
                except OSError:
                    logging.error('OS error')
                    raise
                except:
                    e = sys.exc_info()[0]
                    logging.error('Unexpected error: {0}'.format(e))
                    raise
    # Move files
    b = []
    mv = []
    use_mv_bash = True
    for f in d['path']:
        aux = os.path.basename(f).replace('.npy', '').split('_')
        subset, ccd, tape, band, unitname = aux[2:]
        dest = os.path.join('stamps_a01', band, ccd, tape, os.path.basename(f))
        if use_mv_bash:
           auxil = 'mv {0} {1}'.format(f, dest)
           mv.append(auxil)
        else:
            try:
                os.renames(f, dest)
            except:
                logging.error('Error moving: {0}'.format(f))
                b.append(f)

    if use_mv_bash:
        with open('mv_files_PID{0}.sh'.format(os.getpid()), 'w+') as f:
            f.write('#!/bin/bash \n')
            f.write('echo Starting, $(date) \n')
            for l in mv:
                f.write(l + '\n')
            f.write('echo Ending, $(date)')
                
    if (len(b) > 0):
        np.dump(b, open('b.pickle', 'wb'))
