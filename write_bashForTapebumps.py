''' quick script to write out the bash call to generate the tapebump stamps 
'''

import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    # Columns: ccd, x1, x2, y1, y2, bpm
    # Shape (372, 6)
    x = np.loadtxt('badlist_tapebumps.lst', dtype='int')
    ccd = np.unique(x[:, 0])
    # Randomize ascending ccd order. Shuffle does it in place
    np.random.shuffle(ccd)
    with open('run_stamp_tapebump.sh', 'w') as fy:
        for c in ccd:
            subx = x[np.where(x[:, 0] == c)]
            for t in range(subx.shape[0]):
                # For Y4E1
                line = 'python timeseries_ccdx_statSection.py Y4A1_Y4E1.csv'
                line += ' --ccd {0}'.format(c)
                line += ' --coo {0} {1} {2} {3}'.format(subx[t , 1], subx[t , 2],
                                                        subx[t , 3], subx[t , 4])
                line += ' --label y4a1_y4e1_c{0:02}_s{1}'.format(c, t + 1)
                line += ' --stamp'
                line += ' --suffix y4a1_y4e1_c{0:02}_s{1}'.format(c, t + 1)
                # For Y4X1
                l = 'python timeseries_ccdx_statSection.py Y4A1_Y4X1.csv'
                l += ' --ccd {0}'.format(c)
                l += ' --coo {0} {1} {2} {3}'.format(subx[t , 1], subx[t , 2],
                                                     subx[t , 3], subx[t , 4])
                l += ' --label y4a1_y4x1_c{0:02}_s{1}'.format(c, t + 1)
                l += ' --stamp'
                l += ' --suffix y4a1_y4x1_c{0:02}_s{1}'.format(c, t + 1)
                # Write lines 
                fy.write(line + '\n')
                fy.write(l + '\n')

# python timeseries_ccdx_statSection.py Y4A1_Y4E1.csv --ccd 1 --coo 1 129 1 88 --label y4a1_y4e1_c01_s1 --stamp --suffix y4a1_y4e1_c01_s1
# python timeseries_ccdx_statSection.py Y4A1_Y4X1.csv --ccd 1 --coo 1 129 1 88 --label y4a1_y4x1_c01_s1 --stamp --suffix y4a1_y4x1_c01_s1

