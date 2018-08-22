import pandas as pd
import numpy as np

df = pd.read_csv('c01_Y4A1_Y4E1.csv', 
                 names=['nite', 'expnum', 'band', 'exptime', 'path'],
                 comment='#')
df = df.loc[df['nite'] >= 20161018]

CCD = np.r_[[1], np.arange(3, 60 + 1), [62]]

nite = []
expnum = []
band = []
exptime = []
ccdnum = []
path = []
for idx, row in df.iterrows():
    for c in CCD:
        nite.append(row['nite'])
        expnum.append(row['expnum'])
        band.append(row['band'])
        exptime.append(row['exptime'])
        ccdnum.append(c)
        path.append(row['path'].replace('_c01_', '_c{0:02}_'.format(c)))

dfout = pd.DataFrame({'1nite' : nite, '2expnum' : expnum,
                      '3band' : band, '4exptime' : exptime,
                      '5ccdnum' : ccdnum, '6path' : path,})
dfout.to_csv('Y4A1_Y4E1_a01.csv', index=False, header=True)
