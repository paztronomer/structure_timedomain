''' Francisco Paz-Chinchon
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import (MinMaxInterval, SqrtStretch,
                                   ImageNormalize, ZScaleInterval)


def plot_grid(ccd, path='medImg_a01/', ep='y4e1', 
              sub=['a01a', 'a01b'], band='g', 
              plotdir='plot_a01/'):
    
    fig = plt.figure(figsize=(10,6))

    # row, col
    gs1 = gridspec.GridSpec(3, 2)
    gs1.update(left=0.05, right=0.5, wspace=0.01, hspace=0.01,)
    ax1b = plt.subplot(gs1[2, 0])
    ax2b = plt.subplot(gs1[2, 1])
    ax3b = plt.subplot(gs1[1, 0])
    ax4b = plt.subplot(gs1[1, 1])
    ax5b = plt.subplot(gs1[0, 0])
    ax6b = plt.subplot(gs1[0, 1])

    gs2 = gridspec.GridSpec(3, 2)
    gs2.update(left=0.6, right=0.95, wspace=0.01, hspace=0.01,)
    ax1a = plt.subplot(gs2[2, 0])
    ax2a = plt.subplot(gs2[2, 1])
    ax3a = plt.subplot(gs2[1, 0])
    ax4a = plt.subplot(gs2[1, 1])
    ax5a = plt.subplot(gs2[0, 0])
    ax6a = plt.subplot(gs2[0, 1])

    axb = [ax1b, ax2b, ax3b, ax4b, ax5b, ax6b]
    axa = [ax1a, ax2a, ax3a, ax4a, ax5a, ax6a]
    for ax in axa:
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                       left=False, right=False, labelbottom=False, 
                       labelleft=False,)
    for ax in axb:
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.tick_params(axis='both', which='both', bottom=False, top=False, 
                       left=False, right=False, labelbottom=False, 
                       labelleft=False,)
    
    # Create filename list
    # Filename set a
    f = 'medImg_{0}_{1}_c{2:02}_XXXX_{3}.npy'.format(ep, sub[0], 
                                                     ccd, band)
    f = os.path.join(path, f)
    sub_a_path = [f.replace('XXXX', i) 
                  for i in ['s1', 's2', 's3', 's4', 's5', 's6']]
    # Filename set b
    g = 'medImg_{0}_{1}_c{2:02}_XXXX_{3}.npy'.format(ep, sub[1], 
                                                     ccd, band)
    g = os.path.join(path, g)
    sub_b_path = [g.replace('XXXX', i) 
                  for i in ['s1', 's2', 's3', 's4', 's5', 's6']]
    
    # Load files
    sub_a_im = [np.load(j) for j in sub_a_path]
    sub_b_im = [np.load(j) for j in sub_b_path]

    # Iterate and plot. Use same scale
    # min_a_im = np.min([np.min(i) for i in sub_a_im])
    # max_a_im = np.max([np.max(i) for i in sub_a_im])
    
    kw_a = {
        'cmap' : 'gray_r',
        'origin' : 'lower',
    }
    for i in range(len(axa)):
        im_x = sub_a_im[i]
        im_norm = ImageNormalize(im_x, 
                                 interval=ZScaleInterval(),
                                 stretch=SqrtStretch(),)

        im = axa[i].imshow(im_x, norm=im_norm, **kw_a)
        # create an axes on the r/l side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        if (i + 1) % 2:
            pos = 'left'
        else:
            pos = 'right'
        divider = make_axes_locatable(axa[i])
        cax = divider.append_axes(pos, size='5%', pad=0.05)
        cbar = fig.colorbar(im, cax=cax)#orientation='horizontal',)
        cbar.ax.tick_params(labelsize=8) 
        if (i + 1) % 2:
            cax.yaxis.set_ticks_position('left')

    
    kw_b = {
        'cmap' : 'gray_r',
        'origin' : 'lower',
    }
    for k in range(len(axb)):
        im_x2 = sub_b_im[k]
        im_norm2 = ImageNormalize(im_x2, 
                                  interval=ZScaleInterval(),
                                  stretch=SqrtStretch(),)

        im = axb[k].imshow(im_x2, norm=im_norm2, **kw_b)
        # create an axes on the r/l side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        if (k + 1) % 2:
            pos = 'left'
        else:
            pos = 'right'
        divider = make_axes_locatable(axb[k])
        cax2 = divider.append_axes(pos, size='5%', pad=0.05)
        cbar2 = fig.colorbar(im, cax=cax2)#orientation='horizontal',)
        cbar2.ax.tick_params(labelsize=8) 
        if (k + 1) % 2:
            cax2.yaxis.set_ticks_position('left')
    # Title
    txt = 'Median image for Y4E1 > 20171018, CCD={0}, {1}-band'.format(ccd, band)
    txt += '\nleft:subset below T_EFF cutval; right:subset above T_EFF cutval'
    plt.suptitle(txt, color='blue', fontsize=10, weight='semibold')

    plt.subplots_adjust(top=0.93, bottom=0.03)
    
    # Check if destination folder exists, if no exists, create it
    if (not os.path.exists(plotdir)):
        os.path

    # Write out the figure 
    outf = 'gridTape_{0}_c{1:02}_{2}.pdf'.format(ep, ccd, band)
    outf = os.path.join(plotdir, outf)
    plt.savefig(outf, dpi=500, format='pdf')

    #plt.show()
    return True


if __name__ == '__main__':
    
    CCD = np.r_[[1], np.arange(3, 60 + 1), [62]]

    for b in ['g', 'r', 'i', 'z', 'Y']:
        for c in CCD:
            print(c,b)
            aux_path = os.path.join('medImg_a01', b, 'c{0:02}'.format(c))
            plot_path = os.path.join('plot_a01', b, 'c{0:02}'.format(c))
            plot_grid(c, path=aux_path, plotdir=plot_path,
                      band=b, sub=['a01a', 'a01b'], ep='y4e1')

