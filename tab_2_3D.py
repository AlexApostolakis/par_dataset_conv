import nppar
import time
import numpy as np
import math
import datatable as dt
import time
import xarray
import os
import fileutils
import traceback
from datetime import datetime
import random
from functools import partial
from multiprocessing import Pool, Array, Process, Manager, Queue, cpu_count
from functools import partial
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1 as axes_grid1


def gridinfo():
    rdiff = 2227
    minnorth = 333237
    # maxwest=1160624
    minwest = 1156167
    maxeast = 2747676
    maxsouth = 3504175
    gridwidth = ((maxeast - minwest) % rdiff) + 1
    firstid = minwest - math.ceil((minwest - minnorth) / rdiff) * rdiff
    gridheight = math.ceil((maxsouth - firstid) / rdiff)
    return rdiff, firstid, gridwidth, gridheight


def walkmonthdays(sfolder):
    # sfolder = '/data2/ffp/datasets/daily/2015/08'
    exfeat = ["id", "firedate"]
    dayfiles = []
    for dayf in fileutils.find_files(sfolder, '*_norm.csv', listtype="walk"):
        dayfiles += [dayf]
        # print(fday)
        '''
        try:
            #fday = '/data2/ffp/datasets/daily/2021/08/20210804_norm.csv'
            creategrid_xs(fday, rdiff, firstid, gridwidth, gridheight)
        except:
            print("Fail to convert %s"%fday)
            traceback.print_exc()
        '''
    return dayfiles


def get_grid_xy(firstid, rdiff, _id, ):
    row = int((_id - firstid) / rdiff)
    col = int(_id - firstid - rdiff * row)
    return row, col


def assignrow(ggrid, tabrow):
    row, col = get_xy(tabrow[0])
    ggrid[row, col, :] = tabrow[:]


def assignrowshared(ggrid_sh, grid_shape, tabrows):
    for i in range(tabrows.shape[0]):
        tabrow = tabrows[i, :]
        try:
            row, col = get_xy(tabrow[0])
            idx = row * grid_shape[1] * grid_shape[2] + col * grid_shape[2]
            ggrid_sh[idx:idx + grid_shape[2]] = tabrow[:]
        except:
            "Error row: %s\n" % i + traceback.print_exc()


def assignrowshared2(ggrid_sh, tabrows):
    for i in range(tabrows.shape[0]):
        tabrow = tabrows[i, :]
        try:
            row, col = get_xy(tabrow[0])
            ggrid_sh[row, col] = tabrow[:]
            # print(row, col)
        except:
            "Error row: %s\n" % i + traceback.print_exc()

def creategrid_xs_small(rdiff, firstid, gridwidth, gridheight, dayfile, pcpus, ccpus, queue):
    #print(rdiff, firstid, gridwidth, gridheight, dayfile, pcpus, ccpus)
    try:
        stpr = time.process_time()
        orig_path = os.path.dirname(dayfile)
        fname = os.path.basename(dayfile)
        daygrid = "%s_grid.nc" % (fname[0:8])
        # if os.path.isfile(os.path.join(orig_path, daygrid)): return
        dt_df = dt.fread(dayfile, nthreads=1)  # pcpus)
        firstfeat = dt_df.names.index('id')
        # npday = dt_df[:, firstfeat:].to_numpy(dt.float32)

        dynamic_feat = ['id', 'max_temp', 'min_temp', 'mean_temp', 'res_max',
                        'dom_vel', 'rain_7days',
                        'ndvi_new', 'evi', 'lst_day', 'lst_night', 'max_dew_temp',
                        'mean_dew_temp', 'min_dew_temp', 'fire', 'dir_max_1', 'dir_max_2',
                        'dir_max_3', 'dir_max_4', 'dir_max_5', 'dir_max_6', 'dir_max_7',
                        'dir_max_8', 'dom_dir_1', 'dom_dir_2', 'dom_dir_3', 'dom_dir_4',
                        'dom_dir_5', 'dom_dir_6', 'dom_dir_7', 'dom_dir_8',
                        'frequency', 'f81', ]

        dyn_df = dt_df[:, dynamic_feat]
        npday = dyn_df.to_numpy(dt.float32)
        #print(npday.shape)
        # id2xy, grid = nppar.fillcube(ccpus, npday, firstid, rdiff, gridwidth, gridheight, sched='static')
        id2xy, grid = nppar.fillcube(ccpus, npday, firstid, rdiff, gridwidth, gridheight, sched='static', intensive=0)
        #print('finished')
        vardict = {}
        for i in range(0, len(dyn_df.names)):
            varname = dyn_df.names[i]
            if dyn_df.names[i] == 'x' or dyn_df.names[i] == 'y':
                varname = '%spos' % varname
            vardict[varname] = (["x", "y", "time"], np.expand_dims(grid[:, :, i], axis=2))
        t = datetime.strptime(os.path.basename(dayfile)[0:8], '%Y%m%d')
        xsday = xarray.Dataset(data_vars=vardict, coords=dict(x=range(gridwidth), y=range(gridheight), time=[t]))
        xsday.to_netcdf(os.path.join(orig_path, daygrid))
        # print("Successfull convertion %s" % dayfile)
        epr = time.process_time()
        queue.put(epr - stpr)
    except:
        print("Fail to convert %s" % dayfile)
        traceback.print_exc()
        with open("/data2/ffp/datasets/daily/failedgrids.log", "a") as f:
            f.write(dayfile)


def new_process(creategrid, proclist, day, pthreads, cthreads):
    q = Queue()
    proclist += [{'proc': Process(target=creategrid, args=(day, pthreads, cthreads, q)), 'queue': q}]
    proclist[-1]['proc'].start()

def create_xs_files(creategrid, days, pthreads, cthreads):
    procs = []
    proctimetotal = 0
    dayscompleted = []
    #print(days)
    for cpu in range(pthreads):
        d = days.pop()
        dayscompleted += [d]
        #print('initial proc')
        new_process(creategrid, procs, d, pthreads, cthreads)
    while len(procs) > 0:
        time.sleep(0.1)
        for p in procs:
            try:
                proctimetotal += p['queue'].get_nowait()
            except:
                pass
            if not p['proc'].is_alive():
                #print('remove, tot procs: %d' % len(procs))
                procs.remove(p)
                #print('tot procs: %d' % len(procs))
        while len(procs) < pthreads:
            if len(days) == 0: break
            #print('new proc')
            d = days.pop()
            dayscompleted += [d]
            new_process(creategrid, procs, d, pthreads, cthreads)
    return proctimetotal


def plotscatter(atimes, typ=0):
    for y in range(1, atimes.shape[1]):
        plt.scatter(range(1, atimes.shape[0]), [y] * (atimes.shape[0] - 1), s=atimes[1:, y, typ])
    plt.show()


def plotheatmap(atimes, title1='Wall time', title2='Process (CPU) time'):
    fig = plt.figure(figsize=(12, 6))
    grid = axes_grid1.AxesGrid(
        fig, 111, nrows_ncols=(1, 2), axes_pad=0.5, cbar_location="right",
        cbar_mode="each", cbar_size="7%", cbar_pad="5%", )
    data = atimes[1:, 1:, 0]
    im0 = grid[0].imshow(data, cmap='turbo', interpolation='None')
    grid.cbar_axes[0].colorbar(im0)
    # Show all ticks and label them with the respective list entries
    grid[0].set_xticks(np.arange(data.shape[1]), labels=np.arange(1, data.shape[1] + 1))
    grid[0].set_yticks(np.arange(data.shape[0]), labels=np.arange(1, data.shape[0] + 1))
    grid[0].set_ylabel('# cython threads')
    grid[0].set_xlabel('# python threads')
    grid[0].title.set_text(title1)
    # ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
    if not title2 is None:
        data = atimes[1:, 1:, 1]
        im1 = grid[1].imshow(data, cmap='turbo', interpolation='None')
        grid[1].set_xticks(np.arange(data.shape[1]), labels=np.arange(1, data.shape[1] + 1))
        grid[1].set_xlabel('# python threads')
        grid[1].title.set_text(title2)
        grid.cbar_axes[1].colorbar(im1)

def dataset_conv_opt(creategrid, maxcpus, dayfiles):
    nruns=1
    totalrun=0
    ctr=range(1,maxcpus+1,1)
    ptr=range(1,maxcpus+1,1)
    atimes2=np.zeros((max(list(ctr))+1,max(list(ptr))+1,2))
    for cthreads in ctr:
        for pthreads in ptr:
            start=time.time()
            nfiles=pthreads
            proctime=create_xs_files(creategrid, dayfiles[:nfiles], pthreads, cthreads)
            end=time.time()
            print('wall time: %.1f sec, process time: %.1f sec, python threads %s, cython threads %s' % (end - start, proctime, pthreads, cthreads))
            atimes2[cthreads, pthreads,:]=np.array([end-start,proctime])[:]
    return atimes2

#initialize
rdiff, firstid, gridwidth, gridheight = gridinfo()
get_xy=partial(get_grid_xy, firstid, rdiff)
dayfiles=walkmonthdays('/data2/ffp/datasets/daily/')
#fday='/data2/ffp/datasets/daily/2021/08/20210823_norm.csv'
fday='/data2/ffp/datasets/daily/2015/09/20150923_norm.csv'
dt_df = dt.fread(fday)
firstfeat=dt_df.names.index('id')
npday = dt_df[:, firstfeat:].to_numpy(dt.float32)
maxcpus=cpu_count()
featn = len(dt_df[:, firstfeat:].names)
ggrid = np.zeros((gridwidth, gridheight, featn))
ggrid[:,:]=np.nan
assignr=partial(assignrow, ggrid)
print('max cpu count %s'%maxcpus)
print('array rows: %s'%npday.shape[0])

creategrid = partial(creategrid_xs_small, rdiff, firstid, gridwidth, gridheight)
atimes2=dataset_conv_opt(creategrid, maxcpus, dayfiles)
np.save('conv_thread_eq_files_03',atimes2)

#create_xs_files(creategrid, dayfiles[:1], 1, 2)