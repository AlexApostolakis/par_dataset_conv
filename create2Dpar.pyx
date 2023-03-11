import numpy as np
cimport numpy as cnp
import cython
from cython.parallel import prange
INT32 = np.int32
ctypedef cnp.int32_t INT32_t
ctypedef cnp.float32_t FLOAT32_t
from cython cimport view
from libc.math cimport atan
import math

cnp.import_array()

cdef long* get_grid_xy(long _id, long firstid, long rdiff) nogil:
    cdef long[2] res
    res[0] = int((_id-firstid)/rdiff) #row
    res[1] = _id-firstid-rdiff*res[0] #col
    return res

cdef long get_grid_x(long _id, long firstid, long rdiff, long row) nogil:
    cdef int col
    col = _id-firstid-rdiff*row #col
    return col

cdef long get_grid_y(long _id, long firstid, long rdiff) nogil:
    cdef int row
    row = int((_id-firstid)/rdiff) #row
    return row


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def npapplypar(int nt, cnp.ndarray[INT32_t, ndim=1] ids, long firstid, long rdiff):
    cdef int i, M, res
    M = ids.shape[0]
    id2xy = view.array(shape=(M,2), itemsize=sizeof(long), format="l")
    cdef long [:, :] id2xy_mv = id2xy
    cdef int [:] ids_mv = ids
    for i in prange(M, num_threads=nt, nogil=True):
        id2xy_mv[i,1]=get_grid_y(ids_mv[i], firstid, rdiff)
        id2xy_mv[i,0]=get_grid_x(ids_mv[i], firstid, rdiff, id2xy_mv[i,1])
    id2xypy=np.asarray(id2xy_mv)
    return id2xypy

#cdef float some_function_not_requiring_gil(float[:] x) nogil:
#    return x[0]

@cython.boundscheck(False)
cdef void setgridxy(int i, long [:, :] id2xy_mv, float [:, :, :] grid_mv, float [:, :] ids_mv, long firstid, long rdiff, int intensive=0) nogil:
    cdef float p
    cdef float p2
    id2xy_mv[i, 1] = get_grid_y(int(ids_mv[i, 0]), firstid, rdiff)
    id2xy_mv[i, 0] = get_grid_x(int(ids_mv[i, 0]), firstid, rdiff, id2xy_mv[i, 1])
    if intensive!=-1:
        grid_mv[id2xy_mv[i, 0] - 1, id2xy_mv[i, 1] - 1] = ids_mv[i]
    if intensive>0:
        while intensive>0:
            p = intensive**intensive
            p2 = atan(p)
            intensive-=1

# @cython.wraparound(False)
@cython.boundscheck(False)
# @cython.nonecheck(False)
def fillcube(int nt, cnp.ndarray[FLOAT32_t, ndim=2] tab, long firstid, long rdiff, gW, gH, sched=None, chunks='auto', intensive=0):
    cdef int i, M, N, res
    M = tab.shape[0]
    id2xy = view.array(shape=(M,2), itemsize=sizeof(long), format="l")
    #grid = view.array(shape=(gW,gH,tab.shape[1]), itemsize=sizeof(float), format="f")
    grid = np.empty((gW,gH,tab.shape[1]), dtype=np.float32)
    grid[:,:] = np.nan
    #grid = np.nan
    cdef float [:, :, :] grid_mv = grid
    cdef long [:, :] id2xy_mv = id2xy
    cdef float [:, :] ids_mv = tab
    cdef long rdiff1 = rdiff
    cdef long firstid1 = firstid
    cdef int threads=math.ceil(M/nt)
    cdef int autochunk=math.floor(M/nt)
    cdef int chunk
    cdef int intensiveness=intensive

    if chunks == 'auto': chunk=autochunk
    else: chunk=chunks
    if sched=='static':
        for i in prange(M, num_threads=nt, nogil=True, schedule='static', chunksize=chunk):
        #with gil:
        #    print(ids_mv[i,0])
            setgridxy(i, id2xy_mv, grid_mv, ids_mv, firstid1, rdiff1, intensiveness)
    elif sched == 'guided':
        for i in prange(M, num_threads=nt, nogil=True, schedule='guided'):
            setgridxy(i, id2xy_mv, grid_mv, ids_mv, firstid1, rdiff1, intensiveness)
    elif sched == 'dynamic':
        for i in prange(M, num_threads=nt, nogil=True, schedule='dynamic'):
            setgridxy(i, id2xy_mv, grid_mv, ids_mv, firstid1, rdiff1, intensiveness)
    else:
        for i in prange(M, num_threads=nt, nogil=True):
            setgridxy(i, id2xy_mv, grid_mv, ids_mv, firstid1, rdiff1, intensiveness)


    id2xy_py=np.asarray(id2xy_mv)
    grid_py = np.asarray(grid_mv)
    return id2xy_py, grid_py
