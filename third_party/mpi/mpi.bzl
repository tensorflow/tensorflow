#OpenMPI and Mvapich/mpich require different headers
#based on the configuration options return one or the other

def mpi_hdr():
    MPI_LIB_IS_OPENMPI=True
    hdrs = []    
    if MPI_LIB_IS_OPENMPI:
        hdrs = ["mpi.h", "mpi_portable_platform.h"]   #When using OpenMPI
    else:
        hdrs = ["mpi.h",  "mpio.h", "mpicxx.h"]        #When using MVAPICH
    return hdrs
