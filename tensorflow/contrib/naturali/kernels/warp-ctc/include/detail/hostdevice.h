#pragma once

#ifdef __CUDACC__
    #define HOSTDEVICE __host__ __device__
#else
    #define HOSTDEVICE
#endif
