#if TENSORFLOW_USE_ROCM

#ifndef _ROCM_KERNEL_HELPER_H
#define _ROCM_KERNEL_HELPER_H

#define cub hipcub
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaStream_t hipStream_t
#define cudaGetLastError hipGetLastError
#define cudaError hipError

#define CUDA_1D_KERNEL_LOOP(i, n)                                   \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
        i += blockDim.x * gridDim.x)

#define GetCudaStream(context) context->eigen_gpu_device().stream()

#endif

#endif