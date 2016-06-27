#pragma once

#include <stdexcept>
#include <vector>
#include <random>

#include "tensorflow/core/kernels/warp-ctc/include/ctc.h"

inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
      printf("error in cpu ctc\n");
    }
}

#ifdef __CUDACC__
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

inline void throw_on_error(cudaError_t error, const char* message) {
    if (error) {
        throw thrust::system_error(error, thrust::cuda_category(), message);
    }
}

#endif

