#pragma once

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include <inttypes.h>

namespace tensorflow {

template <typename D, typename T>
struct CustomL2NormFunctor {
  void operator()(const D& d, 
    uint64_t N, uint64_t k,
    const T* in, T* temp, T* out,
    const float* eps, const float* bias, const float* scale
    );  
};

template <typename D, typename T>
struct CustomL2NormGradFunctor {
  void operator()(const D& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, T* temp, T* out,
    const float* eps, const float* bias, const float* scale);
};

#if GOOGLE_CUDA
template <typename T>
struct CustomL2NormFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const float* eps,
    const float* bias, const float* scale);
};

template <typename T>
struct CustomL2NormGradFunctor<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, float* temp, T* out,
    const float* eps, const float* bias, const float* scale);
};


#endif

};

