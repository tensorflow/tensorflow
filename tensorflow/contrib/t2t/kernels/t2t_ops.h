#pragma once

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include <inttypes.h>

namespace tensorflow {

template <typename D, typename T, typename U>
struct CustomL2NormFunctor {
  void operator()(const D& d, 
    uint64_t N, uint64_t k,
    const T* in, T* temp, T* out,
    const U* eps, const U* bias, const U* scale
    );  
};

template <typename D, typename T, typename U>
struct CustomL2NormGradFunctor {
  void operator()(const D& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, T* temp, T* out,
    const U* eps, const U* bias, const U* scale
    );
};

template <typename Device, typename T>
struct CustomDropoutFunctor2 {
  void operator()(const Device& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1,
    int s0, int s1,
    int r0, int r1
    );
};

template <typename Device, typename T>
struct CustomDropoutFunctor3 {
  void operator()(const Device& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2
    );
};


template <typename Device, typename T>
struct CustomDropoutFunctor4 {
  void operator()(const Device& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,
    int r0, int r1, int r2, int r3
    );
};

#if GOOGLE_CUDA
template <typename T, typename U>
struct CustomL2NormFunctor<Eigen::GpuDevice, T, U> {
  void operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, float* temp, T* out,
    const U* eps, const U* bias, const U* scale
    );
};

template <typename T, typename U>
struct CustomL2NormGradFunctor<Eigen::GpuDevice, T, U> {
  void operator()(const Eigen::GpuDevice& d, 
    uint64_t N, uint64_t k,
    const T* in, const T* outgrad, float* temp, T* out,
    const U* eps, const U* bias, const U* scale
    );
};

template <typename T>
struct CustomDropoutFunctor2<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1,
    int s0, int s1,
    int r0, int r1
    );
};

template <typename T>
struct CustomDropoutFunctor3<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2,
    int s0, int s1, int s2,
    int r0, int r1, int r2
    );
};


template <typename T>
struct CustomDropoutFunctor4<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d, 
    const T* in,
    const T* rng,
    T* out,
    const T* pthr,
    int d0, int d1, int d2, int d3,
    int s0, int s1, int s2, int s3,
    int r0, int r1, int r2, int r3
    );
};

#endif

};

