/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_

#define EIGEN_USE_CUSTOM_THREAD_POOL
#define EIGEN_USE_THREADS

#define Eigen EigenForTFLite

// NOTE: We need to define our own tensor contraction dispatch method before
// including the unsupported/Eigen/CXX11/Tensor header in order to reduce the
// total number of kernel instantiations.
// If you have trouble simply undef out the reducer macro e.g.
// TFLITE_REDUCE_INSTANTIATIONS, but be aware this will make
// the binary much bigger!
#define TFLITE_REDUCE_INSTANTIATIONS
#if defined(TFLITE_REDUCE_INSTANTIATIONS)
// Override Eigen tensor contraction dispatch method.
#define TENSOR_CONTRACTION_DISPATCH(METHOD, ALIGNMENT, ARGS)                  \
  if (this->m_lhs_inner_dim_contiguous && this->m_rhs_inner_dim_contiguous && \
      !this->m_rhs_inner_dim_reordered) {                                     \
    METHOD<true, true, false, ALIGNMENT> ARGS;                                \
  } else {                                                                    \
    eigen_assert(false && "Unsupported contraction formats");                 \
  }
#endif

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tsl/framework/convolution/eigen_spatial_convolutions-inl.h"

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
