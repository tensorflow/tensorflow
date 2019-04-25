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

// NOTE: Eigen is slightly different internally and externally. We need to
// hack the unsupported/Eigen/CXX11/Tensor header instantiation macros at
// specific places, so we need two copies of the hacked file, one for
// internal and one for external.
// If you have trouble simply undef out the reducer macro e.g.
// TFLITE_REDUCE_INSTANTIATIONS_GOOGLE, but be aware this will make
// the binary much bigger!
#define TFLITE_REDUCE_INSTANTIATIONS_OPEN_SOURCE
#define Eigen EigenForTFLite
#if defined(TFLITE_REDUCE_INSTANTIATIONS_GOOGLE)
#include "tensorflow/lite/kernels/internal/optimized/eigen_tensor_reduced_instantiations_google.h"
#elif defined(TFLITE_REDUCE_INSTANTIATIONS_OPEN_SOURCE)
#include "tensorflow/lite/kernels/internal/optimized/eigen_tensor_reduced_instantiations_oss.h"
#else
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif

#include "tensorflow/core/kernels/eigen_spatial_convolutions-inl.h"

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
