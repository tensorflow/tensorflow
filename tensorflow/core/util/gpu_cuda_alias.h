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

#ifndef TENSORFLOW_CORE_UTIL_GPU_CUDA_ALIAS_H_
#define TENSORFLOW_CORE_UTIL_GPU_CUDA_ALIAS_H_

// Several forwarding macros are defined in this file to serve for backward
// compatibility usage as we migrating from CUDA prefixed function to GPU
// prefixed functions. Both Cuda and ROCm can unify under the new GPU prefix
// naming scheme. In the migration period, we provide equivalent CUDA* and GPU*
// function. Over time, all CUDA* functions will be deprecated.

namespace tensorflow {

// CREATE_CUDA_HOST_FUNCTION_ALIAS forward the host function to its CUDA Alias.
#ifndef TENSORFLOW_USE_ROCM
#define CREATE_CUDA_HOST_FUNCTION_ALIAS(func, cuda_alias) \
  template <typename... Args>                             \
  auto cuda_alias(Args&&... args)                         \
      ->decltype(func(std::forward<Args>(args)...)) {     \
    return func(std::forward<Args>(args)...);             \
  }
#else
#define CREATE_CUDA_HOST_FUNCTION_ALIAS(func, cuda_alias)
#endif

// CREATE_CUDA_DEVICE_FUNCTION_ALIAS forward the device function to its CUDA
// Alias.
#ifndef TENSORFLOW_USE_ROCM
#define CREATE_CUDA_DEVICE_FUNCTION_ALIAS(func, cuda_alias) \
  template <typename... Args>                               \
  __device__ auto cuda_alias(Args&&... args)                \
      ->decltype(func(std::forward<Args>(args)...)) {       \
    return func(std::forward<Args>(args)...);               \
  }
#else
#define CREATE_CUDA_DEVICE_FUNCTION_ALIAS(func, cuda_alias)
#endif

// CREATE_CUDA_TYPE_ALIAS forward the type to its CUDA Alias.
#ifndef TENSORFLOW_USE_ROCM
#define CREATE_CUDA_TYPE_ALIAS(type, cuda_alias) using cuda_alias = type;
#else
#define CREATE_CUDA_TYPE_ALIAS(type, cuda_alias)
#endif
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_GPU_CUDA_ALIAS_H_
