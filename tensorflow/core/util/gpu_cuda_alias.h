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

namespace tensorflow {

#ifndef TENSORFLOW_USE_ROCM
  #define CREATE_CUDA_HOST_FUNCTION_ALIAS(func, cuda_alias) \
  template <typename... Args> \
  auto cuda_alias(Args&&... args) -> decltype(func(std::forward<Args>(args)...)) { \
    return func(std::forward<Args>(args)...); \
  }
#else
  #define CREATE_CUDA_HOST_FUNCTION_ALIAS(func, cuda_alias)
#endif

#ifndef TENSORFLOW_USE_ROCM
  #define CREATE_CUDA_DEVICE_FUNCTION_ALIAS(func, cuda_alias) \
  template <typename... Args> \
  __device__ auto cuda_alias(Args&&... args) -> decltype(func(std::forward<Args>(args)...)) { \
    return func(std::forward<Args>(args)...); \
  }
#else
  #define CREATE_CUDA_DEVICE_FUNCTION_ALIAS(func, cuda_alias)
#endif


#ifndef TENSORFLOW_USE_ROCM
  #define CREATE_CUDA_TYPE_ALIAS(type, cuda_alias) \
  using cuda_alias = type;
#else
  #define CREATE_CUDA_TYPE_ALIAS(type, cuda_alias)
#endif
}


