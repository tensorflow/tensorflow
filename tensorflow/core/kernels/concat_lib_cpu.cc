/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/concat_lib_cpu.h"
#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"

namespace tensorflow {

namespace {
template <typename T>
struct MemCpyCopier {
  inline void Copy(T* dst, const T* src, int input_index, size_t n) {
    if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
      memcpy(dst, src, n * sizeof(T));
    } else {
      for (size_t k = 0; k < n; ++k) {
        *dst++ = *src++;
      }
    }
  }
};
template <>
struct MemCpyCopier<ResourceHandle> {
  inline void Copy(ResourceHandle* dst, const ResourceHandle* src,
                   int input_index, size_t n) {
    for (size_t k = 0; k < n; ++k) {
      *dst++ = *src++;
    }
  }
};

template <typename T>
int64_t EstimateBytesPerElement(
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs) {
  return sizeof(T);
}

// EstimateBytesPerElement for strings estimates the total bytes involved in
// concatenating the strings in the "inputs" matrices (higher-level code
// reshapes all the inputs to matrices), by sampling the lengths of the actual
// strings in the various tensors.
template <>
int64_t EstimateBytesPerElement<tstring>(
    const std::vector<
        std::unique_ptr<typename TTypes<tstring, 2>::ConstMatrix>>& inputs) {
  // randomly sample a few input strings to get a sense of the average size
  // of each element
  int num_samples = 0;
  int64_t num_bytes_in_samples = 0;
  for (const auto& input : inputs) {
    const auto dim0 = input->dimension(0);
    const auto dim1 = input->dimension(1);
    const auto zero = dim0 - dim0;  // Make type match
    if (dim0 > 0 && dim1 > 0) {
      // Draw 9 samples of string sizes from the input, in this sort of pattern
      // ("*" is sample), to get an estimate of the lengths of each string
      // element in the tensors:
      //
      //    *...*...*
      //    .........
      //    *...*...*
      //    .........
      //    *...*...*
      for (auto i : {zero, dim0 / 2, dim0 - 1}) {
        for (auto j : {zero, dim1 / 2, dim1 - 1}) {
          num_bytes_in_samples += (*input)(i, j).size();
          num_samples++;
        }
      }
    }
  }
  // We don't use sizeof(std::string) as the overhead, since that would
  // overestimate the memory touched for copying a string.
  int64_t string_overhead = sizeof(char*) + sizeof(size_t);
  return string_overhead +
         ((num_samples > 0) ? (num_bytes_in_samples / num_samples) : 0);
}

}  // namespace

template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  int64_t cost_per_unit = EstimateBytesPerElement<T>(inputs);
  ConcatCPUImpl<T>(d, inputs, cost_per_unit, MemCpyCopier<T>(), output);
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);
TF_CALL_ALL_TYPES(REGISTER)
REGISTER(quint8)
REGISTER(qint8)
REGISTER(quint16)
REGISTER(qint16)
REGISTER(qint32)

#if defined(IS_MOBILE_PLATFORM) && !defined(SUPPORT_SELECTIVE_REGISTRATION) && \
    !defined(__ANDROID_TYPES_FULL__)
// Primarily used for SavedModel support on mobile. Registering it here only
// if __ANDROID_TYPES_FULL__ is not defined (which already registers string)
// to avoid duplicate registration.
REGISTER(tstring);
#endif  // defined(IS_MOBILE_PLATFORM) &&
        // !defined(SUPPORT_SELECTIVE_REGISTRATION) &&
        // !defined(__ANDROID_TYPES_FULL__)

}  // namespace tensorflow
