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

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/platform/tstring.h"

namespace tensorflow {

namespace {
template <typename T>
struct MemCpyCopier {
  inline void Copy(T* dst, const T* src, int input_index, size_t n) {
    if constexpr (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
      std::memcpy(dst, src, n * sizeof(T));
    } else {
      for (size_t k = 0; k < n; ++k) {
        *dst++ = *src++;
      }
    }
  }
};

// Specializes `tstring` copy is required here because `tstring` can be of
// view-type. This means that the underlying `tstring` lifetime cannot be
// controlled by this operation.
// For example, if `*src` is a view from a `Tensor` and we invoke
// the default copy constructor here to shallow copy `*src` to `*dst`, but later
// `Tensor` that owns `*src` is freed, we will be in use-after-free trouble.
// Therefore, we should always copy `tstring` here to avoid use-after-free
// due to `tstring` view type. This also matches the semantic of `MemCpyCopier`
// to always copy data from upstream.
template <>
struct MemCpyCopier<tstring> {
  inline void Copy(tstring* dst, const tstring* src, int input_index,
                   size_t n) {
    for (size_t k = 0; k < n; ++k) {
      // Copies `src` to `cpy` to decouple the lifetime from the original
      // `src` string which can be potentially a `view`.
      (*dst++).assign(src->data(), src->size());
      src++;
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
int64_t EstimateBytesPerElement(absl::Span<const ConcatSlice<T>>, int64_t) {
  return sizeof(T);
}

template <>
int64_t EstimateBytesPerElement<tstring>(
    absl::Span<const ConcatSlice<tstring>> slices, int64_t dim0) {
  int num_samples = 0;
  int64_t num_bytes_in_samples = 0;
  const size_t num_inputs = slices.size();
  if (dim0 > 0) {
    const size_t step = std::max<size_t>(1, num_inputs / 4);
    for (size_t k = 0; k < num_inputs; k += step) {
      const ptrdiff_t dim1 = slices[k].dim1;
      if (dim1 > 0 && slices[k].data != nullptr) {
        const int64_t total = dim0 * static_cast<int64_t>(dim1);
        num_bytes_in_samples += slices[k].data[0].size();
        num_bytes_in_samples += slices[k].data[total / 2].size();
        num_bytes_in_samples += slices[k].data[total - 1].size();
        num_samples += 3;
      }
    }
  }
  const int64_t string_overhead = sizeof(char*) + sizeof(size_t);
  return string_overhead +
         ((num_samples > 0) ? (num_bytes_in_samples / num_samples) : 0);
}

}  // namespace

template <typename T>
void ConcatCPUSlices(DeviceBase* device,
                     absl::Span<const ConcatSlice<T>> slices, int64_t dim0,
                     int64_t row_size, T* output_data) {
  const int64_t cost_per_unit = EstimateBytesPerElement<T>(slices, dim0);
  ConcatCPUSlicesImpl<T, MemCpyCopier<T>,
                      DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())>(
      device, slices, dim0, row_size, cost_per_unit, MemCpyCopier<T>(),
      output_data);
}

template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  absl::InlinedVector<ConcatSlice<T>, 32> slices(inputs.size());
  for (size_t j = 0; j < inputs.size(); ++j) {
    slices[j] = ConcatSlice<T>{inputs[j]->data(), inputs[j]->dimension(1)};
  }
  ConcatCPUSlices<T>(d, slices, output->dimension(0), output->dimension(1),
                     output->data());
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);                                  \
  template void ConcatCPUSlices<T>(                                            \
      DeviceBase*, absl::Span<const ConcatSlice<T>>, int64_t, int64_t, T*);
TF_CALL_ALL_TYPES(REGISTER)
TF_CALL_float8_e5m2(REGISTER) TF_CALL_float8_e4m3fn(REGISTER) REGISTER(quint8)
    REGISTER(qint8) REGISTER(quint16) REGISTER(qint16) REGISTER(qint32)

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
