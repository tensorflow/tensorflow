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

}  // namespace

template <typename T>
void ConcatCPU(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  if (std::is_same<T, string>::value) {
    // use a large cost here to force strings to be handled by separate threads
    ConcatCPUImpl<T>(d, inputs, 100000, MemCpyCopier<T>(), output);
  } else {
    ConcatCPUImpl<T>(d, inputs, sizeof(T) /* cost_per_unit */,
                     MemCpyCopier<T>(), output);
  }
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
REGISTER(bfloat16)

#if defined(IS_MOBILE_PLATFORM) && !defined(SUPPORT_SELECTIVE_REGISTRATION) && \
    !defined(__ANDROID_TYPES_FULL__)
    // Primarily used for SavedModel support on mobile. Registering it here only
    // if __ANDROID_TYPES_FULL__ is not defined (which already registers string)
    // to avoid duplicate registration.
    REGISTER(string);
#endif  // defined(IS_MOBILE_PLATFORM) &&
        // !defined(SUPPORT_SELECTIVE_REGISTRATION) &&
        // !defined(__ANDROID_TYPES_FULL__)

#ifdef TENSORFLOW_USE_SYCL
template <typename T>
void ConcatSYCL(
    const Eigen::SyclDevice& d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    typename TTypes<T, 2>::Matrix* output) {
  ConcatSYCLImpl<T>(d, inputs, sizeof(T) /* cost_per_unit */, MemCpyCopier<T>(),
                    output);
}
#define REGISTER_SYCL(T)                                                       \
  template void ConcatSYCL<T>(                                                 \
      const Eigen::SyclDevice&,                                                \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_SYCL)

#undef REGISTER_SYCL
#endif  // TENSORFLOW_USE_SYCL
}  // namespace tensorflow
