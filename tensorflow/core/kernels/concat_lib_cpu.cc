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
}  // namespace

template <typename T>
void ConcatCPU(DeviceBase* d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output) {
  ConcatCPUImpl<T>(d, inputs, sizeof(T) /* cost_per_unit */, MemCpyCopier<T>(),
                   output);
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);
TF_CALL_POD_STRING_TYPES(REGISTER)
REGISTER(quint8)
REGISTER(qint8)
REGISTER(quint16)
REGISTER(qint16)
REGISTER(qint32)
REGISTER(bfloat16)

}  // namespace tensorflow
