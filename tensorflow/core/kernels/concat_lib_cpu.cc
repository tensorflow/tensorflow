/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/concat_lib.h"
#include <vector>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

template <typename T>
static inline void Copy(T* dst, const T* src, size_t n) {
  if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
    memcpy(dst, src, n * sizeof(T));
  } else {
    for (size_t k = 0; k < n; ++k) {
      *dst++ = *src++;
    }
  }
}

template <typename T>
void ConcatCPU(DeviceBase* d,
               const std::vector<
                   std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>& inputs,
               typename TTypes<T, 2>::Matrix* output) {
  size_t num_inputs = inputs.size();

  std::vector<ptrdiff_t> sizes;
  sizes.reserve(num_inputs);
  int64 row_size = 0;
  for (const auto& input : inputs) {
    sizes.push_back(input->dimension(1));
    row_size += sizes.back();
  }

  auto worker_threads = d->tensorflow_cpu_worker_threads();
  int num_threads = static_cast<int>(std::min<int64>(
      std::min(4, worker_threads->num_threads), output->size() / 4096));
  // Single threaded mode.
  // TODO(dga):  Deduplicate this code w.r.t. sharded code below.
  if (num_threads == 0) {
    T* out = &(*output)(0, 0);
    std::vector<const T*> inp;
    inp.reserve(num_inputs);
    for (const auto& input : inputs) {
      inp.push_back(&(*input)(0, 0));
    }
    const int64 dim0 = output->dimension(0);
    for (int64 i = 0; i < dim0; ++i) {
      for (int64 j = 0; j < num_inputs; ++j) {
        auto size = sizes[j];
        Copy(out, inp[j], size);
        out += size;
        inp[j] += size;
      }
    }
    return;
  }

  // Sharded mode.
  auto work = [&row_size, &sizes, &inputs, &output, &num_inputs](int64 start,
                                                                 int64 end) {
    int64 skipped_rows = start / row_size;
    T* out = output->data() + skipped_rows * row_size;
    T* out_start = output->data() + start;
    T* out_end = output->data() + end;

    // Handle partial row at start
    if (out < out_start) {
      for (size_t j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = sizes[j];
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T* inp = &(*inputs[j])(skipped_rows, 0);
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, out_end - out);
        if (size <= 0) break;
        Copy(out, inp, size);
        out += size;
      }
      ++skipped_rows;
    }
    if (out == out_end) return;
    CHECK(out >= out_start);
    CHECK(out < out_end);

    // Copy remaining data.
    std::vector<const T*> inp;
    inp.reserve(num_inputs);
    for (const auto& input : inputs) {
      inp.push_back(&(*input)(skipped_rows, 0));
    }
    const int64 dim0 = output->dimension(0);
    for (int64 i = skipped_rows; i < dim0; ++i) {
      for (int64 j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = std::min(sizes[j], out_end - out);
        Copy(out, inp[j], size);
        out += size;
        inp[j] += size;
        if (out == out_end) return;
      }
    }
  };
  Shard(num_threads, worker_threads->workers, output->size(), 100, work);
}

#define REGISTER(T)                                                            \
  template void ConcatCPU<T>(                                                  \
      DeviceBase*,                                                             \
      const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&, \
      typename TTypes<T, 2>::Matrix* output);
TF_CALL_ALL_TYPES(REGISTER)
REGISTER(quint8)
REGISTER(qint8)
REGISTER(qint32)
REGISTER(bfloat16)

}  // namespace tensorflow
