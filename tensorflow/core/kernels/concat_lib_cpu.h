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

#ifndef TENSORFLOW_CORE_KERNELS_CONCAT_LIB_CPU_H_
#define TENSORFLOW_CORE_KERNELS_CONCAT_LIB_CPU_H_

#define EIGEN_USE_THREADS

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {
namespace internal {
// Require >= 2 MiB of raw copy work per thread (< 4 MiB runs inline on a
// single thread) to avoid thread-pool wakeup and cross-CCX L3 thrashing.
inline constexpr int64_t kConcatBytesPerThread = 2LL * 1024 * 1024;
inline constexpr int64_t kConcatNonRawCostPerThread = 256LL * 1024;
}  // namespace internal

template <typename T, typename ElementCopier, bool kRawBytes = false>
void ConcatCPUSlicesImpl(DeviceBase* device,
                         absl::Span<const ConcatSlice<T>> slices, int64_t dim0,
                         int64_t row_size, int64_t cost_per_unit,
                         ElementCopier copier, T* output_data) {
  const ConcatSlice<T>* slices_data = slices.data();
  const size_t num_inputs = slices.size();
  const int64_t output_size = dim0 * row_size;
  if (output_size <= 0 || num_inputs == 0) return;

  auto copy_rows = [slices_data, num_inputs, row_size, output_data, &copier](
                       int64_t r0, int64_t r1) {
    T* out = output_data + r0 * row_size;
    for (int64_t i = r0; i < r1; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        const ptrdiff_t sz = slices_data[j].dim1;
        if constexpr (kRawBytes) {
          if (sz == 1) {
            *out++ = slices_data[j].data[i];
            continue;
          }
        }
        if (sz > 0) {
          copier.Copy(out, slices_data[j].data + i * sz, static_cast<int>(j),
                      sz);
          out += sz;
        }
      }
    }
  };

  constexpr int64_t kBudgetPerThread =
      kRawBytes ? internal::kConcatBytesPerThread
                : internal::kConcatNonRawCostPerThread;
  if (output_size * cost_per_unit < 2 * kBudgetPerThread) {
    copy_rows(0, dim0);
    return;
  }
  const DeviceBase::CpuWorkerThreads* worker_threads =
      device->tensorflow_cpu_worker_threads();
  const int max_threads =
      std::min(worker_threads->num_threads, GetPerThreadMaxParallelism());
  const int64_t eff_threads = std::min<int64_t>(
      max_threads, (output_size * cost_per_unit) / kBudgetPerThread);
  if (eff_threads <= 1) {
    copy_rows(0, dim0);
    return;
  }

  // Partition work into at most `eff_threads` contiguous chunks (>= 2 MiB each
  // for raw types) rather than fine-grained `Shard`/`parallelFor` micro-blocks,
  // preserving per-core L3 locality and avoiding per-block division overhead.
  auto run_chunks = [&](int64_t total_units, const auto& fn) {
    const int64_t nchunks = std::min<int64_t>(total_units, eff_threads);
    const int64_t chunk = (total_units + nchunks - 1) / nchunks;
    worker_threads->workers->ParallelFor(
        nchunks, [&fn, total_units, chunk](int64_t c0, int64_t c1) {
          const int64_t start = c0 * chunk;
          const int64_t end = std::min<int64_t>(total_units, c1 * chunk);
          if (start < end) fn(start, end);
        });
  };

  // Shard along whole outer rows whenever dim0 >= eff_threads.
  if (dim0 >= eff_threads) {
    run_chunks(dim0, copy_rows);
    return;
  }

  // Fallback for wide rows with small dim0 (1 <= dim0 < eff_threads).
  run_chunks(output_size, [row_size, slices_data, output_data, &copier,
                           num_inputs, dim0](int64_t start, int64_t end) {
    int64_t skipped_rows = start / row_size;
    T* out = output_data + skipped_rows * row_size;
    T* out_start = output_data + start;
    T* out_end = output_data + end;

    if (out < out_start) {
      for (size_t j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = slices_data[j].dim1;
        if (size == 0) continue;
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T* inp = slices_data[j].data + skipped_rows * size;
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, static_cast<ptrdiff_t>(out_end - out));
        if (size <= 0) break;
        copier.Copy(out, inp, static_cast<int>(j), size);
        out += size;
      }
      ++skipped_rows;
    }
    if (out == out_end) return;

    for (int64_t i = skipped_rows; i < dim0; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        const ptrdiff_t slice_dim1 = slices_data[j].dim1;
        const ptrdiff_t size =
            std::min(slice_dim1, static_cast<ptrdiff_t>(out_end - out));
        if (size > 0) {
          copier.Copy(out, slices_data[j].data + i * slice_dim1,
                      static_cast<int>(j), size);
          out += size;
        }
        if (out == out_end) return;
      }
    }
  });
}

// ElementCopier must be a struct with a single Copy function, which is passed
// the output pointer, input pointer, input index, and number of elements to
// copy from input to output.
template <typename T, typename ElementCopier>
void ConcatCPUImpl(
    DeviceBase* d,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    int64_t cost_per_unit, ElementCopier copier,
    typename TTypes<T, 2>::Matrix* output) {
  absl::InlinedVector<ConcatSlice<T>, 32> slices(inputs.size());
  for (size_t j = 0; j < inputs.size(); ++j) {
    slices[j] = ConcatSlice<T>{inputs[j]->data(), inputs[j]->dimension(1)};
  }
  ConcatCPUSlicesImpl<T, ElementCopier, /*kRawBytes=*/false>(
      d, slices, output->dimension(0), output->dimension(1), cost_per_unit,
      copier, output->data());
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONCAT_LIB_CPU_H_
