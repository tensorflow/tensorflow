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
#include <vector>

#include "absl/container/inlined_vector.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/concat_lib.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

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
  const size_t num_inputs = inputs.size();
  const int64_t estimated_total_cost = output->size() * cost_per_unit;
  auto worker_threads = d->tensorflow_cpu_worker_threads();
  const int64_t dim0 = output->dimension(0);
  // Cap at 4 threads: memory-bound copies saturate RAM bandwidth quickly;
  // beyond 4 threads synchronization and cache thrash regress large tensors
  // on high-core-count processors.
  const int num_threads = std::min(4, worker_threads->num_threads);

  // Fast-path for single-threaded contiguous concatenation along the outermost
  // dimension (dim0 == 1): avoid constructing sizes and row_size arrays.
  // Guard with size > 0 to safely skip 0-sized tensors with null data pointers.
  // Threshold is 64 KiB, consistent with the multi-row single-threaded cutoff.
  if (dim0 == 1 &&
      (estimated_total_cost < 65536 || worker_threads->num_threads <= 1)) {
    T* out = output->data();
    for (size_t j = 0; j < num_inputs; ++j) {
      const ptrdiff_t size = inputs[j]->dimension(1);
      if (size > 0) {
        copier.Copy(out, inputs[j]->data(), j, size);
        out += size;
      }
    }
    return;
  }

  // Multi-threaded contiguous copy across slices when dim0 == 1.
  // Uses a direct sequential walk over inputs within each shard range,
  // avoiding std::upper_bound binary search and its std::distance overhead.
  if (dim0 == 1) {
    absl::InlinedVector<int64_t, 17> offsets(num_inputs + 1, 0);
    absl::InlinedVector<ptrdiff_t, 16> sizes(num_inputs);
    for (size_t j = 0; j < num_inputs; ++j) {
      sizes[j] = inputs[j]->dimension(1);
      offsets[j + 1] = offsets[j] + sizes[j];
    }

    T* const out_base = output->data();
    auto work = [&offsets, &sizes, &inputs, out_base, &copier,
                 num_inputs](int64_t start, int64_t end) {
      // Walk inputs sequentially to find the starting input for this shard.
      // Avoids std::upper_bound + std::distance overhead.
      size_t j = 0;
      while (j + 1 < num_inputs && offsets[j + 1] <= start) {
        ++j;
      }
      int64_t cur = start;
      while (cur < end && j < num_inputs) {
        if (sizes[j] == 0) {
          ++j;
          continue;
        }
        int64_t in_offset = cur - offsets[j];
        int64_t copy_size = std::min<int64_t>(
            static_cast<int64_t>(sizes[j]) - in_offset, end - cur);
        copier.Copy(out_base + cur, inputs[j]->data() + in_offset, j,
                    copy_size);
        cur += copy_size;
        ++j;
      }
    };
    Shard(num_threads, worker_threads->workers, output->size(), cost_per_unit,
          work);
    return;
  }

  // Multi-row concatenation (dim0 > 1):
  absl::InlinedVector<ptrdiff_t, 16> sizes;
  sizes.reserve(num_inputs);
  int64_t row_size = 0;
  for (const auto& input : inputs) {
    sizes.push_back(input->dimension(1));
    row_size += sizes.back();
  }

  // Single threaded mode: threshold is 64KB to avoid thread pool
  // synchronization overhead where single-threaded memcpy dominates.
  if (estimated_total_cost < 65536 || worker_threads->num_threads <= 1) {
    T* out = output->data();
    for (int64_t i = 0; i < dim0; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        const ptrdiff_t size = sizes[j];
        if (size > 0) {
          copier.Copy(out, inputs[j]->data() + i * size, j, size);
          out += size;
        }
      }
    }
    return;
  }

  // Sharded mode for multi-row concatenation.
  auto work = [&row_size, &sizes, &inputs, &output, &copier, num_inputs,
               dim0](int64_t start, int64_t end) {
    int64_t skipped_rows = start / row_size;
    T* out = output->data() + skipped_rows * row_size;
    T* out_start = output->data() + start;
    T* out_end = output->data() + end;

    // Handle partial row at start
    if (out < out_start) {
      for (size_t j = 0; j < num_inputs; ++j) {
        ptrdiff_t size = sizes[j];
        // Skip 0-sized inputs: data() == nullptr; pointer arithmetic is UB.
        if (size == 0) continue;
        ptrdiff_t offset = out_start - out;
        if (size <= offset) {
          out += size;
          continue;
        }
        const T* inp = inputs[j]->data() + skipped_rows * sizes[j];
        if (offset > 0) {
          out += offset;
          inp += offset;
          size -= offset;
        }
        size = std::min(size, static_cast<ptrdiff_t>(out_end - out));
        if (size <= 0) break;
        copier.Copy(out, inp, j, size);
        out += size;
      }
      ++skipped_rows;
    }
    if (out == out_end) return;
    CHECK(out >= out_start);
    CHECK(out < out_end);

    // Copy remaining data with true zero heap allocations on worker threads.
    for (int64_t i = skipped_rows; i < dim0; ++i) {
      for (size_t j = 0; j < num_inputs; ++j) {
        const ptrdiff_t size =
            std::min(sizes[j], static_cast<ptrdiff_t>(out_end - out));
        if (size > 0) {
          copier.Copy(out, inputs[j]->data() + i * sizes[j], j, size);
          out += size;
        }
        if (out == out_end) return;
      }
    }
  };
  Shard(num_threads, worker_threads->workers, output->size(), cost_per_unit,
        work);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONCAT_LIB_CPU_H_
