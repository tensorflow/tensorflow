/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_IN_TOPK_OP_H_
#define TENSORFLOW_CORE_KERNELS_IN_TOPK_OP_H_

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/threadpool.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// InTopK argument can be passed either via mode attribute (InTopK op), or as an
// input tensor (InTopKV2 op).
struct TopKArg {
  int64_t k_value = -1;
  const Tensor* k_tensor = nullptr;
};

template <typename Device, typename T, typename TargetT>
struct InTopKFunctor {
  template <int ndims>
  using Dims = Eigen::DSizes<Eigen::Index, ndims>;

  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor predictions,
                  typename TTypes<TargetT>::ConstVec targets, const TopKArg k,
                  typename TTypes<bool>::Vec output) {}
};

template <typename T, typename TargetT>
struct InTopKFunctor<CPUDevice, T, TargetT> {
  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor predictions,
                  typename TTypes<TargetT>::ConstVec targets, const TopKArg k,
                  typename TTypes<bool>::Vec output) {
    const Eigen::Index num_targets = predictions.dimension(0);
    const Eigen::Index num_classes = predictions.dimension(1);

    int64_t k_val = k.k_value;
    if (k.k_tensor != nullptr) {
      if (k.k_tensor->dtype() == DT_INT32) {
        k_val = k.k_tensor->scalar<int32_t>()();
      } else {
        k_val = k.k_tensor->scalar<int64_t>()();
      }
    }

    // Raw pointers keep the hot loop free of reloads through the captured
    // TensorMaps (the bool output store may otherwise alias them).
    const T* const predictions_data = predictions.data();
    const TargetT* const targets_data = targets.data();
    bool* const output_data = output.data();

    // Returns the number of predictions read, which is the row's actual cost:
    // rows usually stop early once more than k larger predictions are seen.
    auto process_row = [predictions_data, targets_data, output_data,
                        num_classes, k_val](Eigen::Index batch_idx) -> int64_t {
      const T* const row = predictions_data + batch_idx * num_classes;
      auto target = internal::SubtleMustCopy(targets_data[batch_idx]);

      bool cannot_say = !FastBoundsCheck(target, num_classes) ||
                        !std::isfinite(row[static_cast<Eigen::Index>(target)]);

      int64_t more_probable_classes = 0;
      Eigen::Index class_idx = 0;
      if (!cannot_say) {
        const T target_prediction = row[static_cast<Eigen::Index>(target)];

        for (; class_idx < num_classes; ++class_idx) {
          const T pred = row[class_idx];
          if (!std::isfinite(pred)) {
            cannot_say = true;
            break;
          } else if (pred > target_prediction) {
            ++more_probable_classes;
            if (more_probable_classes > k_val) break;
          }
        }
      }
      output_data[batch_idx] =
          cannot_say ? false : (more_probable_classes < k_val);
      // On a full scan class_idx == num_classes; on an early exit it indexes
      // the last element read, so the read count is class_idx + 1.
      return class_idx == num_classes ? class_idx : class_idx + 1;
    };

    // Rows are independent, so they can be sharded across the CPU worker
    // threads. How much of each row is scanned depends on the data, so first
    // sample a few rows spread across the batch and use the observed average
    // to decide whether the rest are worth dispatching.
    constexpr Eigen::Index kSampleRows = 32;
    constexpr int64_t kMinParallelPredictions = 16 * 1024;
    const Eigen::Index sample = std::min(num_targets, kSampleRows);
    std::vector<char> sampled_row(num_targets, 0);
    int64_t sampled = 0;
    for (Eigen::Index s = 0; s < sample; ++s) {
      // Stride across the batch so a cheap prefix cannot hide expensive rows.
      const Eigen::Index batch_idx = (s * num_targets) / sample;
      sampled_row[batch_idx] = 1;
      sampled += process_row(batch_idx);
    }
    if (sample == num_targets) return;
    const int64_t avg_scanned = std::max<int64_t>(1, sampled / sample);
    auto work = [&](int64_t start, int64_t limit) {
      for (int64_t batch_idx = start; batch_idx < limit; ++batch_idx) {
        if (!sampled_row[batch_idx]) process_row(batch_idx);
      }
    };
    // Estimate total work from the sample; small / early-exit batches stay
    // on this thread where ParallelFor dispatch would dominate.
    if (num_targets * avg_scanned < kMinParallelPredictions) {
      work(0, num_targets);
      return;
    }
    context->device()->tensorflow_cpu_worker_threads()->workers->ParallelFor(
        num_targets, 2 * avg_scanned + 10, work);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IN_TOPK_OP_H_
