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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {
namespace functor {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// InTopK argument can be passed either via mode attribute (InTopK op), or as an
// input tensor (InTopKV2 op).
struct TopKArg {
  int64 k_value = -1;
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

    int64 k_val = k.k_value;
    if (k.k_tensor != nullptr) {
      if (k.k_tensor->dtype() == DT_INT32) {
        k_val = k.k_tensor->scalar<int32>()();
      } else {
        k_val = k.k_tensor->scalar<int64>()();
      }
    }

    for (int batch_idx = 0; batch_idx < num_targets; batch_idx++) {
      auto target = internal::SubtleMustCopy(targets(batch_idx));

      bool cannot_say = !FastBoundsCheck(target, num_classes) ||
                        !std::isfinite(predictions(batch_idx, target));

      int more_probable_classes = 0;
      if (!cannot_say) {
        const T target_prediction = predictions(batch_idx, target);

        for (int class_idx = 0; class_idx < num_classes; ++class_idx) {
          T pred = predictions(batch_idx, class_idx);
          if (!std::isfinite(pred)) {
            cannot_say = true;
            break;
          } else if (pred > target_prediction) {
            ++more_probable_classes;
            if (more_probable_classes > k_val) break;
          }
        }
      }
      output(batch_idx) = cannot_say ? false : (more_probable_classes < k_val);
    }
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IN_TOPK_OP_H_
