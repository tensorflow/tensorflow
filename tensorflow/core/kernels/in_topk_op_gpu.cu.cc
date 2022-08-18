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

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/in_topk_op.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// Compare each prediction in 'predictions' with a target prediction for the
// batch, and write result to the 'mask':
//  -1: If the target class is out of range, or if the prediction value is not
//      finite and can't be compared to target prediction (and vice versa).
//   0: If prediction is smaller than the target prediction for the batch.
//   1: If prediction is larger than the target prediction for the batch.
template <typename T, typename TargetT>
__global__ void ComputePredictionMaskKernel(
    const T* __restrict__ predictions,    // dims: [ num_targets x num_classes ]
    const TargetT* __restrict__ targets,  // dims: [ num_targets ]
    int64* __restrict__ mask,             // dims: [ num_targets x num_classes ]
    int num_targets, int num_classes) {
  GPU_1D_KERNEL_LOOP(i, num_targets * num_classes) {
    const int batch_index = i / num_classes;
    TargetT target_idx = ldg(targets + batch_index);

    if (!FastBoundsCheck(target_idx, num_classes)) {
      mask[i] = -1;
      return;
    }

    T prediction = ldg(predictions + i);
    T target_prediction =
        ldg(predictions + batch_index * num_classes + target_idx);

    if (!Eigen::numext::isfinite(prediction) ||
        !Eigen::numext::isfinite(target_prediction)) {
      mask[i] = -1;
    } else {
      mask[i] = prediction > target_prediction ? 1 : 0;
    }
  }
}

// Reduce all prediction masks either to the sum of '1' for each prediction
// larger than the target, or to '-1' if target class in invalid of predictions
// in a batch have non-finite values.
struct MaskSum {
  __host__ __device__ int64 operator()(const int64& a, const int64& b) const {
    if (a < 0 || b < 0)
      return -1;
    else
      return a + b;
  }
};

namespace reduction_op_helper {
template <>
struct IdentityValue<int64, MaskSum> {
  int64 operator()() { return 0; }
};

}  // namespace reduction_op_helper

template <typename T, typename TargetT>
struct InTopKFunctor<GPUDevice, T, TargetT> {
  template <int ndims>
  using Dims = Eigen::DSizes<Eigen::Index, ndims>;

  void operator()(OpKernelContext* context,
                  typename TTypes<T, 2>::ConstTensor predictions,
                  typename TTypes<TargetT>::ConstVec targets, const TopKArg k,
                  typename TTypes<bool>::Vec output) {
    const Eigen::Index num_targets = predictions.dimension(0);
    const Eigen::Index num_classes = predictions.dimension(1);

    OP_REQUIRES(
        context, num_targets * num_classes < std::numeric_limits<int>::max(),
        errors::InvalidArgument(
            "Number of targets * number of classes must be less than INT_MAX"));

    if (num_targets == 0 || num_classes == 0) {
      // Result is empty, so shortcut the rest of the function to avoid
      // launching kernels with empty input.
      return;
    }

    // Temporary storage for a mask computed by  `ComputePredictionMaskKernel`.
    Tensor predictions_mask;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_INT64,
                                        TensorShape({num_targets, num_classes}),
                                        &predictions_mask));

    // Number of predictions for each target that are larger than the target
    // prediction (or -1 if we can't compute this number, because not all
    // predictions are finite or target class is out of range).
    Tensor num_larger_prediction;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_targets}),
                                          &num_larger_prediction));

    const auto& d = context->eigen_device<GPUDevice>();

    // Compute a mask for all predictions.
    GpuLaunchConfig config = GetGpuLaunchConfig(num_targets * num_classes, d);
    OP_REQUIRES_OK(
        context, GpuLaunchKernel(ComputePredictionMaskKernel<T, TargetT>,
                                 config.block_count, config.thread_per_block, 0,
                                 d.stream(), predictions.data(), targets.data(),
                                 predictions_mask.flat<int64_t>().data(),
                                 num_targets, num_classes));

    // Reduce prediction masks to number of predictions larger than the target
    // prediction, or to the negative value if we can't compute an answer.
    {
      auto in = predictions_mask.matrix<int64_t>();
      auto out = num_larger_prediction.flat<int64_t>();

      ReduceImpl<int64, MaskSum, int64*, int64*, Dims<1>>(
          context, (int64*)out.data(), (int64*)in.data(), in.rank(),
          in.dimension(0), in.rank() >= 2 ? in.dimension(1) : 1,
          in.rank() >= 3 ? in.dimension(2) : 1, out.rank(), Dims<1>(1),
          MaskSum());
    }

    // Compute if target prediction is in top K predictions.
    auto cnt = num_larger_prediction.flat<int64_t>();

    if (k.k_tensor != nullptr) {
      if (k.k_tensor->dtype() == DT_INT32) {
        output.device(d) =
            (cnt >= cnt.constant(0)) &&
            (cnt < k.k_tensor->flat<int32>().template cast<int64_t>().broadcast(
                       Dims<1>(num_targets)));
      } else {
        output.device(d) =
            (cnt >= cnt.constant(0)) &&
            (cnt < k.k_tensor->flat<int64_t>().broadcast(Dims<1>(num_targets)));
      }
    } else {
      output.device(d) =
          (cnt >= cnt.constant(0)) && (cnt < targets.constant(k.k_value));
    }
  }
};

}  // namespace functor

// Definition of the GPU implementations declared in in_topk_op.cc.
#define DEFINE_GPU_KERNELS(T, TARGET_T) \
  template struct functor::InTopKFunctor<GPUDevice, T, TARGET_T>;

DEFINE_GPU_KERNELS(float, int32);
DEFINE_GPU_KERNELS(float, int64);

#undef DEFINE_GPU_KERNELS

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
