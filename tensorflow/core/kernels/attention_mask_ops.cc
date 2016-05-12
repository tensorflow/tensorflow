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

// See docs in ../ops/image_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/core/kernels/attention_mask_ops.h"

#include <vector>
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device>
class AttentionMaskOp : public OpKernel {
 public:
  explicit AttentionMaskOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_value", &fill_value_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    const Tensor* input_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    const int64 batch_size = sequence_len_tensor->dim_size(0);
    
    OP_REQUIRES(
        ctx, input_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument(
            "input_tensor.dims(0) must equal batch_size, (",
            input_tensor->dim_size(0), " vs. ", batch_size, ")"));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                             &output_tensor));

    functor::AttentionMask<Device>::Compute(
        ctx->eigen_device<Device>(), fill_value_,
        sequence_len_tensor->vec<int64>(), input_tensor->matrix<float>(),
        output_tensor->matrix<float>());
  }

 private:
  float fill_value_;
};

REGISTER_KERNEL_BUILDER(Name("AttentionMask")
                             .Device(DEVICE_CPU),
                        AttentionMaskOp<CPUDevice>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void AttentionMask<GPUDevice>::Compute(
      const GPUDevice& d, float fill_value,
      typename TTypes<int64>::ConstVec sequence_len,
      typename TTypes<float>::ConstMatrix input,
      typename TTypes<float>::Matrix output);
  extern template struct AttentionMask<GPUDevice>;
}  // end namespace functor

REGISTER_KERNEL_BUILDER(Name("AttentionMask")
                             .Device(DEVICE_GPU),
                        AttentionMaskOp<GPUDevice>);
#endif  // GOOGLE_CUDA

template <typename Device>
class AttentionMaskMedianOp : public OpKernel {
 public:
  explicit AttentionMaskMedianOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fill_value", &fill_value_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_l", &window_l_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("window_r", &window_r_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* sequence_len_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("sequence_len", &sequence_len_tensor));

    const Tensor* input_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("input", &input_tensor));

    const Tensor* prev_alignment_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->input("prev_alignment", &prev_alignment_tensor));

    const int64 batch_size = input_tensor->dim_size(0);

    OP_REQUIRES(
        ctx, input_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument(
            "input_tensor.dims(0) must equal batch_size, (",
            input_tensor->dim_size(0), " vs. ", batch_size, ")"));
    
    OP_REQUIRES(
        ctx, prev_alignment_tensor->dim_size(0) == batch_size,
        errors::InvalidArgument(
            "prev_alignment_tensor.dims(0) must equal batch_size, (",
            prev_alignment_tensor->dim_size(0), " vs. ", batch_size, ")"));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output("output", input_tensor->shape(),
                                             &output_tensor));

    Tensor median_tensor;
    OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_INT64, TensorShape({batch_size}),
                                           &median_tensor));

    functor::ComputeMedian<Device>::Compute(
        ctx->eigen_device<Device>(), prev_alignment_tensor->matrix<float>(),
        median_tensor.vec<int64>());

    const Tensor& const_median_tensor = median_tensor;
    functor::AttentionMaskMedian<Device>::Compute(
        ctx->eigen_device<Device>(), fill_value_, window_l_, window_r_,
        sequence_len_tensor->vec<int64>(), input_tensor->matrix<float>(),
        const_median_tensor.vec<int64>(), output_tensor->matrix<float>());
  }

 private:
  float fill_value_;
  int64 window_l_;
  int64 window_r_;
};

namespace functor {
template<>
void ComputeMedian<CPUDevice>::Compute(
    const CPUDevice& d, typename TTypes<float>::ConstMatrix input,
    typename TTypes<int64>::Vec median) {
  const int64 batch_size = input.dimensions()[0];
  const int64 dist_size = input.dimensions()[1];

  for (int64 b = 0; b < batch_size; ++b) {
    float sum = 0.0;
    int64 median_idx = 0;
    for (; median_idx < dist_size; ++median_idx) {
      sum += input(b, median_idx);
      if (sum >= 0.5f) {
        break;
      }
    }

    median(b) = median_idx;
  }
}
}  // end namespace functor

REGISTER_KERNEL_BUILDER(Name("AttentionMaskMedian")
                             .Device(DEVICE_CPU),
                        AttentionMaskMedianOp<CPUDevice>);

#if GOOGLE_CUDA
namespace functor {
  template <>
  void ComputeMedian<GPUDevice>::Compute(
      const GPUDevice& d, typename TTypes<float>::ConstMatrix input,
      typename TTypes<int64>::Vec median);

  template <>
  void AttentionMaskMedian<GPUDevice>::Compute(
      const GPUDevice& d, float fill_value, int64 window_l, int64 window_r,
      typename TTypes<int64>::ConstVec sequence_len,
      typename TTypes<float>::ConstMatrix input,
      typename TTypes<int64>::ConstVec median,
      typename TTypes<float>::Matrix output);

  extern template struct ComputeMedian<GPUDevice>;
  extern template struct AttentionMaskMedian<GPUDevice>;
}  // end namespace functor

REGISTER_KERNEL_BUILDER(Name("AttentionMaskMedian")
                             .Device(DEVICE_GPU),
                        AttentionMaskMedianOp<GPUDevice>);
#endif  // GOOGLE_CUDA

}  // end namespace tensorflow
