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

#if TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/util/guarded_philox_random.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/stream_executor/temporary_device_memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class DropoutOp : public OpKernel {
 private:
  GuardedPhiloxRandom generator_;

 public:
  explicit DropoutOp(OpKernelConstruction* context) : OpKernel(context) {
    generator_.Init(0, 0);
  }

  ~DropoutOp() override {}

  void Compute(OpKernelContext* ctx) override {
    auto* stream = ctx->op_device_context()->stream();

    const Tensor& in0 = ctx->input(0);

    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    auto rate_src_ptr = AsDeviceMemory<T>(&in1.scalar<T>()(), sizeof(T));
    T rate;
    stream->ThenMemcpy(&rate, rate_src_ptr, sizeof(T));

    const Tensor& in2 = ctx->input(2);
    auto noise_shape_src_ptr = AsDeviceMemory<int32>(
        in2.flat<int32>().data(), in2.flat<int32>().size() * sizeof(int32));
    std::vector<int32> noise_dim_size(in2.shape().num_elements(), 0);
    stream->ThenMemcpy(noise_dim_size.data(), noise_shape_src_ptr,
                       in2.flat<int32>().size() * sizeof(int32));
    OP_REQUIRES(ctx, in0.dims() == noise_dim_size.size(),
                errors::InvalidArgument("MIOpen only supports input dimensions "
                                        "to match noise dimensions."));

    const Tensor& in3 = ctx->input(3);
    OP_REQUIRES(
        ctx, in3.dims() == 0,
        errors::InvalidArgument("Dropout seed must be a scalar tensor."));
    auto seed_src_ptr =
        AsDeviceMemory<int64>(&in3.scalar<int64>()(), sizeof(int64));
    int64 seed = 0;
    stream->ThenMemcpy(&seed, seed_src_ptr, sizeof(int64));
    generator_.ResetSeeds(seed, 0);

    se::dnn::DropoutDescriptor dropout_desc;
    dropout_desc.set_rate(static_cast<float>(rate));
    dropout_desc.set_seed(seed);

    // Build random uniform distribution
    typedef random::UniformDistribution<random::PhiloxRandom, T> Distribution;
    Distribution dist;

    std::vector<T> random_nums(in0.shape().num_elements());
    functor::FillPhiloxRandom<Eigen::ThreadPoolDevice, Distribution>()(
        ctx, ctx->eigen_device<Eigen::ThreadPoolDevice>(),
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        generator_.ReserveRandomOutputs(random_nums.size() * sizeof(T), 256),
        random_nums.data(), random_nums.size(), dist);

    Eigen::Tensor<T, 1> rate_tensor(random_nums.size());
    rate_tensor.setConstant(rate);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> random_tensor(random_nums.data(),
                                                        random_nums.size());
    Eigen::Tensor<bool, 1> mask_tensor(random_nums.size());
    mask_tensor = random_tensor >= rate_tensor;
    std::vector<uint8> mask = std::vector<uint8>(
        mask_tensor.data(), mask_tensor.data() + mask_tensor.size());
    dropout_desc.set_mask(mask);

    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &output));
    if (output->NumElements() == 0) return;

    // Fill one to higher dimensions
    gtl::InlinedVector<int64, 4> input_dim_sizes = in0.shape().dim_sizes();
    size_t input_size_to_fill = 4 - input_dim_sizes.size();
    for (size_t i = 0; i < input_size_to_fill; ++i) {
      input_dim_sizes.insert(input_dim_sizes.begin(), 1);
    }
    const int64 in_batch = input_dim_sizes[0];
    const int64 in_depths = input_dim_sizes[1];
    const int64 in_rows = input_dim_sizes[2];
    const int64 in_cols = input_dim_sizes[3];

    // Interpret compute data layout to NCHW to be consistent with input tensor
    se::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depths)
        .set_height(in_rows)
        .set_width(in_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    size_t noise_size_to_fill = 4 - noise_dim_size.size();
    for (size_t i = 0; i < noise_size_to_fill; ++i) {
      noise_dim_size.insert(noise_dim_size.begin(), 1);
    }
    const int64 noise_batch = noise_dim_size[0];
    const int64 noise_depth = noise_dim_size[1];
    const int64 noise_rows = noise_dim_size[2];
    const int64 noise_cols = noise_dim_size[3];

    se::dnn::BatchDescriptor noise_desc;
    noise_desc.set_count(noise_batch)
        .set_feature_map_count(noise_depth)
        .set_height(noise_rows)
        .set_width(noise_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    se::dnn::BatchDescriptor output_desc;
    output_desc.CloneFrom(input_desc);

    auto input_data =
        AsDeviceMemory(in0.flat<T>().data(), in0.flat<T>().size());

    auto output_data =
        AsDeviceMemory(output->flat<T>().data(), output->flat<T>().size());

    static int64 DropoutScratchSize = GetDnnWorkspaceLimit(
        // default value is in bytes despite the name of the environment
        // variable
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
    );
    DnnScratchAllocator scratch_allocator(DropoutScratchSize, ctx);

    bool status = stream
                      ->ThenDropoutForward(dropout_desc, noise_desc, input_desc,
                                           input_data, output_desc,
                                           &output_data, &scratch_allocator)
                      .ok();
    OP_REQUIRES(ctx, status,
                errors::Internal("dnn DropoutForward launch failed"));
  }
};

#define REGISTER_DROPOUT_GPU(TYPE)                                  \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("Dropout").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      DropoutOp<GPUDevice, TYPE>);

TF_CALL_float(REGISTER_DROPOUT_GPU);
TF_CALL_half(REGISTER_DROPOUT_GPU);
// TODO Enable when MIOpen supports the following data types
//TF_CALL_double(REGISTER_DROPOUT_GPU);

template <typename Device, typename T>
class DropoutGradOp : public OpKernel {
 private:
  GuardedPhiloxRandom generator_;

 public:
  explicit DropoutGradOp(OpKernelConstruction* context) : OpKernel(context) {
    generator_.Init(0, 0);
  }

  ~DropoutGradOp() override {}

  void Compute(OpKernelContext* ctx) override {
    auto* stream = ctx->op_device_context()->stream();

    const Tensor& in0 = ctx->input(0);

    const Tensor& in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dtype() == in1.dtype(),
                errors::InvalidArgument(
                    "Dropout rate must be same type as input tensor."));
    OP_REQUIRES(
        ctx, in1.dims() == 0,
        errors::InvalidArgument("Dropout rate must be a scalar tensor."));
    auto rate_src_ptr = AsDeviceMemory<T>(&in1.scalar<T>()(), sizeof(T));
    T rate;
    stream->ThenMemcpy(&rate, rate_src_ptr, sizeof(T));

    const Tensor& in2 = ctx->input(2);
    auto noise_shape_src_ptr = AsDeviceMemory<int32>(
        in2.flat<int32>().data(), in2.flat<int32>().size() * sizeof(int32));
    std::vector<int32> noise_dim_size(in2.shape().num_elements(), 0);
    stream->ThenMemcpy(noise_dim_size.data(), noise_shape_src_ptr,
                       in2.flat<int32>().size() * sizeof(int32));
    OP_REQUIRES(ctx, in0.dims() == noise_dim_size.size(),
                errors::InvalidArgument("MIOpen only supports input dimensions "
                                        "to match noise dimensions."));

    const Tensor& in3 = ctx->input(3);
    OP_REQUIRES(
        ctx, in3.dims() == 0,
        errors::InvalidArgument("Dropout seed must be a scalar tensor."));
    auto seed_src_ptr =
        AsDeviceMemory<int64>(&in3.scalar<int64>()(), sizeof(int64));
    int64 seed = 0;
    stream->ThenMemcpy(&seed, seed_src_ptr, sizeof(int64));
    generator_.ResetSeeds(seed, 0);

    se::dnn::DropoutDescriptor dropout_desc;
    dropout_desc.set_rate(static_cast<float>(rate));
    dropout_desc.set_seed(seed);

    // Build random uniform distribution
    typedef random::UniformDistribution<random::PhiloxRandom, T> Distribution;
    Distribution dist;

    std::vector<T> random_nums(in0.shape().num_elements());
    functor::FillPhiloxRandom<Eigen::ThreadPoolDevice, Distribution>()(
        ctx, ctx->eigen_device<Eigen::ThreadPoolDevice>(),
        // Multiplier 256 is the same as in FillPhiloxRandomTask; do not change
        // it just here.
        generator_.ReserveRandomOutputs(random_nums.size() * sizeof(T), 256),
        random_nums.data(), random_nums.size(), dist);

    Eigen::Tensor<T, 1> rate_tensor(random_nums.size());
    rate_tensor.setConstant(rate);
    Eigen::TensorMap<Eigen::Tensor<T, 1>> random_tensor(random_nums.data(),
                                                        random_nums.size());
    Eigen::Tensor<bool, 1> mask_tensor(random_nums.size());
    mask_tensor = random_tensor >= rate_tensor;
    std::vector<uint8> mask = std::vector<uint8>(
        mask_tensor.data(), mask_tensor.data() + mask_tensor.size());
    dropout_desc.set_mask(mask);

    // Allocate output, and exit early if possible
    Tensor* output;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, in0.shape(), &output));
    if (output->NumElements() == 0) return;

    // Fill one to higher dimensions
    gtl::InlinedVector<int64, 4> input_dim_sizes = in0.shape().dim_sizes();
    size_t input_size_to_fill = 4 - input_dim_sizes.size();
    for (size_t i = 0; i < input_size_to_fill; ++i) {
      input_dim_sizes.insert(input_dim_sizes.begin(), 1);
    }
    const int64 in_batch = input_dim_sizes[0];
    const int64 in_depths = input_dim_sizes[1];
    const int64 in_rows = input_dim_sizes[2];
    const int64 in_cols = input_dim_sizes[3];

    // Interpret compute data layout to NCHW to be consistent with input tensor
    se::dnn::BatchDescriptor input_desc;
    input_desc.set_count(in_batch)
        .set_feature_map_count(in_depths)
        .set_height(in_rows)
        .set_width(in_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    size_t noise_size_to_fill = 4 - noise_dim_size.size();
    for (size_t i = 0; i < noise_size_to_fill; ++i) {
      noise_dim_size.insert(noise_dim_size.begin(), 1);
    }
    const int64 noise_batch = noise_dim_size[0];
    const int64 noise_depth = noise_dim_size[1];
    const int64 noise_rows = noise_dim_size[2];
    const int64 noise_cols = noise_dim_size[3];

    se::dnn::BatchDescriptor noise_desc;
    noise_desc.set_count(noise_batch)
        .set_feature_map_count(noise_depth)
        .set_height(noise_rows)
        .set_width(noise_cols)
        .set_layout(se::dnn::DataLayout::kBatchDepthYX);

    se::dnn::BatchDescriptor output_desc;
    output_desc.CloneFrom(input_desc);

    auto input_data =
        AsDeviceMemory(in0.flat<T>().data(), in0.flat<T>().size());

    auto output_data =
        AsDeviceMemory(output->flat<T>().data(), output->flat<T>().size());

    static int64 DropoutScratchSize = GetDnnWorkspaceLimit(
        // default value is in bytes despite the name of the environment
        // variable
        "TF_CUDNN_WORKSPACE_LIMIT_IN_MB", 1LL << 32  // 4GB
    );
    DnnScratchAllocator scratch_allocator(DropoutScratchSize, ctx);

    bool status = stream
                      ->ThenDropoutBackward(dropout_desc, noise_desc,
                                            input_desc, input_data, output_desc,
                                            &output_data, &scratch_allocator)
                      .ok();
    OP_REQUIRES(ctx, status,
                errors::Internal("dnn DropoutBackward launch failed"));
  }
};

#define REGISTER_DROPOUT_GRAD_GPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("DropoutGrad").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      DropoutGradOp<GPUDevice, TYPE>);

TF_CALL_float(REGISTER_DROPOUT_GRAD_GPU);
TF_CALL_half(REGISTER_DROPOUT_GRAD_GPU);
// TODO Enable when MIOpen supports the following data types
//TF_CALL_double(REGISTER_DROPOUT_GRAD_GPU);

}  // namespace tensorflow
#endif  // TENSORFLOW_USE_ROCM
