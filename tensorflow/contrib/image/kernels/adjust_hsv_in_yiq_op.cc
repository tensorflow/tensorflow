/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/contrib/image/kernels/adjust_hsv_in_yiq_op.h"
#include <memory>
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class AdjustHsvInYiqOpBase : public OpKernel {
 protected:
  explicit AdjustHsvInYiqOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input = nullptr;
    Tensor* output = nullptr;
    const Tensor* delta_h = nullptr;
    const Tensor* scale_s = nullptr;
    const Tensor* scale_v = nullptr;
    int64 channel_count = 0;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& delta_h = context->input(1);
    const Tensor& scale_s = context->input(2);
    const Tensor& scale_v = context->input(3);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta_h.shape()),
                errors::InvalidArgument("delta_h must be scalar: ",
                                        delta_h.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_s.shape()),
                errors::InvalidArgument("scale_s must be scalar: ",
                                        scale_s.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_v.shape()),
                errors::InvalidArgument("scale_v must be scalar: ",
                                        scale_v.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        context, channels == kChannelSize,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64 channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.delta_h = &delta_h;
      options.scale_s = &scale_s;
      options.scale_v = &scale_v;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device>
class AdjustHsvInYiqOp;

template <>
class AdjustHsvInYiqOp<CPUDevice> : public AdjustHsvInYiqOpBase {
 public:
  explicit AdjustHsvInYiqOp(OpKernelConstruction* context)
      : AdjustHsvInYiqOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    Tensor* output = options.output;
    const int64 channel_count = options.channel_count;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float delta_h = options.delta_h->scalar<float>()();
    const float scale_s = options.scale_s->scalar<float>()();
    const float scale_v = options.scale_v->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    float tranformation_matrix[kChannelSize * kChannelSize] = {0};
    internal::compute_tranformation_matrix<kChannelSize * kChannelSize>(
        delta_h, scale_s, scale_v, tranformation_matrix);
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [channel_count, &input_data, &output_data, &tranformation_matrix](
              int64 start_channel, int64 end_channel) {
            // Applying projection matrix to input RGB vectors.
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              for (int q_index = 0; q_index < kChannelSize; q_index++) {
                q[q_index] = 0;
                for (int p_index = 0; p_index < kChannelSize; p_index++) {
                  q[q_index] +=
                      p[p_index] *
                      tranformation_matrix[q_index + kChannelSize * p_index];
                }
              }
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustHsvInYiq").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustHsvInYiqOp<CPUDevice>);

#if GOOGLE_CUDA
template <>
class AdjustHsvInYiqOp<GPUDevice> : public AdjustHsvInYiqOpBase {
 public:
  explicit AdjustHsvInYiqOp(OpKernelConstruction* context)
      : AdjustHsvInYiqOpBase(context) {}

  void DoCompute(OpKernelContext* ctx, const ComputeOptions& options) override {
    const int64 number_of_elements = options.input->NumElements();
    if (number_of_elements <= 0) {
      return;
    }
    const float* delta_h = options.delta_h->flat<float>().data();
    const float* scale_s = options.scale_s->flat<float>().data();
    const float* scale_v = options.scale_v->flat<float>().data();
    functor::AdjustHsvInYiqGPU()(ctx, options.channel_count, options.input,
                                 delta_h, scale_s, scale_v, options.output);
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustHsvInYiq").Device(DEVICE_GPU).TypeConstraint<float>("T"),
    AdjustHsvInYiqOp<GPUDevice>);
#endif

}  // namespace tensorflow
