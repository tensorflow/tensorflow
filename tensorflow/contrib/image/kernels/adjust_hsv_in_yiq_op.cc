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
#include <cmath>
#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
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
    const Tensor* delta_h = nullptr;
    const Tensor* scale_s = nullptr;
    const Tensor* scale_v = nullptr;
    Tensor* output = nullptr;
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
        context, channels == 3,
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
    static const int kChannelSize = 3;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float delta_h = options.delta_h->scalar<float>()();
    const float scale_s = options.scale_s->scalar<float>()();
    const float scale_v = options.scale_v->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [channel_count, &input_data, &output_data, delta_h, scale_s, scale_v](
              int64 start_channel, int64 end_channel) {
            // Using approximate linear transfomation described in:
            // https://beesbuzz.biz/code/hsv_color_transforms.php
            /** Get the constants from sympy
             from sympy import Matrix
             from sympy.abc import u, w
             # Projection matrix to YIQ. http://en.wikipedia.org/wiki/YIQ
             tyiq = Matrix([[0.299, 0.587, 0.114],
                            [0.596, -0.274, -0.322],
                            [0.211, -0.523, 0.312]])
             # Hue rotation matrix in YIQ space.
             hue_proj = Matrix(3,3, [v, 0, 0, 0, vsu, -vsw, 0, vsw, vsu])
             m = tyiq.inv() * hue_proj * tyiq
             **/
            // TODO(huangyp): directly compute the projection matrix from tyiq.
            static const float t[kChannelSize][kChannelSize][kChannelSize] = {
                {{.299, .701, .16862179492229},
                 {.587, -.587, .329804745287403},
                 {.114, -.114, -0.498426540209694}},
                {{.299, -.299, -.327963394172371},
                 {.587, .413, .0346106879248821},
                 {.114, -.114, .293352706247489}},
                {{.299, -.299, 1.24646136576682},
                 {.587, -.587, -1.04322888291964},
                 {.114, .886, -.203232482847173}}};
            float m[kChannelSize][kChannelSize] = {{0.}};
            float su = scale_s * std::cos(delta_h);
            float sw = scale_s * std::sin(delta_h);
            for (int q_index = 0; q_index < kChannelSize; q_index++) {
              for (int p_index = 0; p_index < kChannelSize; p_index++) {
                m[q_index][p_index] = scale_v * (t[q_index][p_index][0] +
                                                 t[q_index][p_index][1] * su +
                                                 t[q_index][p_index][2] * sw);
              }
            }
            // Applying projection matrix to input RGB vectors.
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              for (int q_index = 0; q_index < kChannelSize; q_index++) {
                q[q_index] = 0;
                for (int p_index = 0; p_index < kChannelSize; p_index++) {
                  q[q_index] += m[q_index][p_index] * p[p_index];
                }
              }
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(Name("AdjustHsvInYiq").Device(DEVICE_CPU),
                        AdjustHsvInYiqOp<CPUDevice>);

// TODO(huangyp): add the GPU kernel
}  // namespace tensorflow
