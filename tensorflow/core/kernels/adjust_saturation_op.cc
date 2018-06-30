/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/kernels/adjust_saturation_op.h"
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

class AdjustSaturationOpBase : public OpKernel {
 protected:
  explicit AdjustSaturationOpBase(OpKernelConstruction* context)
      : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* scale;
    Tensor* output;
    int64 channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& scale = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale.shape()),
                errors::InvalidArgument("scale must be scalar: ",
                                        scale.shape().DebugString()));
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
      options.scale = &scale;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device>
class AdjustSaturationOp;

namespace internal {
static void rgb_to_hsv(float r, float g, float b, float* h, float* s,
                       float* v) {
  float vv = std::max(r, std::max(g, b));
  float range = vv - std::min(r, std::min(g, b));
  if (vv > 0) {
    *s = range / vv;
  } else {
    *s = 0;
  }
  float norm = 1.0f / (6.0f * range);
  float hh;
  if (r == vv) {
    hh = norm * (g - b);
  } else if (g == vv) {
    hh = norm * (b - r) + 2.0 / 6.0;
  } else {
    hh = norm * (r - g) + 4.0 / 6.0;
  }
  if (range <= 0.0) {
    hh = 0;
  }
  if (hh < 0.0) {
    hh = hh + 1;
  }
  *v = vv;
  *h = hh;
}

// Algorithm from wikipedia, https://en.wikipedia.org/wiki/HSL_and_HSV#From_HSV
static void hsv_to_rgb(float h, float s, float v, float* r, float* g,
                       float* b) {
  float c = s * v;
  float m = v - c;
  float dh = h * 6;
  float rr, gg, bb;
  int h_category = static_cast<int>(dh);
  float fmodu = dh;
  while (fmodu <= 0) {
    fmodu += 2.0f;
  }
  while (fmodu >= 2.0f) {
    fmodu -= 2.0f;
  }
  float x = c * (1 - std::abs(fmodu - 1));
  switch (h_category) {
    case 0:
      rr = c;
      gg = x;
      bb = 0;
      break;
    case 1:
      rr = x;
      gg = c;
      bb = 0;
      break;
    case 2:
      rr = 0;
      gg = c;
      bb = x;
      break;
    case 3:
      rr = 0;
      gg = x;
      bb = c;
      break;
    case 4:
      rr = x;
      gg = 0;
      bb = c;
      break;
    case 5:
      rr = c;
      gg = 0;
      bb = x;
      break;
    default:
      rr = 0;
      gg = 0;
      bb = 0;
  }
  *r = rr + m;
  *g = gg + m;
  *b = bb + m;
}

}  // namespace internal

template <>
class AdjustSaturationOp<CPUDevice> : public AdjustSaturationOpBase {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : AdjustSaturationOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    const Tensor* scale = options.scale;
    Tensor* output = options.output;
    const int64 channel_count = options.channel_count;
    static const int kChannelSize = 3;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float scale_h = scale->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [channel_count, &input_data, &output_data, scale_h](
              int64 start_channel, int64 end_channel) {
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              float h, s, v;
              // Convert the RGB color to Hue/V-range.
              internal::rgb_to_hsv(p[0], p[1], p[2], &h, &s, &v);
              s = std::min(1.0f, std::max(0.0f, s * scale_h));
              // Convert the hue and v-range back into RGB.
              internal::hsv_to_rgb(h, s, v, q, q + 1, q + 2);
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(Name("AdjustSaturation").Device(DEVICE_CPU),
                        AdjustSaturationOp<CPUDevice>);

#if GOOGLE_CUDA
template <>
class AdjustSaturationOp<GPUDevice> : public AdjustSaturationOpBase {
 public:
  explicit AdjustSaturationOp(OpKernelConstruction* context)
      : AdjustSaturationOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    const Tensor* scale = options.scale;
    Tensor* output = options.output;
    const int64 number_of_elements = input->NumElements();
    GPUDevice device = context->eigen_gpu_device();
    const auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    if (number_of_elements > 0) {
      const float* input_data = input->flat<float>().data();
      const float* scale_data = scale->flat<float>().data();
      float* const output_data = output->flat<float>().data();
      functor::AdjustSaturationGPU()(&device, number_of_elements, input_data,
                                     scale_data, output_data);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("AdjustSaturation").Device(DEVICE_GPU),
                        AdjustSaturationOp<GPUDevice>);

#endif

}  // namespace tensorflow
