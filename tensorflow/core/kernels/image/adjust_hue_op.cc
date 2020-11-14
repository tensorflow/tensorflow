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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "tensorflow/core/kernels/image/adjust_hue_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/work_sharder.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

class AdjustHueOpBase : public OpKernel {
 protected:
  explicit AdjustHueOpBase(OpKernelConstruction* context) : OpKernel(context) {}

  struct ComputeOptions {
    const Tensor* input;
    const Tensor* delta;
    Tensor* output;
    int64 channel_count;
  };

  virtual void DoCompute(OpKernelContext* context,
                         const ComputeOptions& options) = 0;

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& delta = context->input(1);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(delta.shape()),
                errors::InvalidArgument("delta must be scalar: ",
                                        delta.shape().DebugString()));
    auto channels = input.dim_size(input.dims() - 1);
    OP_REQUIRES(
        context, channels == 3,
        errors::InvalidArgument("input must have 3 channels but instead has ",
                                channels, " channels."));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, input.shape(), &output));

    if (input.NumElements() > 0) {
      const int64 channel_count = input.NumElements() / channels;
      ComputeOptions options;
      options.input = &input;
      options.delta = &delta;
      options.output = output;
      options.channel_count = channel_count;
      DoCompute(context, options);
    }
  }
};

template <class Device, typename T>
class AdjustHueOp;

namespace internal {

// Helper function to convert a RGB color to H-and-V-range. H is in the range
// of [0, 6] instead of the normal [0, 1]
static void rgb_to_hv_range(float r, float g, float b, float* h, float* v_min,
                            float* v_max) {
  float v_mid;
  int h_category;
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  // For the conditions, we don't care about the case where two components are
  // equal. It is okay to count it in either side in that case.
  if (r < g) {
    if (b < r) {
      // b < r < g
      *v_max = g;
      v_mid = r;
      *v_min = b;
      h_category = 1;
    } else if (b > g) {
      // r < g < b
      *v_max = b;
      v_mid = g;
      *v_min = r;
      h_category = 3;
    } else {
      // r < b < g
      *v_max = g;
      v_mid = b;
      *v_min = r;
      h_category = 2;
    }
  } else {
    // g < r
    if (b < g) {
      // b < g < r
      *v_max = r;
      v_mid = g;
      *v_min = b;
      h_category = 0;
    } else if (b > r) {
      // g < r < b
      *v_max = b;
      v_mid = r;
      *v_min = g;
      h_category = 4;
    } else {
      // g < b < r
      *v_max = r;
      v_mid = b;
      *v_min = g;
      h_category = 5;
    }
  }
  if (*v_max == *v_min) {
    *h = 0;
    return;
  }
  auto ratio = (v_mid - *v_min) / (*v_max - *v_min);
  bool increase = ((h_category & 0x1) == 0);
  *h = h_category + (increase ? ratio : (1 - ratio));
}

// Helper function to convert from H-and-V-range to RGB.
static void hv_range_to_rgb(float h, float v_min, float v_max, float* r,
                            float* g, float* b) {
  int h_category = static_cast<int>(h);
  float ratio = h - h_category;
  bool increase = ((h_category & 0x1) == 0);
  if (!increase) {
    ratio = 1 - ratio;
  }
  float v_mid = v_min + ratio * (v_max - v_min);
  // According to the figures in:
  // https://en.wikipedia.org/wiki/HSL_and_HSV#Hue_and_chroma
  switch (h_category) {
    case 0:
      *r = v_max;
      *g = v_mid;
      *b = v_min;
      break;
    case 1:
      *r = v_mid;
      *g = v_max;
      *b = v_min;
      break;
    case 2:
      *r = v_min;
      *g = v_max;
      *b = v_mid;
      break;
    case 3:
      *r = v_min;
      *g = v_mid;
      *b = v_max;
      break;
    case 4:
      *r = v_mid;
      *g = v_min;
      *b = v_max;
      break;
    case 5:
    default:
      *r = v_max;
      *g = v_min;
      *b = v_mid;
  }
}
}  // namespace internal

template <>
class AdjustHueOp<CPUDevice, float> : public AdjustHueOpBase {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context)
      : AdjustHueOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    const Tensor* delta = options.delta;
    Tensor* output = options.output;
    const int64 channel_count = options.channel_count;
    static const int kChannelSize = 3;
    auto input_data = input->shaped<float, 2>({channel_count, kChannelSize});
    const float delta_h = delta->scalar<float>()();
    auto output_data = output->shaped<float, 2>({channel_count, kChannelSize});
    const int kCostPerChannel = 10;
    const DeviceBase::CpuWorkerThreads& worker_threads =
        *context->device()->tensorflow_cpu_worker_threads();
    Shard(worker_threads.num_threads, worker_threads.workers, channel_count,
          kCostPerChannel,
          [&input_data, &output_data, delta_h](int64 start_channel,
                                               int64 end_channel) {
            const float* p = input_data.data() + start_channel * kChannelSize;
            float* q = output_data.data() + start_channel * kChannelSize;
            for (int i = start_channel; i < end_channel; i++) {
              float h, v_min, v_max;
              // Convert the RGB color to Hue/V-range.
              internal::rgb_to_hv_range(p[0], p[1], p[2], &h, &v_min, &v_max);
              static const int kChannelRange = 6;
              // Adjust the hue value. And adjust the hue back into the valid
              // range of [0, 6). It is faster than a fmod by avoiding
              // a float-point division since h is often very close to this
              // range.
              h += delta_h * kChannelRange;
              while (h < 0) {
                h += kChannelRange;
              }
              while (h >= kChannelRange) {
                h -= kChannelRange;
              }
              // Convert the hue and v-range back into RGB.
              internal::hv_range_to_rgb(h, v_min, v_max, q, q + 1, q + 2);
              p += kChannelSize;
              q += kChannelSize;
            }
          });
  }
};

REGISTER_KERNEL_BUILDER(
    Name("AdjustHue").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    AdjustHueOp<CPUDevice, float>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T>
class AdjustHueOp<GPUDevice, T> : public AdjustHueOpBase {
 public:
  explicit AdjustHueOp(OpKernelConstruction* context)
      : AdjustHueOpBase(context) {}

  void DoCompute(OpKernelContext* context,
                 const ComputeOptions& options) override {
    const Tensor* input = options.input;
    const Tensor* delta = options.delta;
    Tensor* output = options.output;
    const int64 number_of_elements = input->NumElements();
    GPUDevice device = context->eigen_gpu_device();
    const auto stream = device.stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));
    if (number_of_elements > 0) {
      const T* input_data = input->flat<T>().data();
      const float* delta_h = delta->flat<float>().data();
      T* const output_data = output->flat<T>().data();
      functor::AdjustHueGPU<T>()(&device, number_of_elements, input_data,
                                 delta_h, output_data);
    }
  }
};

#define REGISTER_GPU(T)                                            \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AdjustHue").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustHueOp<GPUDevice, T>);

REGISTER_GPU(float)
REGISTER_GPU(Eigen::half)

#undef REGISTER_GPU

#endif

//} // namespace functor
}  // namespace tensorflow
