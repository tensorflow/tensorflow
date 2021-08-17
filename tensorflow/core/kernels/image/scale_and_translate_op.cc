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

// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/image/scale_and_translate_op.h"

#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/image/sampling_kernels.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
using strings::Printf;
typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
namespace {
template <typename T>
inline const T& Clamp(const T& low, const T& high, const T& value) {
  if (high < value) return high;
  if (value < low) return low;
  return value;
}

template <typename Kernel>
Status ComputeSpansCore(OpKernelContext* context, const Kernel& kernel,
                        const int64_t output_size, const int64_t input_size,
                        const float scale, const float translate,
                        const bool antialias, Spans* spans) {
  // When sampling, we need the inverse scale and translation, to map from an
  // output to an input pixel.
  const float inv_scale = 1.0 / scale;
  const float inv_translate = -inv_scale * translate;
  // When downsampling the kernel should be scaled since we want to low pass
  // filter and interpolate, but when upsampling it should not be since we only
  // want to interpolate.
  const float kernel_scale = antialias ? std::max(inv_scale, 1.0f) : 1.0f;
  spans->span_size = std::min(
      2 * static_cast<int>(std::ceil(kernel.Radius() * kernel_scale)) + 1,
      static_cast<int>(input_size));
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DT_INT32, tensorflow::TensorShape({output_size}),
      &spans->starts, alloc_attr));
  auto starts_vec = spans->starts.vec<int32>();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({spans->span_size * output_size}),
      &spans->weights, alloc_attr));
  auto weights_vec = spans->weights.vec<float>();
  weights_vec.setZero();

  const float one_over_kernel_scale = 1.0f / kernel_scale;
  int max_span_size = 0;
  std::vector<float> temp_weights;
  for (int x = 0; x < output_size; ++x) {
    const float col_f = x + 0.5f;
    const float sample_f = col_f * inv_scale + inv_translate;

    // Don't sample when the sampling location is outside the source image.
    if (sample_f < 0 || sample_f > input_size) {
      // Add an empty span.
      starts_vec(x) = 0;
      continue;
    }
    int64_t span_start =
        std::ceil(sample_f - kernel.Radius() * kernel_scale - 0.5f);
    int64_t span_end =
        std::floor(sample_f + kernel.Radius() * kernel_scale - 0.5f);
    span_start = Clamp(static_cast<int64_t>(0), input_size - 1, span_start);
    span_end = Clamp(static_cast<int64_t>(0), input_size - 1, span_end) + 1;
    const int this_span_size = span_end - span_start;
    if (this_span_size > spans->span_size) {
      return errors::Internal(Printf("Span is too large: %d vs %d.",
                                     this_span_size, spans->span_size));
    }
    float total_weight_sum = 0.0f;
    temp_weights.clear();
    for (int source = span_start; source < span_end; ++source) {
      float kernel_pos = static_cast<float>(source) + 0.5f - sample_f;
      float weight = kernel(std::abs(kernel_pos * one_over_kernel_scale));
      total_weight_sum += weight;
      temp_weights.push_back(weight);
    }
    max_span_size = std::max(max_span_size, this_span_size);
    if (std::abs(total_weight_sum) >=
        1000.0f * std::numeric_limits<float>::min()) {
      float one_over_total_weight_sum = 1.0f / total_weight_sum;
      int out_index = spans->span_size * x;
      for (float weight : temp_weights) {
        weights_vec(out_index) = weight * one_over_total_weight_sum;
        ++out_index;
      }
    }
    starts_vec(x) = span_start;
  }
  return Status::OK();
}

Status ComputeGradSpansCore(OpKernelContext* context, const Spans& spans,
                            const int64_t forward_output_size,
                            const int64_t forward_input_size,
                            Spans* grad_spans) {
  struct GradComponent {
    int index;
    float weight;
  };
  std::vector<std::vector<GradComponent>> grad_components(forward_input_size);
  auto weights_vec = spans.weights.vec<float>();
  auto starts_vec = spans.starts.vec<int32>();
  for (int output_index = 0; output_index < forward_output_size;
       ++output_index) {
    int input_index = starts_vec(output_index);
    for (int j = 0; j < spans.span_size; ++j, ++input_index) {
      const float weight = weights_vec(output_index * spans.span_size + j);
      if (weight != 0.0f && input_index < forward_input_size) {
        grad_components[input_index].push_back(
            GradComponent{output_index, weight});
      }
    }
  }
  int max_size = 0;
  for (std::vector<GradComponent>& gc : grad_components) {
    if (!gc.empty()) {
      std::sort(gc.begin(), gc.end(),
                [](const GradComponent& x1, const GradComponent& x2) {
                  return x1.index < x2.index;
                });
      max_size = std::max(gc.back().index - gc.front().index + 1, max_size);
    }
  }
  grad_spans->span_size = max_size;
  AllocatorAttributes alloc_attr;
  alloc_attr.set_on_host(true);
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DT_INT32, tensorflow::TensorShape({forward_input_size}),
      &grad_spans->starts, alloc_attr));
  auto grad_starts_vec = grad_spans->starts.vec<int32>();
  TF_RETURN_IF_ERROR(context->allocate_temp(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({grad_spans->span_size * forward_input_size}),
      &grad_spans->weights, alloc_attr));
  auto grad_weights_vec = grad_spans->weights.vec<float>();
  grad_weights_vec.setZero();
  for (int input_index = 0; input_index < forward_input_size; ++input_index) {
    if (!grad_components[input_index].empty()) {
      const int start_span = grad_components[input_index].front().index;
      grad_starts_vec(input_index) = start_span;
      for (const GradComponent& gc : grad_components[input_index]) {
        grad_weights_vec(input_index * grad_spans->span_size + gc.index -
                         start_span) += gc.weight;
      }
    } else {
      grad_starts_vec(input_index) = 0;
    }
  }
  return Status::OK();
}

// Computes the spans for the passed kernel, for a input dimension of length
// input_size transformed by scale and translate to an output dimension of
// length output_size. Note that there's no requirement that;
// output_size = input_size * scale.
Status ComputeSpans(OpKernelContext* context,
                    const functor::SamplingKernelType kernel_type,
                    const int64_t output_size, const int64_t input_size,
                    const float scale, const float translate,
                    const bool antialias, Spans* spans) {
  switch (kernel_type) {
    case functor::Lanczos1Kernel: {
      return ComputeSpansCore(context, CreateLanczos1Kernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::Lanczos3Kernel: {
      return ComputeSpansCore(context, CreateLanczos3Kernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::Lanczos5Kernel: {
      return ComputeSpansCore(context, CreateLanczos5Kernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::GaussianKernel: {
      return ComputeSpansCore(context, CreateGaussianKernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::BoxKernel: {
      return ComputeSpansCore(context, CreateBoxKernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::TriangleKernel: {
      return ComputeSpansCore(context, CreateTriangleKernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::KeysCubicKernel: {
      return ComputeSpansCore(context, CreateKeysCubicKernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    case functor::MitchellCubicKernel: {
      return ComputeSpansCore(context, CreateMitchellCubicKernel(), output_size,
                              input_size, scale, translate, antialias, spans);
    }
    default:
      return errors::InvalidArgument(Printf("Unrecognized kernel type: %d",
                                            static_cast<int>(kernel_type)));
  }
  return Status::OK();
}

// Computes the grad spans for the passed kernel.
// forward_input_size and forward_output_size are the input and output size from
// the forward operation.
Status ComputeGradSpans(OpKernelContext* context,
                        const functor::SamplingKernelType kernel_type,
                        const int64_t forward_output_size,
                        const int64_t forward_input_size, const float scale,
                        const float translate, const bool antialias,
                        Spans* grad_spans) {
  Spans spans;
  TF_RETURN_IF_ERROR(ComputeSpans(context, kernel_type, forward_output_size,
                                  forward_input_size, scale, translate,
                                  antialias, &spans));
  return ComputeGradSpansCore(context, spans, forward_output_size,
                              forward_input_size, grad_spans);
}

void GetValues(OpKernelContext* context, int input_index, float* v_1,
               float* v_2) {
  // Tensor mutable_input(int index, False);
  const Tensor& t = context->input(input_index);
  OP_REQUIRES(context, t.dims() == 1,
              errors::InvalidArgument("t must be 1-dimensional",
                                      t.shape().DebugString()));
  OP_REQUIRES(context, t.NumElements() == 2,
              errors::InvalidArgument("t must have two elements",
                                      t.shape().DebugString()));

  auto data_vec = t.flat<float>().data();
  *v_1 = data_vec[0];
  *v_2 = data_vec[1];
}

template <typename Device, typename T>
class ScaleAndTranslateOp : public OpKernel {
 public:
  explicit ScaleAndTranslateOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias_));
    string kernel_type_str;
    OP_REQUIRES_OK(context, context->GetAttr("kernel_type", &kernel_type_str));
    kernel_type_ = functor::SamplingKernelTypeFromString(kernel_type_str);
    OP_REQUIRES(context, kernel_type_ != functor::SamplingKernelTypeEnd,
                errors::InvalidArgument("Unrecognized kernel type: " +
                                        kernel_type_str));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    const Tensor& output_shape_t = context->input(1);
    OP_REQUIRES(context, output_shape_t.dims() == 1,
                errors::InvalidArgument("output_shape_t must be 1-dimensional",
                                        output_shape_t.shape().DebugString()));
    OP_REQUIRES(context, output_shape_t.NumElements() == 2,
                errors::InvalidArgument("output_shape_t must have two elements",
                                        output_shape_t.shape().DebugString()));
    auto output_shape_vec = output_shape_t.vec<int32>();
    const int64_t output_height = internal::SubtleMustCopy(output_shape_vec(0));
    const int64_t output_width = internal::SubtleMustCopy(output_shape_vec(1));

    OP_REQUIRES(
        context,
        FastBoundsCheck(input.dim_size(1), std::numeric_limits<int32>::max()) &&
            FastBoundsCheck(input.dim_size(2),
                            std::numeric_limits<int32>::max()),
        errors::InvalidArgument("input sizes must be between 0 and max int32"));

    const int64_t batch_size = input.dim_size(0);
    const int64_t input_height = input.dim_size(1);
    const int64_t input_width = input.dim_size(2);
    const int64_t channels = input.dim_size(3);
    OP_REQUIRES(context, output_height > 0 && output_width > 0,
                errors::InvalidArgument("output dimensions must be positive"));
    OP_REQUIRES(
        context, channels > 0,
        errors::InvalidArgument("image must have at least one channel"));
    OP_REQUIRES(
        context, input.dim_size(1) > 0 && input.dim_size(2) > 0,
        errors::InvalidArgument("input image must be of non-zero size"));

    float row_scale, col_scale;
    GetValues(context, 2, &row_scale, &col_scale);
    OP_REQUIRES(context, row_scale > 0 && col_scale > 0,
                errors::InvalidArgument("Scale must be greater than zero."));
    float row_translation, col_translation;
    GetValues(context, 3, &row_translation, &col_translation);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({input.dim_size(0), output_height,
                                             output_width, input.dim_size(3)}),
                                &output));
    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data(input.tensor<T, 4>());
    TTypes<float, 4>::Tensor output_data = output->tensor<float, 4>();

    functor::Spans col_spans;
    OP_REQUIRES_OK(
        context,
        ComputeSpans(context, kernel_type_, output_width, input_width,
                     col_scale, col_translation, antialias_, &col_spans));
    functor::Spans row_spans;
    OP_REQUIRES_OK(
        context,
        ComputeSpans(context, kernel_type_, output_height, input_height,
                     row_scale, row_translation, antialias_, &row_spans));
    Tensor intermediate_t;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DT_FLOAT,
                                        TensorShape({batch_size, output_height,
                                                     input_width, channels}),
                                        &intermediate_t));
    TTypes<float, 4>::Tensor intermediate_data =
        intermediate_t.tensor<float, 4>();

    const functor::Spans& const_row_spans = row_spans;
    typename TTypes<int32, 1>::ConstTensor row_starts(
        const_row_spans.starts.tensor<int32, 1>());
    typename TTypes<float, 1>::ConstTensor row_weights(
        const_row_spans.weights.tensor<float, 1>());
    const functor::Spans& const_col_spans = col_spans;
    typename TTypes<int32, 1>::ConstTensor col_starts(
        const_col_spans.starts.tensor<int32, 1>());
    typename TTypes<float, 1>::ConstTensor col_weights(
        const_col_spans.weights.tensor<float, 1>());

    functor::GatherSpans<Device, T>()(
        context->eigen_device<Device>(), row_spans.span_size, row_starts,
        row_weights, col_spans.span_size, col_starts, col_weights, image_data,
        intermediate_data, output_data);
  }
  functor::SamplingKernelType kernel_type_;
  bool antialias_;
};

template <typename Device, typename T>
class ScaleAndTranslateGradOp : public OpKernel {
 public:
  explicit ScaleAndTranslateGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("antialias", &antialias_));
    string kernel_type_str;
    OP_REQUIRES_OK(context, context->GetAttr("kernel_type", &kernel_type_str));
    kernel_type_ = functor::SamplingKernelTypeFromString(kernel_type_str);
    OP_REQUIRES(context, kernel_type_ != functor::SamplingKernelTypeEnd,
                errors::InvalidArgument("Unrecognized kernel type: " +
                                        kernel_type_str));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& original_image = context->input(1);

    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input_grad must be 4-dimensional",
                                        input.shape().DebugString()));
    // Resizers always produce float images, so input gradient must
    // always be a float.
    OP_REQUIRES(context, input.dtype() == DT_FLOAT,
                errors::InvalidArgument("input_grad must be of type float",
                                        DataTypeString(input.dtype())));

    OP_REQUIRES(context, original_image.dims() == 4,
                errors::InvalidArgument("original_image must be 4-dimensional",
                                        original_image.shape().DebugString()));

    // Allocate output and initialize to zeros.
    const int64_t batch_size = input.dim_size(0);
    const int64_t channels = input.dim_size(3);
    const int64_t forward_input_height = original_image.dim_size(1);
    const int64_t forward_input_width = original_image.dim_size(2);

    OP_REQUIRES(context,
                FastBoundsCheck(forward_input_height,
                                std::numeric_limits<int32>::max()) &&
                    FastBoundsCheck(forward_input_width,
                                    std::numeric_limits<int32>::max()),
                errors::InvalidArgument(
                    "original sizes must be between 0 and max int32"));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0,
                                TensorShape({batch_size, forward_input_height,
                                             forward_input_width, channels}),
                                &output));

    float row_scale, col_scale;
    GetValues(context, 2, &row_scale, &col_scale);
    OP_REQUIRES(context, row_scale > 0 && col_scale > 0,
                errors::InvalidArgument("Scale must be greater than zero."));
    float row_translation, col_translation;
    GetValues(context, 3, &row_translation, &col_translation);

    if (!context->status().ok()) return;

    TTypes<float, 4>::ConstTensor input_grad = input.tensor<float, 4>();
    typename TTypes<T, 4>::Tensor output_grad(output->tensor<T, 4>());

    const int64_t forward_output_height = input_grad.dimension(1);
    const int64_t forward_output_width = input_grad.dimension(2);

    functor::Spans col_spans;
    OP_REQUIRES_OK(context,
                   ComputeGradSpans(context, kernel_type_, forward_output_width,
                                    forward_input_width, col_scale,
                                    col_translation, antialias_, &col_spans));
    functor::Spans row_spans;
    OP_REQUIRES_OK(
        context, ComputeGradSpans(context, kernel_type_, forward_output_height,
                                  forward_input_height, row_scale,
                                  row_translation, antialias_, &row_spans));
    Tensor intermediate_t;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DT_FLOAT,
                                TensorShape({batch_size, forward_input_height,
                                             forward_output_width, channels}),
                                &intermediate_t));
    TTypes<float, 4>::Tensor intermediate_data =
        intermediate_t.tensor<float, 4>();

    const functor::Spans& const_row_spans = row_spans;
    typename TTypes<int32, 1>::ConstTensor row_starts =
        const_row_spans.starts.tensor<int32, 1>();
    typename TTypes<float, 1>::ConstTensor row_weights(
        const_row_spans.weights.tensor<float, 1>());
    const functor::Spans& const_col_spans = col_spans;
    typename TTypes<int32, 1>::ConstTensor col_starts(
        const_col_spans.starts.tensor<int32, 1>());
    typename TTypes<float, 1>::ConstTensor col_weights(
        const_col_spans.weights.tensor<float, 1>());

    functor::GatherSpans<Device, T>()(
        context->eigen_device<Device>(), row_spans.span_size, row_starts,
        row_weights, col_spans.span_size, col_starts, col_weights, input_grad,
        intermediate_data, output_grad);
  }

  functor::SamplingKernelType kernel_type_;
  bool antialias_;
};

template <typename T>
void GatherColumns(int span_size, const int32* starts, const float* weights,
                   const T* image, const int64_t input_height,
                   const int64_t input_width, const int64_t output_height,
                   const int64_t output_width, const int channels,
                   float* output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;

  for (int y = 0; y < output_height; ++y) {
    const T* input_row_start = image + in_row_size * y;
    float* out_pix = output + out_row_size * y;
    for (int x = 0; x < output_width; ++x, out_pix += channels) {
      const T* in_pix = input_row_start + starts[x] * channels;
      const float* weights_start = weights + x * span_size;
      const int real_span_size =
          std::min(starts[x] + span_size, static_cast<int>(input_width)) -
          starts[x];
      const float* weights_end = weights_start + real_span_size;
      for (int c = 0; c < channels; ++c) {
        out_pix[c] = 0.0f;
      }
      for (const float* weight_ptr = weights_start; weight_ptr != weights_end;
           ++weight_ptr) {
        float w = *weight_ptr;
        for (int c = 0; c < channels; ++c) {
          out_pix[c] += w * static_cast<float>(in_pix[c]);
        }
        in_pix += channels;
      }
    }
  }
}

template <typename T>
inline void AddScaledVector(const T* in_vec, int vec_len, float weight,
                            float* out_vec) {
  float* out_vec_end = out_vec + vec_len;
  for (; out_vec != out_vec_end; ++out_vec, ++in_vec) {
    *out_vec += weight * static_cast<float>(*in_vec);
  }
}

template <typename T>
void GatherRows(int span_size, const int32* starts, const float* weights,
                const T* image, const int64_t input_height,
                const int64_t input_width, const int64_t output_height,
                const int64_t output_width, const int channels, float* output) {
  const int64_t in_row_size = input_width * channels;
  const int64_t out_row_size = output_width * channels;

  for (int y = 0; y < output_height; ++y) {
    float* out_row_data = output + out_row_size * y;
    std::fill(out_row_data, out_row_data + out_row_size, 0.0f);
    int in_row = starts[y];
    const T* in_row_data = image + in_row_size * in_row;
    const float* weights_start = weights + y * span_size;
    const int real_span_size =
        std::min(starts[y] + span_size, static_cast<int>(input_height)) -
        starts[y];
    const float* const weights_end = weights_start + real_span_size;
    for (const float* weight_it = weights_start; weight_it != weights_end;
         ++weight_it) {
      AddScaledVector(in_row_data, in_row_size, *weight_it, out_row_data);
      in_row_data += in_row_size;
    }
  }
}

}  // namespace

// Partial specialization of GatherSpans functor for a CPUDevice.
template <typename T>
struct GatherSpans<CPUDevice, T> {
  void operator()(const CPUDevice& d, int row_span_size,
                  typename TTypes<int32, 1>::ConstTensor row_starts,
                  typename TTypes<float, 1>::ConstTensor row_weights,
                  int col_span_size,
                  typename TTypes<int32, 1>::ConstTensor col_starts,
                  typename TTypes<float, 1>::ConstTensor col_weights,
                  typename TTypes<T, 4>::ConstTensor images,
                  typename TTypes<float, 4>::Tensor intermediate_buffer,
                  typename TTypes<float, 4>::Tensor resized_images) {
    const int batch_size = images.dimension(0);
    const int64_t input_height = images.dimension(1);
    const int64_t input_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int64_t output_height = resized_images.dimension(1);
    const int64_t output_width = resized_images.dimension(2);

    const int64_t input_pix_per_batch = input_width * input_height * channels;
    const int64_t intermediate_pix_per_batch =
        input_width * output_height * channels;
    const int64_t output_pix_per_batch =
        output_width * output_height * channels;
    float* intermediate_ptr = intermediate_buffer.data();

    const T* image_ptr = images.data();
    float* out_ptr = resized_images.data();
    for (int b = 0; b < batch_size; ++b, image_ptr += input_pix_per_batch,
             intermediate_ptr += intermediate_pix_per_batch,
             out_ptr += output_pix_per_batch) {
      GatherRows(row_span_size, row_starts.data(), row_weights.data(),
                 image_ptr, input_height, input_width, output_height,
                 input_width, channels, intermediate_ptr);
      GatherColumns(col_span_size, col_starts.data(), col_weights.data(),
                    intermediate_ptr, output_height, input_width, output_height,
                    output_width, channels, out_ptr);
    }
  }
};

#define REGISTER_KERNEL(T)                                \
  REGISTER_KERNEL_BUILDER(Name("ScaleAndTranslate")       \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("size")         \
                              .HostMemory("scale")        \
                              .HostMemory("translation"), \
                          ScaleAndTranslateOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                           \
  REGISTER_KERNEL_BUILDER(Name("ScaleAndTranslateGrad")   \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<T>("T")     \
                              .HostMemory("scale")        \
                              .HostMemory("translation"), \
                          ScaleAndTranslateGradOp<CPUDevice, T>);

TF_CALL_float(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

}  // namespace functor
}  // namespace tensorflow
