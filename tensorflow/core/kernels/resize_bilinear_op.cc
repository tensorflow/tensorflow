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

#include "tensorflow/core/kernels/resize_bilinear_op.h"

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class ResizeBilinearOp : public OpKernel {
 public:
  explicit ResizeBilinearOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data =
        st.output->tensor<float, 4>();

    functor::ResizeBilinear<Device, T>()(context->eigen_device<Device>(),
                                         image_data, st.height_scale,
                                         st.width_scale, output_data);
  }

 private:
  bool align_corners_;
};

namespace {
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64 lower;  // Lower source index used in the interpolation
  int64 upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
  // How many consecutive points use the same lower & upper indices
  int consecutive;
};

enum ImageScalePattern { SCALE_UP, SIMILAR, SCALE_DOWN };

inline ImageScalePattern compute_image_scale_pattern(const int64 out_height,
                                                     const int64 out_width,
                                                     const int64 in_height,
                                                     const int64 in_width) {
  if (in_height * 2 < out_height || in_width * 2 < out_width) {
    return SCALE_UP;
  } else if (out_height * 2 < in_height || out_width * 2 < in_width) {
    return SCALE_DOWN;
  } else {
    return SIMILAR;
  }
}

inline int compute_scratch_size(const int64 out_height, const int64 out_width,
                                const int64 in_height, const int64 in_width,
                                const int channels,
                                const ImageScalePattern scale_pattern) {
  // Allocate a CachedInterpolation for each y, and each x in the out-height,
  // plus 2 extra to avoid extra branches in the
  // CachedInterpolation.consecutive computation.
  const int cached_computation_size =
      sizeof(CachedInterpolation) * (out_height + out_width + 2);
  if (scale_pattern == SCALE_DOWN) {
    return cached_computation_size;
  } else {
    // In order to avoid paying the cost of data type conversion multiple times,
    // we must allocate a temporary image as well.
    const int tmp_image_size = sizeof(float) * in_height * in_width * channels;
    // We batch up all memory allocations into a single malloc call for
    // performance reasons.
    return cached_computation_size + tmp_image_size;
  }
}

inline void compute_interpolation_weights(const ImageScalePattern scale_pattern,
                                          const int64 out_size,
                                          const int64 in_size,
                                          const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  interpolation[out_size].consecutive = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
    interpolation[i].consecutive =
        interpolation[i + 1].lower == interpolation[i].lower &&
                interpolation[i + 1].upper == interpolation[i].upper
            ? interpolation[i + 1].consecutive + 1
            : 1;
  }
}

template <typename T>
struct Converter {
  static inline const float* convert_image_to_float(
      typename TTypes<T, 4>::ConstTensor images, const int batch_index,
      const int64 in_height, const int64 in_width, const int channels,
      std::vector<float>* converted_image_v) {
    converted_image_v->resize(in_height * in_width * channels);
    float* converted_image = converted_image_v->data();
    for (int64 y = 0; y < in_height; ++y) {
      for (int64 x = 0; x < in_width; ++x) {
        for (int c = 0; c < channels; ++c) {
          converted_image[y * in_width * channels + x * channels + c] =
              static_cast<float>(images(batch_index, y, x, c));
        }
      }
    }
    return converted_image;
  }
};

template <>
struct Converter<float> {
  static inline const float* convert_image_to_float(
      typename TTypes<float, 4>::ConstTensor images, const int b,
      const int64 in_height, const int64 in_width, const int channels,
      std::vector<float>* converted_image_v) {
    return images.data() + (b * in_height * in_width * channels);
  }
};

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template <typename T>
inline void scale_down_image(typename TTypes<T, 4>::ConstTensor images,
                             const int batch_size, const int64 out_height,
                             const int64 out_width, const int channels,
                             const std::vector<CachedInterpolation>& xs,
                             const std::vector<CachedInterpolation>& ys,
                             typename TTypes<float, 4>::Tensor output) {
  // Do not eagerly convert all input data points, as we ignore most.
  for (int b = 0; b < batch_size; ++b) {
    // Compute the interpolation
    for (int64 y = 0; y < out_height; ++y) {
      for (int64 x = 0; x < out_width; ++x) {
        for (int c = 0; c < channels; ++c) {
          const float top_left(images(b, ys[y].lower, xs[x].lower, c));
          const float top_right(images(b, ys[y].lower, xs[x].upper, c));
          const float bottom_left(images(b, ys[y].upper, xs[x].lower, c));
          const float bottom_right(images(b, ys[y].upper, xs[x].upper, c));
          output(b, y, x, c) =
              compute_lerp(top_left, top_right, bottom_left, bottom_right,
                           xs[x].lerp, ys[y].lerp);
        }
      }
    }
  }
}

inline void scale_up_image(const float* input_image, const int batch_index,
                           const int64 out_height, const int64 out_width,
                           const int channels, const int64 in_height,
                           const int64 in_width,
                           const std::vector<CachedInterpolation>& xs,
                           const std::vector<CachedInterpolation>& ys,
                           typename TTypes<float, 4>::Tensor output) {
  for (int64 y = 0; y < out_height; y += ys[y].consecutive) {
    const int64 in_y_lower = ys[y].lower * in_width * channels;
    const int64 in_y_upper = ys[y].upper * in_width * channels;
    for (int64 x = 0; x < out_width; x += xs[x].consecutive) {
      const int64 in_x_lower = xs[x].lower * channels;
      const int64 in_x_upper = xs[x].upper * channels;
      for (int c = 0; c < channels; ++c) {
        const float top_left = input_image[in_y_lower + in_x_lower + c];
        const float top_right = input_image[in_y_lower + in_x_upper + c];
        const float bottom_left = input_image[in_y_upper + in_x_lower + c];
        const float bottom_right = input_image[in_y_upper + in_x_upper + c];
        for (int64 y_inner = y; y_inner < y + ys[y].consecutive; ++y_inner) {
          for (int64 x_inner = x; x_inner < x + xs[x].consecutive; ++x_inner) {
            output(batch_index, y_inner, x_inner, c) =
                compute_lerp(top_left, top_right, bottom_left, bottom_right,
                             xs[x_inner].lerp, ys[y_inner].lerp);
          }
        }
      }
    }
  }
}

inline void scale_similar_image(const float* input_image, const int b,
                                const int64 out_height, const int64 out_width,
                                const int channels, const int64 in_height,
                                const int64 in_width,
                                const std::vector<CachedInterpolation>& xs,
                                const std::vector<CachedInterpolation>& ys,
                                typename TTypes<float, 4>::Tensor output) {
  // Compute the interpolation
  for (int64 y = 0; y < out_height; ++y) {
    const int64 in_y_lower = ys[y].lower * in_width * channels;
    const int64 in_y_upper = ys[y].upper * in_width * channels;
    // Similar-sized images do not have a set of inner loops.
    for (int64 x = 0; x < out_width; ++x) {
      const int64 in_x_lower = xs[x].lower * channels;
      const int64 in_x_upper = xs[x].upper * channels;
      for (int c = 0; c < channels; ++c) {
        const float top_left = input_image[in_y_lower + in_x_lower + c];
        const float top_right = input_image[in_y_lower + in_x_upper + c];
        const float bottom_left = input_image[in_y_upper + in_x_lower + c];
        const float bottom_right = input_image[in_y_upper + in_x_upper + c];
        output(b, y, x, c) = compute_lerp(top_left, top_right, bottom_left,
                                          bottom_right, xs[x].lerp, ys[y].lerp);
      }
    }
  }
}
}  // namespace

// Partial specialization of ResizeBilinear functor for a CPUDevice.
namespace functor {
template <typename T>
struct ResizeBilinear<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                  const float height_scale, const float width_scale,
                  typename TTypes<float, 4>::Tensor output) {
    const int batch_size = images.dimension(0);
    const int64 in_height = images.dimension(1);
    const int64 in_width = images.dimension(2);
    const int channels = images.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    // Handle no-op resizes efficiently.
    if (out_height == in_height && out_width == in_width) {
      output = images.template cast<float>();
      return;
    }

    const ImageScalePattern scale_pattern =
        compute_image_scale_pattern(out_height, out_width, in_height, in_width);
    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);
    std::vector<float> converted_image_v;

    // Compute the cached interpolation weights on the x and y dimensions.
    compute_interpolation_weights(scale_pattern, out_height, in_height,
                                  height_scale, ys.data());
    compute_interpolation_weights(scale_pattern, out_width, in_width,
                                  width_scale, xs.data());

    if (scale_pattern == SCALE_UP) {
      for (int b = 0; b < batch_size; ++b) {
        const float* converted_image = Converter<T>::convert_image_to_float(
            images, b, in_height, in_width, channels, &converted_image_v);
        scale_up_image(converted_image, b, out_height, out_width, channels,
                       in_height, in_width, xs, ys, output);
      }
    } else if (scale_pattern == SCALE_DOWN) {
      // Do not eagerly convert all input data points, as we ignore most.
      scale_down_image<T>(images, batch_size, out_height, out_width, channels,
                          xs, ys, output);
    } else {
      for (int b = 0; b < batch_size; ++b) {
        const float* converted_image = Converter<T>::convert_image_to_float(
            images, b, in_height, in_width, channels, &converted_image_v);
        scale_similar_image(converted_image, b, out_height, out_width, channels,
                            in_height, in_width, xs, ys, output);
      }
    }
  }
};
}  // namespace functor

template <typename Device, typename T>
class ResizeBilinearOpGrad : public OpKernel {
 public:
  explicit ResizeBilinearOpGrad(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    // Validate input.
    // First argument is gradient with respect to resized image.
    const Tensor& input = context->input(0);
    const Tensor& original_image = context->input(1);

    ImageResizerGradientState st(align_corners_);
    st.ValidateAndCreateOutput(context, input, original_image);

    if (!context->status().ok()) return;

    typename TTypes<float, 4>::ConstTensor input_grad =
        input.tensor<float, 4>();
    typename TTypes<T, 4>::Tensor output_grad = st.output->tensor<T, 4>();

    functor::ResizeBilinearGrad<Device, T>()(context->eigen_device<Device>(),
                                             input_grad, st.height_scale,
                                             st.width_scale, output_grad);
  }

 private:
  bool align_corners_;
};

// Partial specialization of ResizeBilinearGrad functor for a CPUDevice.
namespace functor {
template <typename T>
struct ResizeBilinearGrad<CPUDevice, T> {
  void operator()(const CPUDevice& d,
                  typename TTypes<float, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output_grad) {
    const int batch = output_grad.dimension(0);
    const int64 original_height = output_grad.dimension(1);
    const int64 original_width = output_grad.dimension(2);
    const int channels = output_grad.dimension(3);

    const int64 resized_height = input_grad.dimension(1);
    const int64 resized_width = input_grad.dimension(2);

    output_grad.setZero();

    // Each resized pixel was computed as a weighted average of four input
    // pixels. Here we find the pixels that contributed to each output pixel
    // and add the corresponding coefficient to the gradient.
    // resized(b, y, x, c) = top_left * (1 - y) * (1 - x)
    //                       +  top_right * (1 - y) * x
    //                       +  bottom_left * y * (1 - x)
    //                       +  bottom_right * y * x
    for (int64 b = 0; b < batch; ++b) {
      for (int64 y = 0; y < resized_height; ++y) {
        const float in_y = y * height_scale;
        const int64 top_y_index = static_cast<int64>(floorf(in_y));
        const int64 bottom_y_index =
            std::min(static_cast<int64>(ceilf(in_y)), original_height - 1);
        const float y_lerp = in_y - top_y_index;
        const float inverse_y_lerp = (1.0f - y_lerp);
        for (int64 x = 0; x < resized_width; ++x) {
          const float in_x = x * width_scale;
          const int64 left_x_index = static_cast<int64>(floorf(in_x));
          const int64 right_x_index =
              std::min(static_cast<int64>(ceilf(in_x)), original_width - 1);
          const float x_lerp = in_x - left_x_index;
          const float inverse_x_lerp = (1.0f - x_lerp);
          for (int64 c = 0; c < channels; ++c) {
            output_grad(b, top_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * inverse_x_lerp);
            output_grad(b, top_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * inverse_y_lerp * x_lerp);
            output_grad(b, bottom_y_index, left_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * inverse_x_lerp);
            output_grad(b, bottom_y_index, right_x_index, c) +=
                T(input_grad(b, y, x, c) * y_lerp * x_lerp);
          }
        }
      }
    }
  }
};
}  // namespace functor

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<CPUDevice, T>);

TF_CALL_half(REGISTER_GRAD_KERNEL);
TF_CALL_float(REGISTER_GRAD_KERNEL);
TF_CALL_double(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#if GOOGLE_CUDA

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeBilinear")      \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeBilinearOp<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_KERNEL);

#undef REGISTER_KERNEL

#define REGISTER_GRAD_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("ResizeBilinearGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      ResizeBilinearOpGrad<GPUDevice, T>);

TF_CALL_GPU_NUMBER_TYPES_NO_HALF(REGISTER_GRAD_KERNEL);

#undef REGISTER_GRAD_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
