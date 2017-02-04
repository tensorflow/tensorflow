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
inline float image_lerp(const T* input_image, int64 in_x_lower,
                        int64 in_x_upper, float xs_lerp, int64 in_y_lower,
                        int64 in_y_upper, float ys_lerp, int c) {
  const float top_left(input_image[in_y_lower + in_x_lower + c]);
  const float top_right(input_image[in_y_lower + in_x_upper + c]);
  const float bottom_left(input_image[in_y_upper + in_x_lower + c]);
  const float bottom_right(input_image[in_y_upper + in_x_upper + c]);
  return compute_lerp(top_left, top_right, bottom_left, bottom_right, xs_lerp,
                      ys_lerp);
}

template <typename T>
void scale_down_image(
    typename TTypes<T, 4>::ConstTensor images, const int batch_size,
    const int64 out_height, const int64 out_width, const int channels,
    const std::vector<CachedInterpolation>& xs,
    const std::vector<CachedInterpolation>& ys,
    typename TTypes<float, 4>::Tensor output) TF_ATTRIBUTE_NOINLINE;
template <typename T>
void scale_down_image(typename TTypes<T, 4>::ConstTensor images,
                      const int batch_size, const int64 out_height,
                      const int64 out_width, const int channels,
                      const std::vector<CachedInterpolation>& xs_vec,
                      const std::vector<CachedInterpolation>& ys,
                      typename TTypes<float, 4>::Tensor output) {
  // Do not eagerly convert all input data points, as we ignore most.
  if (channels == 3) {
    for (int b = 0; b < batch_size; ++b) {
      // Compute the interpolation
      for (int64 y = 0; y < out_height; ++y) {
        const int64 ys_lower = ys[y].lower;
        const int64 ys_upper = ys[y].upper;
        const float ys_lerp = ys[y].lerp;
        const CachedInterpolation* xs_ptr = xs_vec.data();
        for (int64 x = 0; x < out_width; ++x) {
          const int64 xs_lower = xs_ptr->lower;
          const int64 xs_upper = xs_ptr->upper;
          const float xs_lerp = xs_ptr->lerp;
          xs_ptr++;

          const float top_left0(images(b, ys_lower, xs_lower, 0));
          const float top_right0(images(b, ys_lower, xs_upper, 0));
          const float bottom_left0(images(b, ys_upper, xs_lower, 0));
          const float bottom_right0(images(b, ys_upper, xs_upper, 0));
          const float out0 = compute_lerp(top_left0, top_right0, bottom_left0,
                                          bottom_right0, xs_lerp, ys_lerp);

          const float top_left1(images(b, ys_lower, xs_lower, 1));
          const float top_right1(images(b, ys_lower, xs_upper, 1));
          const float bottom_left1(images(b, ys_upper, xs_lower, 1));
          const float bottom_right1(images(b, ys_upper, xs_upper, 1));
          const float out1 = compute_lerp(top_left1, top_right1, bottom_left1,
                                          bottom_right1, xs_lerp, ys_lerp);

          const float top_left2(images(b, ys_lower, xs_lower, 2));
          const float top_right2(images(b, ys_lower, xs_upper, 2));
          const float bottom_left2(images(b, ys_upper, xs_lower, 2));
          const float bottom_right2(images(b, ys_upper, xs_upper, 2));
          const float out2 = compute_lerp(top_left2, top_right2, bottom_left2,
                                          bottom_right2, xs_lerp, ys_lerp);

          float* dest = &output(b, y, x, 0);
          dest[0] = out0;
          dest[1] = out1;
          dest[2] = out2;
        }
      }
    }
  } else {
    for (int b = 0; b < batch_size; ++b) {
      // Compute the interpolation
      for (int64 y = 0; y < out_height; ++y) {
        const CachedInterpolation* xs = xs_vec.data();
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
}

template <typename T>
void scale_up_image(
    const T* input_image, const int batch_index, const int64 out_height,
    const int64 out_width, const int channels, const int64 in_height,
    const int64 in_width, const std::vector<CachedInterpolation>& xs,
    const std::vector<CachedInterpolation>& ys,
    typename TTypes<float, 4>::Tensor output) TF_ATTRIBUTE_NOINLINE;

template <typename T>
void scale_up_image(const T* input_image, const int batch_index,
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
        const float top_left(input_image[in_y_lower + in_x_lower + c]);
        const float top_right(input_image[in_y_lower + in_x_upper + c]);
        const float bottom_left(input_image[in_y_upper + in_x_lower + c]);
        const float bottom_right(input_image[in_y_upper + in_x_upper + c]);
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

template <typename T>
void scale_similar_image(
    const T* input_image, const int b, const int64 out_height,
    const int64 out_width, const int channels, const int64 in_height,
    const int64 in_width, const std::vector<CachedInterpolation>& xs_vec,
    const std::vector<CachedInterpolation>& ys,
    typename TTypes<float, 4>::Tensor output) TF_ATTRIBUTE_NOINLINE;
template <typename T>
void scale_similar_image(const T* input_image, const int b,
                         const int64 out_height, const int64 out_width,
                         const int channels, const int64 in_height,
                         const int64 in_width,
                         const std::vector<CachedInterpolation>& xs_vec,
                         const std::vector<CachedInterpolation>& ys,
                         typename TTypes<float, 4>::Tensor output) {
  if (channels == 3) {
    // Compute the interpolation
    for (int64 y = 0; y < out_height; ++y) {
      const int64 in_y_lower = ys[y].lower * in_width * channels;
      const int64 in_y_upper = ys[y].upper * in_width * channels;
      const float ys_lerp = ys[y].lerp;
      // Similar-sized images do not have a set of inner loops.
      const CachedInterpolation* xs_ptr = xs_vec.data();
      for (int64 x = 0; x < out_width; ++x) {
        const int64 in_x_lower = xs_ptr->lower * 3;
        const int64 in_x_upper = xs_ptr->upper * 3;
        const float xs_lerp = xs_ptr->lerp;
        xs_ptr++;

        const float out0 =
            image_lerp(input_image, in_x_lower, in_x_upper, xs_lerp, in_y_lower,
                       in_y_upper, ys_lerp, 0);
        const float out1 =
            image_lerp(input_image, in_x_lower, in_x_upper, xs_lerp, in_y_lower,
                       in_y_upper, ys_lerp, 1);
        const float out2 =
            image_lerp(input_image, in_x_lower, in_x_upper, xs_lerp, in_y_lower,
                       in_y_upper, ys_lerp, 2);
        float* dest = &output(b, y, x, 0);
        dest[0] = out0;
        dest[1] = out1;
        dest[2] = out2;
      }
    }
  } else {
    // Compute the interpolation
    for (int64 y = 0; y < out_height; ++y) {
      const int64 in_y_lower = ys[y].lower * in_width * channels;
      const int64 in_y_upper = ys[y].upper * in_width * channels;
      const float ys_lerp = ys[y].lerp;
      // Similar-sized images do not have a set of inner loops.
      const CachedInterpolation* xs_ptr = xs_vec.data();
      for (int64 x = 0; x < out_width; ++x) {
        const int64 in_x_lower = xs_ptr->lower * channels;
        const int64 in_x_upper = xs_ptr->upper * channels;
        const float xs_lerp = xs_ptr->lerp;
        xs_ptr++;
        for (int c = 0; c < channels; ++c) {
          const float top_left(input_image[in_y_lower + in_x_lower + c]);
          const float top_right(input_image[in_y_lower + in_x_upper + c]);
          const float bottom_left(input_image[in_y_upper + in_x_lower + c]);
          const float bottom_right(input_image[in_y_upper + in_x_upper + c]);
          output(b, y, x, c) = compute_lerp(top_left, top_right, bottom_left,
                                            bottom_right, xs_lerp, ys_lerp);
        }
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

    // Compute the cached interpolation weights on the x and y dimensions.
    compute_interpolation_weights(scale_pattern, out_height, in_height,
                                  height_scale, ys.data());
    compute_interpolation_weights(scale_pattern, out_width, in_width,
                                  width_scale, xs.data());

    if (scale_pattern == SCALE_UP) {
      for (int b = 0; b < batch_size; ++b) {
        scale_up_image<T>(&images(b, 0, 0, 0), b, out_height, out_width,
                          channels, in_height, in_width, xs, ys, output);
      }
    } else if (scale_pattern == SCALE_DOWN) {
      // Do not eagerly convert all input data points, as we ignore most.
      scale_down_image<T>(images, batch_size, out_height, out_width, channels,
                          xs, ys, output);
    } else {
      for (int b = 0; b < batch_size; ++b) {
        scale_similar_image<T>(&images(b, 0, 0, 0), b, out_height, out_width,
                               channels, in_height, in_width, xs, ys, output);
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
