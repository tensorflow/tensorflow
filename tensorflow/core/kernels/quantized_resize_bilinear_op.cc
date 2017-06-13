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

// Implements a quantized version of the resize bilinear op.

#define EIGEN_USE_THREADS

#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#define USE_NEON
#define QUANTIZED_ADD_USE_NEON
#include <arm_neon.h>
#endif

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

static constexpr bool USE_REFERENCE = false;

namespace {
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64 lower;  // Lower source index used in the interpolation
  int64 upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
  int32 ilerp;
};

inline void ComputeInterpolationWeights(const int64 out_size,
                                        const int64 in_size, const float scale,
                                        const int resolution,
                                        CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64 i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
    interpolation[i].ilerp =
        static_cast<int32>((in - interpolation[i].lower) * (1 << resolution));
  }
}

inline std::vector<CachedInterpolation> BuildLerpCache(const int64 out_size,
                                                       const int64 in_size,
                                                       const float scale,
                                                       const int index_step,
                                                       const int resolution) {
  std::vector<CachedInterpolation> vec(out_size + 1);
  // Compute the cached interpolation weights on the x and y dimensions.
  ComputeInterpolationWeights(out_size, in_size, scale, resolution, vec.data());
  CHECK(index_step > 0);
  if (index_step > 1) {
    for (int i = 0; i < vec.size(); ++i) {
      vec[i].lower *= index_step;
      vec[i].upper *= index_step;
    }
  }
  return vec;
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
template <typename T>
inline T ComputeLarpReference(const T in_top_left, const T in_top_right,
                              const T in_bottom_left, const T in_bottom_right,
                              const float x_lerp, const float y_lerp,
                              const float min, const float max) {
  const float top_left = QuantizedToFloat<T>(in_top_left, min, max);
  const float top_right = QuantizedToFloat<T>(in_top_right, min, max);
  const float bottom_left = QuantizedToFloat<T>(in_bottom_left, min, max);
  const float bottom_right = QuantizedToFloat<T>(in_bottom_right, min, max);
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  const float out = top + (bottom - top) * y_lerp;
  return FloatToQuantized<T>(out, min, max);
}

template <typename T, typename T_SCALE, typename T_CALC>
inline T_CALC MulOffset(T a, T b, T_SCALE c) {
  return (static_cast<T_CALC>(a) - static_cast<T_CALC>(b)) *
         static_cast<T_CALC>(c);
}

template <int RESOLUTION, typename T, typename T_SCALE, typename T_CALC>
inline T ComputeLarp(const T top_left, const T top_right, const T bottom_left,
                     const T bottom_right, const T_SCALE x_lerp,
                     const T_SCALE y_lerp, const float min, const float max) {
  constexpr T_CALC RESOLUTION_MULT = (1 << RESOLUTION);
  const T_CALC top = static_cast<T_CALC>(top_left) * RESOLUTION_MULT +
                     MulOffset<T, T_SCALE, T_CALC>(top_right, top_left, x_lerp);
  const T_CALC bottom =
      static_cast<T_CALC>(bottom_left) * RESOLUTION_MULT +
      MulOffset<T, T_SCALE, T_CALC>(bottom_right, bottom_left, x_lerp);
  const T_CALC out = top + (bottom - top) / RESOLUTION_MULT * y_lerp;
  return static_cast<T>(
      static_cast<int32>((out + RESOLUTION_MULT / 2) / RESOLUTION_MULT));
}

template <typename T>
void ResizeImageReference(typename TTypes<T, 4>::ConstTensor images,
                          const int batch_size, const int64 in_height,
                          const int64 in_width, const int64 out_height,
                          const int64 out_width, const int channels,
                          const float height_scale, const float width_scale,
                          const float in_min, const float in_max,
                          typename TTypes<T, 4>::Tensor* output) {
  CHECK_NOTNULL(output);

  const std::vector<CachedInterpolation> xs_vec =
      BuildLerpCache(out_width, in_width, width_scale, channels, 0);
  const std::vector<CachedInterpolation> ys_vec =
      BuildLerpCache(out_height, in_height, height_scale, 1, 0);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const T* input_b_ptr = images.data();
  const CachedInterpolation* xs = xs_vec.data();

  T* output_y_ptr = output->data();
  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const T* ys_input_lower_ptr = input_b_ptr + ys_vec[y].lower * in_row_size;
      const T* ys_input_upper_ptr = input_b_ptr + ys_vec[y].upper * in_row_size;
      const float ys_lerp = ys_vec[y].lerp;
      for (int64 x = 0; x < out_width; ++x) {
        const int64 xs_lower = xs[x].lower;
        const int64 xs_upper = xs[x].upper;
        const float xs_lerp = xs[x].lerp;
        for (int c = 0; c < channels; ++c) {
          const T top_left = ys_input_lower_ptr[xs_lower + c];
          const T top_right = ys_input_lower_ptr[xs_upper + c];
          const T bottom_left = ys_input_upper_ptr[xs_lower + c];
          const T bottom_right = ys_input_upper_ptr[xs_upper + c];
          const T val = ComputeLarpReference<T>(
              top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp,
              in_min, in_max);
          output_y_ptr[x * channels + c] = val;
        }
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <typename T>
void ResizeImage(typename TTypes<T, 4>::ConstTensor images,
                 const int batch_size, const int64 in_height,
                 const int64 in_width, const int64 out_height,
                 const int64 out_width, const int channels,
                 const float height_scale, const float width_scale,
                 const float in_min, const float in_max,
                 typename TTypes<T, 4>::Tensor* output) {
  ResizeImageReference<T>(images, batch_size, in_height, in_width, out_height,
                          out_width, channels, height_scale, width_scale,
                          in_min, in_max, output);
}

template <>
void ResizeImage<qint32>(typename TTypes<qint32, 4>::ConstTensor images,
                         const int batch_size, const int64 in_height,
                         const int64 in_width, const int64 out_height,
                         const int64 out_width, const int channels,
                         const float height_scale, const float width_scale,
                         const float in_min, const float in_max,
                         typename TTypes<qint32, 4>::Tensor* output) {
  constexpr int RESOLUTION = 16;

  CHECK_NOTNULL(output);

  const std::vector<CachedInterpolation> xs_vec =
      BuildLerpCache(out_width, in_width, width_scale, channels, RESOLUTION);
  const std::vector<CachedInterpolation> ys_vec =
      BuildLerpCache(out_height, in_height, height_scale, 1, RESOLUTION);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const qint32* input_b_ptr = images.data();
  const CachedInterpolation* xs = xs_vec.data();

  qint32* output_y_ptr = output->data();

  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const qint32* ys_input_lower_ptr =
          input_b_ptr + ys_vec[y].lower * in_row_size;
      const qint32* ys_input_upper_ptr =
          input_b_ptr + ys_vec[y].upper * in_row_size;
      const int32 ys_ilerp = ys_vec[y].ilerp;
      for (int64 x = 0; x < out_width; ++x) {
        const int64 xs_lower = xs[x].lower;
        const int64 xs_upper = xs[x].upper;
        const int32 xs_ilerp = xs[x].ilerp;
        for (int c = 0; c < channels; ++c) {
          const qint32 top_left = ys_input_lower_ptr[xs_lower + c];
          const qint32 top_right = ys_input_lower_ptr[xs_upper + c];
          const qint32 bottom_left = ys_input_upper_ptr[xs_lower + c];
          const qint32 bottom_right = ys_input_upper_ptr[xs_upper + c];
          const qint32 val = ComputeLarp<RESOLUTION, qint32, int32, int64>(
              top_left, top_right, bottom_left, bottom_right, xs_ilerp,
              ys_ilerp, in_min, in_max);
          output_y_ptr[x * channels + c] = val;
        }
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <>
void ResizeImage<quint8>(typename TTypes<quint8, 4>::ConstTensor images,
                         const int batch_size, const int64 in_height,
                         const int64 in_width, const int64 out_height,
                         const int64 out_width, const int channels,
                         const float height_scale, const float width_scale,
                         const float in_min, const float in_max,
                         typename TTypes<quint8, 4>::Tensor* output) {
  constexpr int RESOLUTION = 8;

  CHECK_NOTNULL(output);

  const std::vector<CachedInterpolation> xs_vec =
      BuildLerpCache(out_width, in_width, width_scale, channels, RESOLUTION);
  const std::vector<CachedInterpolation> ys_vec =
      BuildLerpCache(out_height, in_height, height_scale, 1, RESOLUTION);

  const int64 in_row_size = in_width * channels;
  const int64 in_batch_num_values = in_height * in_row_size;
  const int64 out_row_size = out_width * channels;

  const quint8* input_b_ptr = images.data();
  const CachedInterpolation* xs = xs_vec.data();

  quint8* output_y_ptr = output->data();

  for (int b = 0; b < batch_size; ++b) {
    for (int64 y = 0; y < out_height; ++y) {
      const quint8* ys_input_lower_ptr =
          input_b_ptr + ys_vec[y].lower * in_row_size;
      const quint8* ys_input_upper_ptr =
          input_b_ptr + ys_vec[y].upper * in_row_size;
      const int32 ys_ilerp = ys_vec[y].ilerp;
      for (int64 x = 0; x < out_width; ++x) {
        const int64 xs_lower = xs[x].lower;
        const int64 xs_upper = xs[x].upper;
        const int32 xs_ilerp = xs[x].ilerp;
        for (int c = 0; c < channels; ++c) {
          const quint8 top_left = ys_input_lower_ptr[xs_lower + c];
          const quint8 top_right = ys_input_lower_ptr[xs_upper + c];
          const quint8 bottom_left = ys_input_upper_ptr[xs_lower + c];
          const quint8 bottom_right = ys_input_upper_ptr[xs_upper + c];
          const quint8 val = ComputeLarp<RESOLUTION, quint8, int32, int32>(
              top_left, top_right, bottom_left, bottom_right, xs_ilerp,
              ys_ilerp, in_min, in_max);
          output_y_ptr[x * channels + c] = val;
        }
      }
      output_y_ptr += out_row_size;
    }
    input_b_ptr += in_batch_num_values;
  }
}

template <typename T>
void ResizeBilinear(const typename TTypes<T, 4>::ConstTensor& images,
                    const float height_scale, const float width_scale,
                    const float in_min, const float in_max,
                    typename TTypes<T, 4>::Tensor* output) {
  CHECK_NOTNULL(output);

  const int batch_size = images.dimension(0);
  const int64 in_height = images.dimension(1);
  const int64 in_width = images.dimension(2);
  const int channels = images.dimension(3);

  const int64 out_height = output->dimension(1);
  const int64 out_width = output->dimension(2);

  // Handle no-op resizes efficiently.
  if (out_height == in_height && out_width == in_width) {
    *output = images.template cast<T>();
    return;
  }

  if (USE_REFERENCE) {
    ResizeImageReference<T>(images, batch_size, in_height, in_width, out_height,
                            out_width, channels, height_scale, width_scale,
                            in_min, in_max, output);
  } else {
    ResizeImage<T>(images, batch_size, in_height, in_width, out_height,
                   out_width, channels, height_scale, width_scale, in_min,
                   in_max, output);
  }
}

}  // namespace

template <class T>
class QuantizedResizeBilinearOp : public OpKernel {
 public:
  explicit QuantizedResizeBilinearOp(OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const float in_min = context->input(2).flat<float>()(0);
    const float in_max = context->input(3).flat<float>()(0);

    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    // Return if the output is empty.
    if (st.output->NumElements() == 0) return;

    typename TTypes<T, 4>::ConstTensor image_data = input.tensor<T, 4>();
    typename TTypes<T, 4>::Tensor output_data = st.output->tensor<T, 4>();

    ResizeBilinear<T>(image_data, st.height_scale, st.width_scale, in_min,
                      in_max, &output_data);
    Tensor* out_min = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {}, &out_min));
    out_min->flat<float>()(0) = in_min;

    Tensor* out_max = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(2, {}, &out_max));
    out_max->flat<float>()(0) = in_max;
  }

 private:
  bool align_corners_;

  TF_DISALLOW_COPY_AND_ASSIGN(QuantizedResizeBilinearOp<T>);
};

#define REGISTER_CPU_KERNEL(type)                         \
  REGISTER_KERNEL_BUILDER(Name("QuantizedResizeBilinear") \
                              .Device(DEVICE_CPU)         \
                              .HostMemory("size")         \
                              .TypeConstraint<type>("T"), \
                          QuantizedResizeBilinearOp<type>)

REGISTER_CPU_KERNEL(::tensorflow::quint8);
REGISTER_CPU_KERNEL(::tensorflow::qint32);
REGISTER_CPU_KERNEL(float);

}  // namespace tensorflow
