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

#include <algorithm>
#include <memory>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/image_resizer_state.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace {
struct CachedInterpolation {
  int64 start;
  int64 end;
  float start_scale;
  float end_minus_one_scale;
  bool needs_bounding;
};
}  // namespace

template <typename Device, typename T>
class ResizeAreaOp : public OpKernel {
 public:
  explicit ResizeAreaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  // Computes the sum of all x values defined by <x_interp> taken across
  // the y offsets and scales defined by y_ptrs and y_scales, for channel c.
  //
  // Note that <NeedsXBounding> is a template parameter to avoid a performance
  // penalty from dynamically checking it.
  template <bool NeedsXBounding>
  static void ComputePatchSumOf3Channels(float scale,
                                         const ImageResizerState& st,
                                         const std::vector<const T*>& y_ptrs,
                                         const std::vector<float>& y_scales,
                                         const CachedInterpolation& x_interp,
                                         float* output_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))

    float sum_0 = 0;
    float sum_1 = 0;
    float sum_2 = 0;
    for (int i = 0; i < y_ptrs.size(); ++i) {
      const T* ptr = y_ptrs[i];
      float scale_x = x_interp.start_scale;
      int64 offset = 3 * BOUND_IF_NEEDED(x_interp.start, st.in_width);
      float sum_y_0 = static_cast<float>(ptr[offset + 0]) * scale_x;
      float sum_y_1 = static_cast<float>(ptr[offset + 1]) * scale_x;
      float sum_y_2 = static_cast<float>(ptr[offset + 2]) * scale_x;

      if (x_interp.start + 1 != x_interp.end) {
        for (int64 x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
          int64 offset = 3 * BOUND_IF_NEEDED(x, st.in_width);
          sum_y_0 += static_cast<float>(ptr[offset + 0]);
          sum_y_1 += static_cast<float>(ptr[offset + 1]);
          sum_y_2 += static_cast<float>(ptr[offset + 2]);
        }
        scale_x = x_interp.end_minus_one_scale;
        offset = 3 * BOUND_IF_NEEDED(x_interp.end - 1, st.in_width);
        sum_y_0 += static_cast<float>(ptr[offset + 0]) * scale_x;
        sum_y_1 += static_cast<float>(ptr[offset + 1]) * scale_x;
        sum_y_2 += static_cast<float>(ptr[offset + 2]) * scale_x;
      }
      sum_0 += sum_y_0 * y_scales[i];
      sum_1 += sum_y_1 * y_scales[i];
      sum_2 += sum_y_2 * y_scales[i];
    }

    output_ptr[0] = sum_0 * scale;
    output_ptr[1] = sum_1 * scale;
    output_ptr[2] = sum_2 * scale;

#undef BOUND_IF_NEEDED
  }

  // Computes the sum of all x values defined by <x_interp> taken across
  // the y offsets and scales defined by y_ptrs and y_scales, for channel c.
  //
  // Note that <NeedsXBounding> is a template parameter to avoid a performance
  // penalty from dynamically checking it.
  template <bool NeedsXBounding>
  static void ComputePatchSum(float scale, const ImageResizerState& st,
                              const std::vector<const T*>& y_ptrs,
                              const std::vector<float>& y_scales,
                              const CachedInterpolation& x_interp,
                              float* output_ptr) {
#define BOUND_IF_NEEDED(x, y) (NeedsXBounding ? Bound(x, y) : (x))

    const auto num_channels = st.channels;
    for (int64 c = 0; c < num_channels; ++c) {
      float sum = 0;
      for (int i = 0; i < y_ptrs.size(); ++i) {
        const T* ptr = y_ptrs[i];
        float scale_x = x_interp.start_scale;
        float sum_y = static_cast<float>(
                          ptr[num_channels *
                                  BOUND_IF_NEEDED(x_interp.start, st.in_width) +
                              c]) *
                      scale_x;
        if (x_interp.start + 1 != x_interp.end) {
          for (int64 x = x_interp.start + 1; x < x_interp.end - 1; ++x) {
            sum_y += static_cast<float>(
                ptr[num_channels * BOUND_IF_NEEDED(x, st.in_width) + c]);
          }
          scale_x = x_interp.end_minus_one_scale;
          sum_y += static_cast<float>(
                       ptr[num_channels *
                               BOUND_IF_NEEDED(x_interp.end - 1, st.in_width) +
                           c]) *
                   scale_x;
        }
        sum += sum_y * y_scales[i];
      }
      output_ptr[c] = sum * scale;
    }
#undef BOUND_IF_NEEDED
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    // The op always did the correct thing with regard to pixel centers, so we
    // always pass false here for half_pixel_centers since ImageResizerState
    // enforces that if align_corners_ is true, half_pixel_centers must be
    // false.
    ImageResizerState st(align_corners_, /*unused half_pixel_centers=*/false);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data(input.tensor<T, 4>());

    // Precompute values used when iterating over x coordinates within a row.
    // Note that it may be useful to cache x_interps for a given
    // ImageResizerState.
    std::vector<CachedInterpolation> x_interps(st.out_width);
    for (int64 x = 0; x < st.out_width; ++x) {
      auto& x_interp = x_interps[x];
      const float in_x = x * st.width_scale;
      const float in_x1 = (x + 1) * st.width_scale;
      // The start and end width indices of all the cells that could
      // contribute to the target cell.
      int64 v = std::floor(in_x);
      x_interp.start = v;
      // TODO(cwhipkey): simplify this logic.
      x_interp.start_scale =
          v < in_x ? (v + 1 > in_x1 ? st.width_scale : v + 1 - in_x)
                   : (v + 1 > in_x1 ? in_x1 - v : 1.0);

      v = std::ceil(in_x1);
      x_interp.end = v;
      v = x_interp.end - 1;
      x_interp.end_minus_one_scale =
          v < in_x ? (v + 1 > in_x1 ? st.width_scale : v + 1 - in_x)
                   : (v + 1 > in_x1 ? in_x1 - v : 1.0);
      x_interp.needs_bounding =
          Bound(x_interp.start, st.in_width) != x_interp.start ||
          Bound(x_interp.end - 1, st.in_width) != (x_interp.end - 1);
    }

    if (st.channels == 3) {
      ComputeLoop<3>(st, x_interps, input_data);
    } else {
      ComputeLoop<-1>(st, x_interps, input_data);
    }
  }

  template <int64 kKnownNumChannels>
  void ComputeLoop(const ImageResizerState& st,
                   const std::vector<CachedInterpolation>& x_interps,
                   typename TTypes<T, 4>::ConstTensor input_data) {
    TTypes<float, 4>::Tensor output_data = st.output->tensor<float, 4>();

    // When using this algorithm for downsizing, the target pixel value is the
    // weighted average of all the source pixels. The weight is determined by
    // the contribution percentage of the source pixel.
    //
    // Let "scale" be "target_image_size/source_image_size". If 1/n of the
    // source pixel contributes to the target pixel, then the weight is (1/n *
    // scale); if the complete source pixel contributes to the target pixel,
    // then the weight is scale.
    //
    // To visualize the implementation, use one dimension as an example:
    // Resize in[4] to out[3].
    //   scale = 3/4 = 0.75
    //   out[0]: in[0] and 1/3 of in[1]
    //   out[1]: 2/3 of in[1] and 2/3 of in[2]
    //   out[2]: 1/3 of in[2] and in[1]
    // Hence, the output pixel values are:
    //   out[0] = (in[0] * 1.0 + in[1] * 1/3) * scale
    //   out[1] = (in[1] * 2/3 + in[2] * 2/3 * scale
    //   out[2] = (in[3] * 1/3 + in[3] * 1.0) * scale
    const T* const input_ptr = input_data.data();
    std::vector<float> y_scales;
    std::vector<const T*> y_ptrs;
    float scale = 1.0 / (st.height_scale * st.width_scale);
    float* output_ptr = output_data.data();
    for (int64 b = 0; b < st.batch_size; ++b) {
      for (int64 y = 0; y < st.out_height; ++y) {
        const float in_y = y * st.height_scale;
        const float in_y1 = (y + 1) * st.height_scale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        const int64 y_start = std::floor(in_y);
        const int64 y_end = std::ceil(in_y1);
        y_scales.clear();
        y_ptrs.clear();
        for (int64 i = y_start; i < y_end; ++i) {
          float scale_y;
          if (i < in_y) {
            scale_y = (i + 1 > in_y1 ? st.height_scale : i + 1 - in_y);
          } else {
            scale_y = (i + 1 > in_y1 ? in_y1 - i : 1.0);
          }
          // TODO(cwhipkey): can this data unified with CachedInterpolation?
          y_scales.push_back(scale_y);
          y_ptrs.push_back(
              input_ptr + (b * st.in_height * st.in_width * st.channels +
                           Bound(i, st.in_height) * st.in_width * st.channels));
        }

        if (kKnownNumChannels == 3) {
          for (int64 x = 0; x < st.out_width; ++x) {
            const CachedInterpolation& x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSumOf3Channels<true>(scale, st, y_ptrs, y_scales,
                                               x_interp, output_ptr);
            } else {
              ComputePatchSumOf3Channels<false>(scale, st, y_ptrs, y_scales,
                                                x_interp, output_ptr);
            }
            output_ptr += 3;
          }
        } else {
          for (int64 x = 0; x < st.out_width; ++x) {
            const CachedInterpolation& x_interp = x_interps[x];
            if (x_interp.needs_bounding) {
              ComputePatchSum<true>(scale, st, y_ptrs, y_scales, x_interp,
                                    output_ptr);
            } else {
              ComputePatchSum<false>(scale, st, y_ptrs, y_scales, x_interp,
                                     output_ptr);
            }
            output_ptr += st.channels;
          }
        }
      }
    }
  }

 private:
  static EIGEN_ALWAYS_INLINE int64 Bound(int64 val, int64 limit) {
    return std::min(limit - 1, std::max(int64{0}, val));
  }

  bool align_corners_;
};

#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("ResizeArea")          \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("size"),    \
                          ResizeAreaOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL

}  // namespace tensorflow
