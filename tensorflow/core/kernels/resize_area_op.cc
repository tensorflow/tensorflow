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
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
class ResizeAreaOp : public OpKernel {
 public:
  explicit ResizeAreaOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    ImageResizerState st(align_corners_);
    st.ValidateAndCreateOutput(context, input);

    if (!context->status().ok()) return;

    typename TTypes<T, 4>::ConstTensor input_data = input.tensor<T, 4>();
    typename TTypes<float, 4>::Tensor output_data =
        st.output->tensor<float, 4>();

    // A temporary tensor for computing the sum.
    Tensor sum_tensor;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   TensorShape({st.channels}),
                                                   &sum_tensor));
    typename TTypes<float, 1>::Tensor sum_data = sum_tensor.vec<float>();

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
    float scale = 1.0 / (st.height_scale * st.width_scale);
    for (int64 b = 0; b < st.batch_size; ++b) {
      for (int64 y = 0; y < st.out_height; ++y) {
        const float in_y = y * st.height_scale;
        const float in_y1 = (y + 1) * st.height_scale;
        // The start and end height indices of all the cells that could
        // contribute to the target cell.
        int64 y_start = floor(in_y);
        int64 y_end = ceil(in_y1);

        for (int64 x = 0; x < st.out_width; ++x) {
          const float in_x = x * st.width_scale;
          const float in_x1 = (x + 1) * st.width_scale;
          // The start and end width indices of all the cells that could
          // contribute to the target cell.
          int64 x_start = floor(in_x);
          int64 x_end = ceil(in_x1);

          sum_data.setConstant(0.0);
          for (int64 i = y_start; i < y_end; ++i) {
            float scale_y =
                i < in_y ? (i + 1 > in_y1 ? st.height_scale : i + 1 - in_y)
                         : (i + 1 > in_y1 ? in_y1 - i : 1.0);
            for (int64 j = x_start; j < x_end; ++j) {
              float scale_x =
                  j < in_x ? (j + 1 > in_x1 ? st.width_scale : j + 1 - in_x)
                           : (j + 1 > in_x1 ? in_x1 - j : 1.0);
              for (int64 c = 0; c < st.channels; ++c) {
#define BOUND(val, limit) std::min(((limit)-1ll), (std::max(0ll, (val))))
                sum_data(c) += float(input_data(b, BOUND(i, st.in_height),
                                                BOUND(j, st.in_width), c)) *
                               scale_y * scale_x * scale;
#undef BOUND
              }
            }
          }
          for (int64 c = 0; c < st.channels; ++c) {
            output_data(b, y, x, c) = sum_data(c);
          }
        }
      }
    }
  }

 private:
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
