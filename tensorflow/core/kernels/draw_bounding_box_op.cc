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
// See ../ops/image_ops.cc for details.
#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

template <class T>
class DrawBoundingBoxesOp : public OpKernel {
 public:
  explicit DrawBoundingBoxesOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& images = context->input(0);
    const Tensor& boxes = context->input(1);
    const int64 depth = images.dim_size(3);

    OP_REQUIRES(context, images.dims() == 4,
                errors::InvalidArgument("The rank of the images should be 4"));
    OP_REQUIRES(
        context, boxes.dims() == 3,
        errors::InvalidArgument("The rank of the boxes tensor should be 3"));
    OP_REQUIRES(context, images.dim_size(0) == boxes.dim_size(0),
                errors::InvalidArgument("The batch sizes should be the same"));

    OP_REQUIRES(
        context, depth == 4 || depth == 1 || depth == 3,
        errors::InvalidArgument("Channel depth should be either 1 (GRY), "
                                "3 (RGB), or 4 (RGBA)"));

    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    const int64 color_table_length = 10;

    // 0: yellow
    // 1: blue
    // 2: red
    // 3: lime
    // 4: purple
    // 5: olive
    // 6: maroon
    // 7: navy blue
    // 8: aqua
    // 9: fuchsia
    float color_table[color_table_length][4] = {
        {1, 1, 0, 1},     {0, 0, 1, 1},     {1, 0, 0, 1},   {0, 1, 0, 1},
        {0.5, 0, 0.5, 1}, {0.5, 0.5, 0, 1}, {0.5, 0, 0, 1}, {0, 0, 0.5, 1},
        {0, 1, 1, 1},     {1, 0, 1, 1},
    };

    // Reset first color channel to 1 if image is GRY.
    // For GRY images, this means all bounding boxes will be white.
    if (depth == 1) {
      for (int64 i = 0; i < color_table_length; i++) {
        color_table[i][0] = 1;
      }
    }
    Tensor* output;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(
            0, TensorShape({batch_size, height, width, depth}), &output));

    output->tensor<T, 4>() = images.tensor<T, 4>();
    auto canvas = output->tensor<T, 4>();

    for (int64 b = 0; b < batch_size; ++b) {
      const int64 num_boxes = boxes.dim_size(1);
      const auto tboxes = boxes.tensor<T, 3>();
      for (int64 bb = 0; bb < num_boxes; ++bb) {
        int64 color_index = bb % color_table_length;
        const int64 min_box_row =
            static_cast<float>(tboxes(b, bb, 0)) * (height - 1);
        const int64 min_box_row_clamp = std::max<int64>(min_box_row, int64{0});
        const int64 max_box_row =
            static_cast<float>(tboxes(b, bb, 2)) * (height - 1);
        const int64 max_box_row_clamp =
            std::min<int64>(max_box_row, height - 1);
        const int64 min_box_col =
            static_cast<float>(tboxes(b, bb, 1)) * (width - 1);
        const int64 min_box_col_clamp = std::max<int64>(min_box_col, int64{0});
        const int64 max_box_col =
            static_cast<float>(tboxes(b, bb, 3)) * (width - 1);
        const int64 max_box_col_clamp = std::min<int64>(max_box_col, width - 1);

        if (min_box_row > max_box_row || min_box_col > max_box_col) {
          LOG(WARNING) << "Bounding box (" << min_box_row << "," << min_box_col
                       << "," << max_box_row << "," << max_box_col
                       << ") is inverted and will not be drawn.";
          continue;
        }
        if (min_box_row >= height || max_box_row < 0 || min_box_col >= width ||
            max_box_col < 0) {
          LOG(WARNING) << "Bounding box (" << min_box_row << "," << min_box_col
                       << "," << max_box_row << "," << max_box_col
                       << ") is completely outside the image"
                       << " and will not be drawn.";
          continue;
        }

        // At this point, {min,max}_box_{row,col}_clamp are inside the
        // image.
        CHECK_GE(min_box_row_clamp, 0);
        CHECK_GE(max_box_row_clamp, 0);
        CHECK_LT(min_box_row_clamp, height);
        CHECK_LT(max_box_row_clamp, height);
        CHECK_GE(min_box_col_clamp, 0);
        CHECK_GE(max_box_col_clamp, 0);
        CHECK_LT(min_box_col_clamp, width);
        CHECK_LT(max_box_col_clamp, width);

        // At this point, the min_box_row and min_box_col are either
        // in the image or above/left of it, and max_box_row and
        // max_box_col are either in the image or below/right or it.
        CHECK_LT(min_box_row, height);
        CHECK_GE(max_box_row, 0);
        CHECK_LT(min_box_col, width);
        CHECK_GE(max_box_col, 0);

        // Draw top line.
        if (min_box_row >= 0) {
          for (int64 j = min_box_col_clamp; j <= max_box_col_clamp; ++j)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, min_box_row, j, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw bottom line.
        if (max_box_row < height) {
          for (int64 j = min_box_col_clamp; j <= max_box_col_clamp; ++j)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, max_box_row, j, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw left line.
        if (min_box_col >= 0) {
          for (int64 i = min_box_row_clamp; i <= max_box_row_clamp; ++i)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, i, min_box_col, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
        // Draw right line.
        if (max_box_col < width) {
          for (int64 i = min_box_row_clamp; i <= max_box_row_clamp; ++i)
            for (int64 c = 0; c < depth; c++) {
              canvas(b, i, max_box_col, c) =
                  static_cast<T>(color_table[color_index][c]);
            }
        }
      }
    }
  }
};

#define REGISTER_CPU_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("DrawBoundingBoxes").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DrawBoundingBoxesOp<T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);

}  // namespace tensorflow
