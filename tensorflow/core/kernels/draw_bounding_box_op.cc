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

namespace {

std::vector<std::vector<float>> DefaultColorTable(int depth) {
  std::vector<std::vector<float>> color_table;
  color_table.emplace_back(std::vector<float>({1, 1, 0, 1})); // 0: yellow
  color_table.emplace_back(std::vector<float>({0, 0, 1, 1})); // 1: blue
  color_table.emplace_back(std::vector<float>({1, 0, 0, 1})); // 2: red
  color_table.emplace_back(std::vector<float>({0, 1, 0, 1})); // 3: lime
  color_table.emplace_back(std::vector<float>({0.5, 0, 0.5, 1})); // 4: purple
  color_table.emplace_back(std::vector<float>({0.5, 0.5, 0, 1})); // 5: olive
  color_table.emplace_back(std::vector<float>({0.5, 0, 0, 1})); // 6: maroon
  color_table.emplace_back(std::vector<float>({0, 0, 0.5, 1})); // 7: navy blue
  color_table.emplace_back(std::vector<float>({0, 1, 1, 1})); // 8: aqua
  color_table.emplace_back(std::vector<float>({1, 0, 1, 1})); // 9: fuchsia

  if (depth == 1) {
    for (int64 i = 0; i < color_table.size(); i++) {
      color_table[i][0] = 1;
    }
  }
  return  color_table;
}
}  // namespace

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
    std::vector<std::vector<float>> color_table;
    if (context->num_inputs() == 3) {
      const Tensor& colors_tensor = context->input(2);
      OP_REQUIRES(context, colors_tensor.shape().dims() == 2,
                  errors::InvalidArgument("colors must be a 2-D matrix",
                                          colors_tensor.shape().DebugString()));
      OP_REQUIRES(context, colors_tensor.shape().dim_size(1) >= depth,
                  errors::InvalidArgument("colors must have equal or more ",
                                          "channels than the image provided: ",
                                          colors_tensor.shape().DebugString()));
      if (colors_tensor.NumElements() != 0) {
        color_table.clear();

        auto colors = colors_tensor.matrix<float>();
        for (int64 i = 0; i < colors.dimension(0); i++) {
          std::vector<float> color_value(4);
          for (int64 j = 0; j < 4; j++) {
            color_value[j] = colors(i, j);
          }
          color_table.emplace_back(color_value);
        }
      }
    }
    if (color_table.empty()) {
      color_table = DefaultColorTable(depth);
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
        int64 color_index = bb % color_table.size();
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

#define REGISTER_CPU_KERNEL(T)                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DrawBoundingBoxes").Device(DEVICE_CPU).TypeConstraint<T>("T"),   \
      DrawBoundingBoxesOp<T>);                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DrawBoundingBoxesV2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DrawBoundingBoxesOp<T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);

}  // namespace tensorflow
