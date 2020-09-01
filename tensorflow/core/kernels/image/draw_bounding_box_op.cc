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
  color_table.emplace_back(std::vector<float>({1, 1, 0, 1}));      // 0: yellow
  color_table.emplace_back(std::vector<float>({0, 0, 1, 1}));      // 1: blue
  color_table.emplace_back(std::vector<float>({1, 0, 0, 1}));      // 2: red
  color_table.emplace_back(std::vector<float>({0, 1, 0, 1}));      // 3: lime
  color_table.emplace_back(std::vector<float>({0.5, 0, 0.5, 1}));  // 4: purple
  color_table.emplace_back(std::vector<float>({0.5, 0.5, 0, 1}));  // 5: olive
  color_table.emplace_back(std::vector<float>({0.5, 0, 0, 1}));    // 6: maroon
  color_table.emplace_back(std::vector<float>({0, 0, 0.5, 1}));  // 7: navy blue
  color_table.emplace_back(std::vector<float>({0, 1, 1, 1}));    // 8: aqua
  color_table.emplace_back(std::vector<float>({1, 0, 1, 1}));    // 9: fuchsia

  if (depth == 1) {
    for (int64 i = 0; i < color_table.size(); i++) {
      color_table[i][0] = 1;
    }
  }
  return color_table;
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
        errors::InvalidArgument("Channel depth should be either 1 (GREY), "
                                "3 (RGB), or 4 (RGBA)"));

    const int64 batch_size = images.dim_size(0);
    const int64 height = images.dim_size(1);
    const int64 width = images.dim_size(2);
    std::vector<std::vector<float>> color_table;
    int64 thickness = int64{1};
    
    if (context->num_inputs() > 2) {  // DrawBoundingBoxesV2 and V3
      const Tensor& colors_tensor = context->input(2);
      OP_REQUIRES(context, colors_tensor.shape().dims() == 2,
                  errors::InvalidArgument("colors must be a 2-D matrix",
                                          colors_tensor.shape().DebugString()));
      OP_REQUIRES(context, colors_tensor.shape().dim_size(1) >= depth,
                  errors::InvalidArgument("colors must have equal or more ",
                                          "channels than the image provided: ",
                                          colors_tensor.shape().DebugString()));
      // If there's a third argument to the function with more than one element,
      // (it has to be a 2D tensor with as many channels as the image though),
      // update `color_table` with the colors passed instead of the default
      // values it will take.
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

      if (context->num_inputs() > 3) { // DrawBoundingBoxesV3
        thickness = static_cast<int64>(context->input(3));
        OP_REQUIRES(context, thickness > 0,
                errors::InvalidArgument("Thickness should be at least 1"));
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

        // Every box has four lines, top, left, bottom and right
        // And each line has a {_outer,_inner} to account for thickness
        
        // First extract the user-passed coordinates in image coordinates
        const int64 min_box_row =
            static_cast<float>(tboxes(b, bb, 0)) * (height - 1);
        const int64 max_box_row =
            static_cast<float>(tboxes(b, bb, 2)) * (height - 1);
        const int64 min_box_col =
            static_cast<float>(tboxes(b, bb, 1)) * (width - 1);
        const int64 max_box_col =
            static_cast<float>(tboxes(b, bb, 3)) * (width - 1);

        
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

        // Define the outer limits of the box (without thickness)
        const int64 top_outer = std::max<int64>(min_box_row, int64{0});
        const int64 bottom_outer = std::min<int64>(max_box_row, height - 1);
        const int64 left_outer = std::max<int64>(min_box_col, int64{0});
        const int64 right_outer = std::min<int64>(max_box_col, width - 1);

        // Sanity check for fitting the thickness in the box
        // Define a variable `line_width` for THIS box specifically
        const int64 half_dimension = std::min<int64>( 
            (right_outer - left_outer)/2, (bottom_outer - top_outer)/2 );
        const int64 line_width = std::max<int64>(half_dimension, thickness);

        // Define the inner limits of the box (with thickness)
        const int64 top_inner = static_cast<int64>(top_outer + line_width);
        const int64 bottom_inner = static_cast<int64>(bottom_outer - line_width);
        const int64 left_inner = static_cast<int64>(left_outer + line_width);
        const int64 right_inner = static_cast<int64>(right_outer - line_width);

        // At this point, {min,max}_box_{row,col}_clamp are inside the
        // image.
        CHECK_GE(top_outer, 0);
        CHECK_LT(top_outer, height);
        CHECK_GE(bottom_outer, 0);
        CHECK_LT(bottom_outer, height);
        CHECK_GE(left_outer, 0);
        CHECK_LT(left_outer, width);
        CHECK_GE(right_outer, 0);
        CHECK_LT(right_outer, width);

        // At this point, the min_box_row and min_box_col are either
        // in the image or above/left of it, and max_box_row and
        // max_box_col are either in the image or below/right or it.
        CHECK_LT(min_box_row, height);
        CHECK_GE(max_box_row, 0);
        CHECK_LT(min_box_col, width);
        CHECK_GE(max_box_col, 0);

        // Draw top line.
        for (int64 curr_row = top_outer; curr_row < top_inner; curr_row++ )
          for (int64 j = left_outer; j <= right_outer; ++j)
            for (int64 c = 0; c < depth; c++)
              canvas(b, curr_row, j, c) =
                static_cast<T>(color_table[color_index][c]);

        // Draw bottom line.
        for (int64 curr_row = bottom_outer; curr_row > bottom_inner; curr_row-- )
          for (int64 j = left_outer; j <= right_outer; ++j)
            for (int64 c = 0; c < depth; c++)
              canvas(b, curr_row, j, c) =
                static_cast<T>(color_table[color_index][c]);

        // Draw left line.
        for (int64 curr_col= left_outer; curr_col < left_inner; curr_col++ )
          for (int64 i = top_outer; i <= bottom_outer; ++i)
            for (int64 c = 0; c < depth; c++)
              canvas(b, i, curr_col, c) =
                static_cast<T>(color_table[color_index][c]);

        // Draw right line.
        for (int64 curr_col = right_outer; curr_col > right_inner; curr_col-- )
          for (int64 i = top_outer; i <= bottom_outer; ++i)
            for (int64 c = 0; c < depth; c++)
              canvas(b, i, curr_col, c) =
                static_cast<T>(color_table[color_index][c]);
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
      DrawBoundingBoxesOp<T>);                                               \
  REGISTER_KERNEL_BUILDER(                                                   \
      Name("DrawBoundingBoxesV3").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      DrawBoundingBoxesOp<T>);
TF_CALL_half(REGISTER_CPU_KERNEL);
TF_CALL_float(REGISTER_CPU_KERNEL);

}  // namespace tensorflow
