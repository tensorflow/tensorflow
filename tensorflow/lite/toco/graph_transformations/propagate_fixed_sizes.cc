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
#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void ComputeConvSizes(const Shape& input_shape, int output_depth, int kwidth,
                      int kheight, int stride_width, int stride_height,
                      int dilation_width_factor, int dilation_height_factor,
                      PaddingType padding_type, Shape* output_shape,
                      FixedPadding* fixed_padding) {
  const int input_width = input_shape.dims(2);
  const int input_height = input_shape.dims(1);
  const int batch = input_shape.dims(0);

  CHECK_GE(input_width, 1);
  CHECK_GE(input_height, 1);
  CHECK_GE(batch, 1);
  CHECK_GE(kwidth, 1);
  CHECK_GE(kheight, 1);
  CHECK_GE(stride_width, 1);
  CHECK_GE(stride_height, 1);
  CHECK_GE(dilation_width_factor, 1);
  CHECK_GE(dilation_height_factor, 1);

  int dilated_kwidth = dilation_width_factor * (kwidth - 1) + 1;
  int dilated_kheight = dilation_height_factor * (kheight - 1) + 1;

  int output_height = 0;
  int output_width = 0;
  if (padding_type == PaddingType::kValid) {
    output_height =
        (input_height + stride_height - dilated_kheight) / stride_height;
    output_width = (input_width + stride_width - dilated_kwidth) / stride_width;
  } else if (padding_type == PaddingType::kSame) {
    output_height = (input_height + stride_height - 1) / stride_height;
    output_width = (input_width + stride_width - 1) / stride_width;
  } else {
    LOG(FATAL) << "Only supporting SAME or VALID padding";
  }

  fixed_padding->height = std::max(0, ((output_height - 1) * stride_height +
                                       dilated_kheight - input_height) /
                                          2);
  fixed_padding->width = std::max(
      0,
      ((output_width - 1) * stride_width + dilated_kwidth - input_width) / 2);

  // Actually had to debug a situation where those were negative due to bad
  // propagation of placeholder -1 sizes in TensorFlowReshape.
  CHECK_GT(output_width, 0);
  CHECK_GT(output_height, 0);
  output_shape->ReplaceDims({batch, output_height, output_width, output_depth});
}

void ComputeBinaryOperatorOutputSize(const Shape& input_shape_x,
                                     const Shape& input_shape_y,
                                     Array* output_array) {
  // This matches the code in BroadcastBinaryOpShapeFn from tensorflow.
  // It zips together the two input shapes and pads with 1 to make them the
  // same length. For each dimension we broadcast if either dimension is 1 and
  // otherwise expect them to match.
  int rank_x = input_shape_x.dimensions_count();
  int rank_y = input_shape_y.dimensions_count();
  int rank_out = std::max(rank_x, rank_y);
  std::vector<int>* dims_out = output_array->mutable_shape()->mutable_dims();
  dims_out->clear();
  dims_out->reserve(rank_out);
  for (int i = 0; i < rank_out; ++i) {
    int dim_x = i < (rank_out - rank_x)
                    ? 1
                    : input_shape_x.dims(i - (rank_out - rank_x));
    bool dim_y_is_one = i < (rank_out - rank_y);
    int dim_y = dim_y_is_one ? 1 : input_shape_y.dims(i - (rank_out - rank_y));
    if (dim_x == -1 || dim_y == -1) {
      // One or both dimensions is unknown.
      QCHECK(false) << "Shapes must be specified";
    } else if (dim_x == 1 || dim_y == 1) {
      // Broadcast one dimension to the other that is 1.
      if (dim_x == 1 && !dim_y_is_one) {
        // Broadcast dim_y to dim_x (1).
        dims_out->push_back(dim_y);
      } else {
        // Broadcast dim_x to dim_y (1).
        DCHECK_EQ(dim_y, 1);
        dims_out->push_back(dim_x);
      }
    } else {
      // Expect the dimensions to match.
      CHECK_EQ(dim_x, dim_y) << "Dimensions must match";
      dims_out->push_back(dim_x);
    }
  }
  CHECK(output_array->has_shape());
}

void ProcessConvOperator(Model* model, ConvOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK(input_shape.dimensions_count() == 4)
      << "Conv ops require 4D inputs. Input array \"" << op->inputs[0]
      << "\" is " << input_shape.dimensions_count() << "D.";

  const auto& weights_array = model->GetArray(op->inputs[1]);
  // Yield until weights dims have been resolved.
  if (!weights_array.has_shape()) {
    return;
  }
  const auto& weights_shape = weights_array.shape();
  CHECK_EQ(weights_shape.dimensions_count(), 4);

  auto& output_array = model->GetArray(op->outputs[0]);
  const int output_depth = weights_shape.dims(0);
  const int kheight = weights_shape.dims(1);
  const int kwidth = weights_shape.dims(2);
  ComputeConvSizes(input_shape, output_depth, kwidth, kheight, op->stride_width,
                   op->stride_height, op->dilation_width_factor,
                   op->dilation_height_factor, op->padding.type,
                   output_array.mutable_shape(),
                   &op->padding.GetOrCreateFixedPadding());
  CHECK_EQ(output_array.shape().dimensions_count(), 4);

  // Set im2col array dimensions if there is one.
  if (op->outputs.size() == 2) {
    const auto& output_shape = output_array.shape();
    const int input_depth = weights_shape.dims(3);
    auto& im2col_array = model->GetArray(op->outputs[1]);
    im2col_array.copy_shape(Shape{output_shape.dims(0), output_shape.dims(1),
                                  output_shape.dims(2),
                                  input_depth * kheight * kwidth});
  }
}

void ProcessTransposeConvOperator(Model* model, TransposeConvOperator* op) {
  // TransposeConv is unique in that it is specifically given the output shape
  // as a 1D array on it's 1st input. Theoretically then, resolving the output
  // shape is as easy as waiting for this input to be resolved. However, we also
  // have to calculate the padding which requires the weights shape. So, we
  // might as well calculate the output shape and ensure it matches the
  // specified one

  // SPECIFIED OUTPUT SHAPE
  // The below is the specified, or prescribed output shape, _given_ to the
  // operator as an input.
  auto& specified_output_shape_array =
      model->GetArray(op->inputs[TransposeConvOperator::OUTPUT_SHAPE]);
  if (!specified_output_shape_array.has_shape() ||
      !specified_output_shape_array.buffer) {
    // Yield until the specified output shape is resolved as a constant
    return;
  }

  CHECK(specified_output_shape_array.data_type == ArrayDataType::kInt32)
      << "TransposeConv input_dims must be int32";

  CHECK(specified_output_shape_array.shape().dimensions_count() == 1 &&
        specified_output_shape_array.shape().dims(0) == 4)
      << "TransposeConv requires a 1D, 4 element array on it's 0th input "
         "specifying the output shape. \""
      << op->inputs[TransposeConvOperator::OUTPUT_SHAPE] << "\" had shape "
      << toco::ShapeToString(specified_output_shape_array.shape());

  // COMPUTE PADDING
  // We require the weights shape to calculate padding.
  const auto& weights_array =
      model->GetArray(op->inputs[TransposeConvOperator::WEIGHTS]);
  if (!weights_array.has_shape()) {
    // Yield until weights dims have been resolved.
    return;
  }
  const auto& weights_shape = weights_array.shape();
  CHECK_EQ(weights_shape.dimensions_count(), 4)
      << "TransposeConv weights must have 4 input dimensions. Input weights \""
      << op->inputs[TransposeConvOperator::WEIGHTS] << "\" had shape "
      << toco::ShapeToString(weights_shape) << ".";

  // Compute padding
  const int kheight = weights_shape.dims(1);
  const int kwidth = weights_shape.dims(2);
  op->padding.GetOrCreateFixedPadding();
  if (op->padding.type == PaddingType::kValid) {
    op->padding.fixed->height = 0;
    op->padding.fixed->width = 0;
  } else if (op->padding.type == PaddingType::kSame) {
    op->padding.fixed->height = (kheight - 1) / 2;
    op->padding.fixed->width = (kwidth - 1) / 2;
  } else {
    LOG(FATAL) << "TransposeConv only supports SAME or VALID padding";
  }

  // VALIDATE some dimensions and set the output shape.
  const auto& input_array =
      model->GetArray(op->inputs[TransposeConvOperator::DATA_INPUT]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4)
      << "TransposeConv input shape must have 4 dimensions. Input \""
      << op->inputs[TransposeConvOperator::WEIGHTS] << "\" had shape "
      << toco::ShapeToString(weights_shape) << ".";
  CHECK_EQ(input_shape.dims(3), weights_shape.dims(3))
      << "Input shape depth and weight depth do not agree";

  // Set the output shape according to the specified output shape.
  std::vector<int32> const& specified_output_shape =
      specified_output_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  auto& output_array = model->GetArray(op->outputs[0]);
  *(output_array.mutable_shape()->mutable_dims()) = specified_output_shape;

  // Set im2col array dimensions if there is one.
  if (op->outputs.size() == 2) {
    const int input_depth = weights_shape.dims(3);
    auto& im2col_array = model->GetArray(op->outputs[1]);
    im2col_array.copy_shape(
        Shape{specified_output_shape[0], specified_output_shape[1],
              specified_output_shape[2], input_depth * kheight * kwidth});
  }
}

void ProcessDepthwiseConvOperator(Model* model, DepthwiseConvOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);

  const auto& weights_array = model->GetArray(op->inputs[1]);
  // Yield until weights dims have been resolved.
  if (!weights_array.has_shape()) {
    return;
  }
  const auto& weights_shape = weights_array.shape();
  CHECK_EQ(weights_shape.dimensions_count(), 4);

  const string& output_name = op->outputs[0];
  const int input_depth = input_shape.dims(3);
  const int output_depth = weights_shape.dims(3);
  // TensorFlow doesn't define the depth_multiplier value on DepthwiseConv ops,
  // instead it has to be inferred from the weights dims. However, once we are
  // here, weights dims have already been converted to our own internal format,
  // where the multiplier is no longer readily apparent. So instead we get it
  // as the quotient of output and input depths. We only want to do that when
  // depth_multiplier had the zero value: any other value should be checked
  // as done by the next if() below.
  if (!op->depth_multiplier) {
    op->depth_multiplier = output_depth / input_depth;
  }
  CHECK_EQ(output_depth, input_depth * op->depth_multiplier)
      << "input/output depths and depth_multiplier don't match";

  const int kheight = weights_shape.dims(1);
  const int kwidth = weights_shape.dims(2);
  ComputeConvSizes(input_shape, output_depth, kwidth, kheight, op->stride_width,
                   op->stride_height, op->dilation_width_factor,
                   op->dilation_height_factor, op->padding.type,
                   model->GetArray(output_name).mutable_shape(),
                   &op->padding.GetOrCreateFixedPadding());
}

void ProcessDepthToSpaceOperator(Model* model, DepthToSpaceOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);

  const string& output_name = op->outputs[0];
  const int block_size = op->block_size;
  CHECK_NE(block_size, 0) << "Invalid block_size in " << output_name;
  const int batch = input_shape.dims(0);
  const int height = input_shape.dims(1);
  const int width = input_shape.dims(2);
  const int depth = input_shape.dims(3);
  QCHECK_EQ(depth % (block_size * block_size), 0);

  model->GetArray(output_name)
      .copy_shape(Shape({batch, height * block_size, width * block_size,
                         depth / block_size / block_size}));
}

void ProcessSpaceToDepthOperator(Model* model, SpaceToDepthOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);

  const string& output_name = op->outputs[0];
  const int block_size = op->block_size;
  CHECK_NE(block_size, 0) << "Invalid block_size in " << output_name;
  const int batch = input_shape.dims(0);
  const int height = input_shape.dims(1);
  const int width = input_shape.dims(2);
  const int depth = input_shape.dims(3);
  QCHECK_EQ(width % block_size, 0);
  QCHECK_EQ(height % block_size, 0);

  model->GetArray(output_name)
      .copy_shape(Shape({batch, height / block_size, width / block_size,
                         depth * block_size * block_size}));
}

void ProcessOpWithShapeInput(Model* model, Operator* op) {
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // We have already run
    return;
  }

  auto& dims_array = model->GetArray(op->inputs[0]);
  if (!dims_array.has_shape()) {
    // Yield until dims shape been resolved.
    return;
  }
  if (!dims_array.buffer) {
    // Yield until the dims are constant
    return;
  }
  CHECK(dims_array.data_type == ArrayDataType::kInt32) << "dims must be int32";
  CHECK_LE(RequiredBufferSizeForShape(dims_array.shape()), 4)
      << "dims vector can be no larger than 4 values";

  std::vector<int32> const& dims =
      dims_array.GetBuffer<ArrayDataType::kInt32>().data;
  *(output_array.mutable_shape()->mutable_dims()) = dims;
}

void ProcessFullyConnectedOperator(Model* model, FullyConnectedOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_GE(input_shape.dimensions_count(), 1);

  const auto& weights_array = model->GetArray(op->inputs[1]);
  // Yield until weights dims have been resolved.
  if (!weights_array.has_shape()) {
    return;
  }
  const auto& weights_shape = weights_array.shape();

  const int weights_output_depth = weights_shape.dims(0);
  CHECK_EQ(weights_shape.dimensions_count(), 2);

  const int input_overall_size = RequiredBufferSizeForShape(input_shape);
  const int matmul_repeats = input_overall_size / weights_shape.dims(1);
  CHECK_EQ(matmul_repeats * weights_shape.dims(1), input_overall_size);

  auto& output_array = model->GetArray(op->outputs[0]);
  output_array.copy_shape(Shape({matmul_repeats, weights_output_depth}));
}

void ProcessTensorFlowReshapeOperator(Model* model,
                                      TensorFlowReshapeOperator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // We have already run
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }
  const auto& input_shape = input_array.shape();

  auto& shape_array = model->GetArray(op->inputs[1]);
  if (!shape_array.has_shape()) {
    // Yield until target_shape shape been resolved.
    return;
  }
  if (!shape_array.buffer) {
    // Yield until the target_shape is constant
    return;
  }
  CHECK(shape_array.data_type == ArrayDataType::kInt32)
      << "Reshape dims must be int32";

  // shape_data is the raw array of ints describing the shape
  // in the TensorFlow node. We intentionally make a copy here, rather than
  // modify wildcards in-place below, because in some graphs, the same shape
  // array with a wildcard may be referenced from multiple Reshape nodes, where
  // the wildcard needs to resolved to distinct values.
  std::vector<int32> shape_data =
      shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  // The Reshape shape may have a wildcard dim, encoded as -1.
  bool has_wildcard = false;
  int wildcard_index = 0;
  int product_non_wildcard_dims = 1;
  for (int i = 0; i < shape_data.size(); i++) {
    if (shape_data[i] == -1) {
      CHECK(!has_wildcard);
      has_wildcard = true;
      wildcard_index = i;
    } else {
      product_non_wildcard_dims *= shape_data[i];
    }
  }

  const int input_flat_size = RequiredBufferSizeForShape(input_shape);
  if (has_wildcard) {
    CHECK_GE(input_flat_size, product_non_wildcard_dims)
        << "Array not large enough to fill the requested dimensions for "
           "Reshape op with output \""
        << op->outputs[0] << "\". Are your input shapes correct?";
    shape_data[wildcard_index] = input_flat_size / product_non_wildcard_dims;
  }

  if (shape_data.size() == 1 && shape_data[0] == 0) {
    // We have reshaped a scalar, so preserve as a scalar.
    shape_data.clear();
  }

  auto& output_shape = *output_array.mutable_shape();
  *output_shape.mutable_dims() = shape_data;
  CHECK_EQ(input_flat_size, RequiredBufferSizeForShape(output_shape))
      << "Input cannot be reshaped to requested dimensions for Reshape op with "
         "output \""
      << op->outputs[0] << "\". Are your input shapes correct?";
}

void ProcessSimpleOperator(Model* model, Operator* op, int input_index) {
  const auto& input_array = model->GetArray(op->inputs[input_index]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }

  const string& output_name = op->outputs[0];
  auto& output_array = model->GetArray(output_name);
  if (output_array.has_shape()) {
    return;
  }

  output_array.copy_shape(input_array.shape());
}

void ProcessSimpleBinaryOperator(Model* model, Operator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  const auto& input0_array = model->GetArray(op->inputs[0]);
  const auto& input1_array = model->GetArray(op->inputs[1]);
  // Yield until input dims have been resolved.
  if (!input0_array.has_shape() || !input1_array.has_shape()) {
    return;
  }
  const string& output_name = op->outputs[0];
  auto& output_array = model->GetArray(output_name);
  ComputeBinaryOperatorOutputSize(input0_array.shape(), input1_array.shape(),
                                  &output_array);
}

void ProcessSelectOperator(Model* model, SelectOperator* op) {
  // Yield until all input dims have been resolved.
  for (const auto& input : op->inputs) {
    const auto& input_array = model->GetArray(input);
    if (!input_array.has_shape()) {
      return;
    }
  }

  // Select's output matches the second and third output.
  const auto& input1_array = model->GetArray(op->inputs[1]);
  auto& output_array = model->GetArray(op->outputs[0]);
  output_array.copy_shape(input1_array.shape());
}

void ProcessAddNOperator(Model* model, Operator* op) {
  // Yield until all input dims have been resolved.
  //
  // TODO(myenik): Since AddN does not support broadcasting, maybe we could
  // actually use this to improve shape propagation by propagating the shape of
  // one input to all other inputs once it is resolved instead of just the
  // output, since all inputs must be the same size and shape for a well-formed
  // graph.
  for (const auto& input : op->inputs) {
    const auto& input_array = model->GetArray(input);
    if (!input_array.has_shape()) {
      return;
    }
  }

  // AddN does not support broadcasting, all inputs must be the same shape, so
  // we just take the first input shape and apply it to the output.
  const auto& input0_array = model->GetArray(op->inputs[0]);
  auto& output_array = model->GetArray(op->outputs[0]);
  output_array.copy_shape(input0_array.shape());
}

bool KeepDims(const Operator& op) {
  switch (op.type) {
    case OperatorType::kReduceMin:  //  Reduction Min
      return static_cast<const TensorFlowMinOperator&>(op).keep_dims;
    case OperatorType::kReduceMax:  //  Reduction Max
      return static_cast<const TensorFlowMaxOperator&>(op).keep_dims;
    case OperatorType::kSum:
      return static_cast<const TensorFlowSumOperator&>(op).keep_dims;
    case OperatorType::kReduceProd:
      return static_cast<const TensorFlowProdOperator&>(op).keep_dims;
    case OperatorType::kMean:
      return static_cast<const MeanOperator&>(op).keep_dims;
    case OperatorType::kAny:
      return static_cast<const TensorFlowAnyOperator&>(op).keep_dims;
    default:
      LOG(FATAL) << "Not a reduction operator!";
      return false;
  }
}

void ProcessTensorFlowReductionOperator(Model* model, Operator* op) {
  CHECK_LE(op->inputs.size(), 2);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    return;
  }
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const bool keep_dims = KeepDims(*op);
  if (op->inputs.size() == 2) {
    // There is a reduction_indices input.
    const auto& reduction_indices_array = model->GetArray(op->inputs[1]);
    if (!reduction_indices_array.buffer) {
      return;
    }
    CHECK(reduction_indices_array.buffer->type == ArrayDataType::kInt32);

    int input_rank = input_shape.dimensions_count();
    std::set<int32> true_indices;
    const auto& reduction_indices =
        reduction_indices_array.GetBuffer<ArrayDataType::kInt32>().data;
    for (int i = 0; i < reduction_indices.size(); ++i) {
      const int32 reduction_index = reduction_indices[i];
      if (reduction_index < -input_rank || reduction_index >= input_rank) {
        CHECK(false) << "Invalid reduction dimension " << reduction_index
                     << " for input with " << input_rank << " dimensions";
      }
      int32 wrapped_index = reduction_index;
      if (wrapped_index < 0) {
        wrapped_index += input_rank;
      }
      true_indices.insert(wrapped_index);
    }

    auto* mutable_dims = output_array.mutable_shape()->mutable_dims();
    mutable_dims->clear();
    for (int i = 0; i < input_rank; ++i) {
      if (true_indices.count(i) > 0) {
        if (keep_dims) {
          mutable_dims->emplace_back(1);
        }
      } else {
        mutable_dims->emplace_back(input_shape.dims(i));
      }
    }
  } else {
    // No reduction_indices means complete reduction to a single scalar.
    if (keep_dims) {
      output_array.copy_shape(input_shape);
    } else {
      output_array.copy_shape(Shape({}));
    }
  }
}

void ProcessSliceOperator(Model* model, SliceOperator* op) {
  CHECK_EQ(op->inputs.size(), 3);
  CHECK_EQ(op->outputs.size(), 1);

  // Yield until the Slice params have been resolved.
  if (op->begin.empty()) return;

  // Yield until input dims have been resolved.
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) return;
  const Shape& input_shape = input_array.shape();

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) return;

  CHECK_EQ(input_shape.dims().size(), op->size.size());
  CHECK_EQ(op->begin.size(), op->size.size());

  std::vector<int> output_dims;
  for (int i = 0; i < op->begin.size(); ++i) {
    int size = op->size[i];
    if (size == -1) {
      size = input_array.shape().dims(i) - op->begin[i];
    }
    output_dims.push_back(size);
  }

  *output_array.mutable_shape()->mutable_dims() = output_dims;
}

void ProcessReorderAxesOperator(Model* model, ReorderAxesOperator* op) {
  const string& input_name = op->inputs[0];
  const auto& input_array = model->GetArray(input_name);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const string& output_name = op->outputs[0];
  Shape* output_shape = model->GetArray(output_name).mutable_shape();
  ShuffleDims(input_shape, op->input_axes_order, op->output_axes_order,
              output_shape);
}

void ProcessConcatenationOperator(Model* model, ConcatenationOperator* op) {
  // Yield until input dims have been resolved.
  for (const auto& input_name : op->inputs) {
    auto& input_array = model->GetArray(input_name);
    if (!input_array.has_shape()) {
      return;
    }
  }
  auto& output_array = model->GetArray(op->outputs[0]);
  // Use first non-empty input as basis for output dimensions.
  for (const auto& input_name : op->inputs) {
    const auto& input_array = model->GetArray(input_name);
    if (input_array.shape().dimensions_count() > 0) {
      output_array.copy_shape(input_array.shape());
      // Negative axis means the count starts at the back of the dims().
      if (op->axis < 0) op->axis += input_array.shape().dims().size();
      break;
    }
  }
  // Determine the concat size, and enfore that all inputs have
  // the same dimensions count.
  int concat_size = 0;
  for (const auto& input_name : op->inputs) {
    auto& input_array = model->GetArray(input_name);
    CHECK(input_array.has_shape());
    if (input_array.shape().dimensions_count() == 0) {
      continue;
    }
    CHECK_EQ(input_array.shape().dimensions_count(),
             output_array.shape().dimensions_count());
    const std::vector<int>& input_dims = input_array.shape().dims();
    CHECK_LT(op->axis, input_dims.size());
    concat_size += input_dims[op->axis];
  }
  // Write out the concat_size on the output array shape.
  auto& output_shape = *output_array.mutable_shape();
  auto& output_dims = *output_shape.mutable_dims();
  CHECK_LT(op->axis, output_shape.dimensions_count());
  output_dims[op->axis] = concat_size;
}

void ProcessRangeOperator(Model* model, RangeOperator* op) {
  CHECK_EQ(op->inputs.size(), 3);
  const auto& start_array = model->GetArray(op->inputs[0]);
  if (!start_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }
  const auto& limit_array = model->GetArray(op->inputs[1]);
  if (!limit_array.has_shape()) {
    return;
  }
  const auto& delta_array = model->GetArray(op->inputs[2]);
  if (!delta_array.has_shape()) {
    return;
  }

  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    // Yield until inputs are constant.
    return;
  }
  if (!IsConstantParameterArray(*model, op->inputs[1])) {
    return;
  }
  if (!IsConstantParameterArray(*model, op->inputs[2])) {
    return;
  }

  const ArrayDataType& start_dtype = start_array.data_type;
  CHECK(start_dtype == ArrayDataType::kInt32 ||
        start_dtype == ArrayDataType::kFloat)
      << "Range op inputs must be int32 or float.";
  CHECK(limit_array.data_type == start_dtype)
      << "In Range op, limit tensor must have the same data type as start "
         "tensor.";
  CHECK(delta_array.data_type == start_dtype)
      << "In Range op, delta tensor must have the same data type as start "
         "tensor.";
  CHECK_EQ(RequiredBufferSizeForShape(start_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(limit_array.shape()), 1)
      << "Range op inputs must be scalar.";
  CHECK_EQ(RequiredBufferSizeForShape(delta_array.shape()), 1)
      << "Range op inputs must be scalar.";

  int size = 0;
  if (start_dtype == ArrayDataType::kInt32) {
    size = std::floor((limit_array.GetBuffer<ArrayDataType::kInt32>().data[0] -
                       start_array.GetBuffer<ArrayDataType::kInt32>().data[0]) /
                      delta_array.GetBuffer<ArrayDataType::kInt32>().data[0]);
  } else if (start_dtype == ArrayDataType::kFloat) {
    size = std::floor((limit_array.GetBuffer<ArrayDataType::kFloat>().data[0] -
                       start_array.GetBuffer<ArrayDataType::kFloat>().data[0]) /
                      delta_array.GetBuffer<ArrayDataType::kFloat>().data[0]);
  }

  // Only set the output shape. Contents are set by ResolveConstantRange.
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  Shape* output_shape = output_array.mutable_shape();
  output_shape->ReplaceDims({size});
}

void ProcessTensorFlowSplitOperator(Model* model, TensorFlowSplitOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  const string& input_name = op->inputs[1];
  const auto& input_array = model->GetArray(input_name);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const Shape& input_shape = input_array.shape();

  // Yield until axis is constant.
  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    return;
  }

  const auto& axis_array = model->GetArray(op->inputs[0]);

  // Yield until axis dims have been resolved.
  if (!axis_array.has_shape()) {
    return;
  }

  CHECK(axis_array.data_type == ArrayDataType::kInt32)
      << "Axis array must be int32.";
  CHECK_EQ(RequiredBufferSizeForShape(axis_array.shape()), 1)
      << "Axis array must be scalar.";

  int axis = axis_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  if (axis < 0) {
    axis += input_shape.dimensions_count();
  }

  const int split_dim = input_shape.dims(axis);
  CHECK_EQ(split_dim % op->num_split, 0);
  const int split_depth = split_dim / op->num_split;

  Shape output_shape = input_shape;
  (*output_shape.mutable_dims())[axis] = split_depth;

  CHECK_EQ(op->outputs.size(), op->num_split);
  for (const auto& output : op->outputs) {
    model->GetArray(output).copy_shape(output_shape);
  }
}

void ProcessTensorFlowSplitVOperator(Model* model,
                                     TensorFlowSplitVOperator* op) {
  CHECK_EQ(op->inputs.size(), 3);

  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const Shape& input_shape = input_array.shape();

  // Yield until size_splits is constant.
  if (!IsConstantParameterArray(*model, op->inputs[1])) {
    return;
  }
  const auto& size_array = model->GetArray(op->inputs[1]);
  // Yield until size_splits dims have been resolved.
  if (!size_array.has_shape()) {
    return;
  }
  const Shape& size_shape = size_array.shape();

  CHECK(size_array.data_type == ArrayDataType::kInt32 ||
        size_array.data_type == ArrayDataType::kInt64)
      << "size_splits must be int32, int64";
  CHECK_EQ(size_shape.dimensions_count(), 1) << "size_splits must be 1-D";

  std::vector<int64> size_splits_vector;
  if (size_array.data_type == ArrayDataType::kInt32) {
    for (const auto each_size :
         size_array.GetBuffer<ArrayDataType::kInt32>().data) {
      size_splits_vector.push_back(each_size);
    }
  } else {
    size_splits_vector = size_array.GetBuffer<ArrayDataType::kInt64>().data;
  }

  // Yield until axis is constant.
  if (!IsConstantParameterArray(*model, op->inputs[2])) {
    return;
  }
  const auto& axis_array = model->GetArray(op->inputs[2]);
  // Yield until axis dims have been resolved.
  if (!axis_array.has_shape()) {
    return;
  }

  CHECK(axis_array.data_type == ArrayDataType::kInt32)
      << "Axis array must be int32.";
  CHECK_EQ(RequiredBufferSizeForShape(axis_array.shape()), 1)
      << "Axis array must be scalar.";

  int axis = axis_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  if (axis < 0) {
    axis += input_shape.dimensions_count();
  }

  CHECK_EQ(op->num_split, size_splits_vector.size());

  int64_t minus_one_count = 0, size_splits_sum = 0;
  for (auto size : size_splits_vector) {
    if (size == -1) {
      ++minus_one_count;
    } else {
      size_splits_sum += size;
    }
  }

  const int input_size = input_shape.dims(axis);

  CHECK_LE(minus_one_count, 1) << "size_splits can contain at most one -1.";

  if (minus_one_count == 1) {
    CHECK_LE(size_splits_sum, input_size);
    auto iter =
        std::find(size_splits_vector.begin(), size_splits_vector.end(), -1);
    *iter = input_size - size_splits_sum;
  } else {
    CHECK_EQ(size_splits_sum, input_size);
  }

  CHECK_EQ(op->outputs.size(), op->num_split);

  for (int i = 0; i < op->outputs.size(); ++i) {
    const auto& output = op->outputs[i];
    Shape output_shape = input_shape;
    (*output_shape.mutable_dims())[axis] = size_splits_vector.at(i);
    model->GetArray(output).copy_shape(output_shape);
  }
}

void ProcessAveragePoolOperator(Model* model, AveragePoolOperator* op) {
  const string& input_name = op->inputs[0];
  const auto& input_array = model->GetArray(input_name);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);
  const string& output_name = op->outputs[0];
  const int output_depth = input_shape.dims(3);
  ComputeConvSizes(input_shape, output_depth, op->kwidth, op->kheight,
                   op->stride_width, op->stride_height, 1, 1, op->padding.type,
                   model->GetArray(output_name).mutable_shape(),
                   &op->padding.GetOrCreateFixedPadding());
}

void ProcessMaxPoolOperator(Model* model, MaxPoolOperator* op) {
  const string& input_name = op->inputs[0];
  const auto& input_array = model->GetArray(input_name);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);
  const string& output_name = op->outputs[0];
  const int output_depth = input_shape.dims(3);
  ComputeConvSizes(input_shape, output_depth, op->kwidth, op->kheight,
                   op->stride_width, op->stride_height, 1, 1, op->padding.type,
                   model->GetArray(output_name).mutable_shape(),
                   &op->padding.GetOrCreateFixedPadding());
}

void ProcessL2PoolOperator(Model* model, L2PoolOperator* op) {
  const string& input_name = op->inputs[0];
  const auto& input_array = model->GetArray(input_name);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  if (input_shape.dimensions_count() < 4) {
    LOG(FATAL) << "missing dimensions for " << input_name;
  }
  const string& output_name = op->outputs[0];
  const int output_depth = input_shape.dims(3);
  ComputeConvSizes(input_shape, output_depth, op->kwidth, op->kheight,
                   op->stride_width, op->stride_height, 1, 1, op->padding.type,
                   model->GetArray(output_name).mutable_shape(),
                   &op->padding.GetOrCreateFixedPadding());
}

void ProcessResizeBilinearOperator(Model* model, ResizeBilinearOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  if (!model->GetArray(op->inputs[0]).has_shape() ||
      !model->GetArray(op->inputs[1]).has_shape()) {
    return;
  }
  const auto& input_data_shape = model->GetArray(op->inputs[0]).shape();

  const string& output_size_name = op->inputs[1];
  const auto& output_size_array = model->GetArray(output_size_name);
  CHECK(output_size_array.data_type == ArrayDataType::kInt32);
  CHECK(output_size_array.has_shape());
  const auto& output_size_shape = output_size_array.shape();
  CHECK_EQ(output_size_shape.dimensions_count(), 1);
  CHECK_EQ(output_size_shape.dims(0), 2);
  if (!output_size_array.buffer) {
    return;
  }
  std::vector<int32> output_shape =
      output_size_array.GetBuffer<ArrayDataType::kInt32>().data;
  model->GetArray(op->outputs[0])
      .copy_shape(Shape({input_data_shape.dims(0), output_shape[0],
                         output_shape[1], input_data_shape.dims(3)}));
}

void ProcessResizeNearestNeighborOperator(Model* model,
                                          ResizeNearestNeighborOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  if (!model->GetArray(op->inputs[0]).has_shape() ||
      !model->GetArray(op->inputs[1]).has_shape()) {
    return;
  }
  const auto& input_data_shape = model->GetArray(op->inputs[0]).shape();

  const string& output_size_name = op->inputs[1];
  const auto& output_size_array = model->GetArray(output_size_name);
  CHECK(output_size_array.data_type == ArrayDataType::kInt32);
  CHECK(output_size_array.has_shape());
  const auto& output_size_shape = output_size_array.shape();
  CHECK_EQ(output_size_shape.dimensions_count(), 1);
  CHECK_EQ(output_size_shape.dims(0), 2);
  if (!output_size_array.buffer) {
    return;
  }
  std::vector<int32> output_shape =
      output_size_array.GetBuffer<ArrayDataType::kInt32>().data;
  model->GetArray(op->outputs[0])
      .copy_shape(Shape({input_data_shape.dims(0), output_shape[0],
                         output_shape[1], input_data_shape.dims(3)}));
}

void ProcessLstmCellOperator(Model* model, LstmCellOperator* op) {
  // Only required for compact LstmCell with default NUM_INPUTS of inputs.
  if (op->inputs.size() != LstmCellOperator::NUM_INPUTS) return;

  const auto& input_array =
      model->GetArray(op->inputs[LstmCellOperator::DATA_INPUT]);
  // Yield until all input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_GE(input_shape.dimensions_count(), 2);

  const auto& prev_activ_array =
      model->GetArray(op->inputs[LstmCellOperator::PREV_ACTIV_INPUT]);
  // Yield until all input dims have been resolved.
  if (!prev_activ_array.has_shape()) {
    return;
  }
  const auto& prev_activ_shape = prev_activ_array.shape();
  CHECK_GE(prev_activ_shape.dimensions_count(), 2);

  const auto& weights_array =
      model->GetArray(op->inputs[LstmCellOperator::WEIGHTS_INPUT]);
  // Yield until weights dims have been resolved.
  if (!weights_array.has_shape()) {
    return;
  }
  const auto& weights_shape = weights_array.shape();
  CHECK_EQ(weights_shape.dimensions_count(), 2);

  const auto& bias_array =
      model->GetArray(op->inputs[LstmCellOperator::BIASES_INPUT]);
  // Yield until bias dims have been resolved.
  if (!bias_array.has_shape()) {
    return;
  }
  const auto& bias_shape = bias_array.shape();
  CHECK_GE(bias_shape.dimensions_count(), 1);

  const auto& prev_state_array =
      model->GetArray(op->inputs[LstmCellOperator::PREV_STATE_INPUT]);
  // Yield until all input dims have been resolved.
  if (!prev_state_array.has_shape()) {
    return;
  }
  const auto& prev_state_shape = prev_state_array.shape();
  CHECK_GE(prev_state_shape.dimensions_count(), 2);

  const int fc_output_depth = weights_shape.dims(0);
  CHECK_EQ(fc_output_depth, bias_shape.dims(0));
  CHECK_EQ(fc_output_depth % 4, 0);
  const int depth = fc_output_depth / 4;

  const int input_depth = input_shape.dims(input_shape.dimensions_count() - 1);
  const int fc_input_depth = weights_shape.dims(1);
  CHECK_EQ(input_depth + depth, fc_input_depth);
  Shape output_shape(input_shape);
  (*output_shape.mutable_dims())[output_shape.dimensions_count() - 1] = depth;

  // Set output dimensions
  model->GetArray(op->outputs[LstmCellOperator::STATE_OUTPUT])
      .copy_shape(output_shape);
  model->GetArray(op->outputs[LstmCellOperator::ACTIV_OUTPUT])
      .copy_shape(output_shape);

  Shape concat_temp_shape(input_shape);
  (*concat_temp_shape
        .mutable_dims())[concat_temp_shape.dimensions_count() - 1] =
      fc_input_depth;
  model->GetArray(op->outputs[LstmCellOperator::CONCAT_TEMP])
      .copy_shape(concat_temp_shape);

  Shape activ_temp_shape(input_shape);
  (*activ_temp_shape.mutable_dims())[activ_temp_shape.dimensions_count() - 1] =
      fc_output_depth;
  model->GetArray(op->outputs[LstmCellOperator::ACTIV_TEMP])
      .copy_shape(activ_temp_shape);
}

void ProcessUnidirectionalSequenceLstmOperator(
    Model* model, UnidirectionalSequenceLstmOperator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  // TODO(renjieliu): check the inputs, as well as all kinds of weights.
  const auto& input_array = model->GetArray(op->inputs[0]);

  constexpr int kInputActivationStateTensor = 18;
  constexpr int kInputCellStateTensor = 19;

  // TFlite intepreter does not support array which is variable and contains a
  // buffer (see b/115961645 for more discussion).
  // The follow block remove buffer from the array to work around the
  // restriction, as a consequence, downstream applications should not
  // read lstm state as input to other operations.
  model->GetArray(op->inputs[kInputActivationStateTensor]).buffer.reset();
  model->GetArray(op->inputs[kInputCellStateTensor]).buffer.reset();

  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const int batch_size = input_shape.dims(1);
  const int timestamp = input_shape.dims(0);

  const auto& recurrent_to_output_weights_array =
      model->GetArray(op->inputs[8]);
  // Yield until input dims have been resolved.
  if (!recurrent_to_output_weights_array.has_shape()) {
    return;
  }

  const auto& output_weights_shape = recurrent_to_output_weights_array.shape();
  const int output_size = output_weights_shape.dims(1);

  Shape* output_shape = output_array.mutable_shape();
  output_shape->ReplaceDims({timestamp, batch_size, output_size});
}

void ProcessUnidirectionalSequenceRnnOperator(
    Model* model, UnidirectionalSequenceRnnOperator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated.
    return;
  }

  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  constexpr int kHiddenStateTensor = 4;
  // TFlite intepreter does not support array which is variable and contains a
  // buffer (see b/115961645 for more discussion).
  // The follow block remove buffer from the array to work around the
  // restriction, as a consequence, downstream applications should not
  // read lstm state as input to other operations.
  model->GetArray(op->inputs[kHiddenStateTensor]).buffer.reset();

  // TODO(renjieliu): check the inputs, as well as all kinds of weights.
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const int batch_size = input_shape.dims(1);
  const int timestamp = input_shape.dims(0);

  const auto& bias_array = model->GetArray(op->inputs[3]);
  // Yield until input dims have been resolved.
  if (!bias_array.has_shape()) {
    return;
  }

  const auto& bias_shape = bias_array.shape();
  const int output_size = bias_shape.dims(0);

  Shape* output_shape = output_array.mutable_shape();
  output_shape->ReplaceDims({timestamp, batch_size, output_size});
}

void ProcessBidirectionalSequenceLstmOperator(
    Model* model, BidirectionalSequenceLstmOperator* op) {
  // We assume time major.
  auto& fw_output_array = model->GetArray(op->outputs[0]);
  auto& bw_output_array = model->GetArray(op->outputs[1]);
  if (fw_output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (fw_output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  // TODO(renjieliu): check the inputs, as well as all kinds of weights.
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const int batch_size = input_shape.dims(1);
  const int timestamp = input_shape.dims(0);

  constexpr int kBwRecurrentToOutputWeightsTensor = 25;
  const auto& recurrent_to_output_weights_array =
      model->GetArray(op->inputs[kBwRecurrentToOutputWeightsTensor]);
  // Yield until input dims have been resolved.
  if (!recurrent_to_output_weights_array.has_shape()) {
    return;
  }

  constexpr int kFwInputActivationStateTensor = 35;
  constexpr int kFwInputCellStateTensor = 36;
  constexpr int kBwInputActivationStateTensor = 37;
  constexpr int kBwInputCellStateTensor = 38;
  // b(115961645): This is a hack to work around.
  model->GetArray(op->inputs[kFwInputActivationStateTensor]).buffer.reset();
  model->GetArray(op->inputs[kFwInputCellStateTensor]).buffer.reset();
  model->GetArray(op->inputs[kBwInputActivationStateTensor]).buffer.reset();
  model->GetArray(op->inputs[kBwInputCellStateTensor]).buffer.reset();

  const auto& output_weights_shape = recurrent_to_output_weights_array.shape();
  const int output_size = output_weights_shape.dims(1);

  Shape* fw_output_shape = fw_output_array.mutable_shape();
  if (op->merge_outputs) {
    fw_output_shape->ReplaceDims({timestamp, batch_size, 2 * output_size});
  } else {
    fw_output_shape->ReplaceDims({timestamp, batch_size, output_size});
    Shape* bw_output_shape = bw_output_array.mutable_shape();
    bw_output_shape->ReplaceDims({timestamp, batch_size, output_size});
  }
}

void ProcessBidirectionalSequenceRnnOperator(
    Model* model, BidirectionalSequenceRnnOperator* op) {
  // We assume time major.
  auto& fw_output_array = model->GetArray(op->outputs[0]);
  auto& bw_output_array = model->GetArray(op->outputs[1]);
  if (fw_output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (fw_output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  // TODO(renjieliu): check the inputs, as well as all kinds of weights.
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  const int batch_size = input_shape.dims(1);
  const int timestamp = input_shape.dims(0);

  constexpr int kFwWeightsTensor = 1;
  const auto& forward_weights_array =
      model->GetArray(op->inputs[kFwWeightsTensor]);
  // Yield until input dims have been resolved.
  if (!forward_weights_array.has_shape()) {
    return;
  }

  constexpr int kFwHiddenStateTensor = 4;
  constexpr int kBwHiddenStateTensor = 8;
  // b(115961645): This is a hack to work around.
  model->GetArray(op->inputs[kFwHiddenStateTensor]).buffer.reset();
  model->GetArray(op->inputs[kBwHiddenStateTensor]).buffer.reset();

  const auto& output_weights_shape = forward_weights_array.shape();
  const int output_size = output_weights_shape.dims(0);

  Shape* fw_output_shape = fw_output_array.mutable_shape();
  if (op->merge_outputs) {
    fw_output_shape->ReplaceDims({timestamp, batch_size, 2 * output_size});
  } else {
    fw_output_shape->ReplaceDims({timestamp, batch_size, output_size});
    Shape* bw_output_shape = bw_output_array.mutable_shape();
    bw_output_shape->ReplaceDims({timestamp, batch_size, output_size});
  }
}

void ProcessSpaceToBatchNDOperator(Model* model, SpaceToBatchNDOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  // This method only handles input dimensions of 4.
  if (input_shape.dimensions_count() != 4) {
    return;
  }
  const auto input_height = input_shape.dims(1);
  const auto input_width = input_shape.dims(2);

  const auto& block_shape_array = model->GetArray(op->inputs[1]);
  const auto& paddings_array = model->GetArray(op->inputs[2]);
  const auto& block_shape_array_shape = block_shape_array.shape();
  const auto& paddings_array_shape = paddings_array.shape();
  QCHECK_EQ(block_shape_array_shape.dimensions_count(), 1);
  QCHECK_EQ(paddings_array_shape.dimensions_count(), 2);

  // We only support two dimensions.
  QCHECK_EQ(block_shape_array_shape.dims(0), 2);
  if (!block_shape_array.buffer) {
    return;
  }
  QCHECK(block_shape_array.data_type == ArrayDataType::kInt32);
  const auto& block_shape_data =
      block_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  auto block_height = block_shape_data[0];
  auto block_width = block_shape_data[1];

  QCHECK_EQ(paddings_array_shape.dims(0), 2);  // Number of block dimensions
  QCHECK_EQ(paddings_array_shape.dims(1), 2);  // Two parameters per dimension.
  if (!paddings_array.buffer) {
    return;
  }
  QCHECK(paddings_array.data_type == ArrayDataType::kInt32);
  const auto& paddings_data =
      paddings_array.GetBuffer<ArrayDataType::kInt32>().data;
  int height_with_paddings = input_height + paddings_data[0] + paddings_data[1];
  int width_with_paddings = input_width + paddings_data[2] + paddings_data[3];
  QCHECK_EQ(height_with_paddings % block_height, 0);
  QCHECK_EQ(width_with_paddings % block_width, 0);
  int output_height = height_with_paddings / block_height;
  int output_width = width_with_paddings / block_width;

  model->GetArray(op->outputs[0])
      .copy_shape(Shape({input_shape.dims(0) * block_height * block_width,
                         output_height, output_width, input_shape.dims(3)}));
}

void ProcessBatchToSpaceNDOperator(Model* model, BatchToSpaceNDOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }
  const auto& input_shape = input_array.shape();
  CHECK_EQ(input_shape.dimensions_count(), 4);
  const auto input_height = input_shape.dims(1);
  const auto input_width = input_shape.dims(2);

  const auto& block_shape_array = model->GetArray(op->inputs[1]);
  const auto& crops_array = model->GetArray(op->inputs[2]);
  const auto& block_shape_array_shape = block_shape_array.shape();
  const auto& crops_array_shape = crops_array.shape();
  QCHECK_EQ(block_shape_array_shape.dimensions_count(), 1);
  QCHECK_EQ(crops_array_shape.dimensions_count(), 2);

  // We only support two dimensions.
  QCHECK_EQ(block_shape_array_shape.dims(0), 2);
  if (!block_shape_array.buffer) {
    return;
  }
  QCHECK(block_shape_array.data_type == ArrayDataType::kInt32);
  const auto& block_shape_data =
      block_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  auto block_height = block_shape_data[0];
  auto block_width = block_shape_data[1];

  QCHECK_EQ(crops_array_shape.dims(0), 2);  // Number of block dimensions
  QCHECK_EQ(crops_array_shape.dims(1), 2);  // Two parameters per dimension.
  if (!crops_array.buffer) {
    return;
  }
  QCHECK(crops_array.data_type == ArrayDataType::kInt32);
  const auto& crops_data = crops_array.GetBuffer<ArrayDataType::kInt32>().data;
  const int crops_top = crops_data[0];
  const int crops_bottom = crops_data[1];
  const int crops_left = crops_data[2];
  const int crops_right = crops_data[3];
  const int output_height =
      input_height * block_height - crops_top - crops_bottom;
  const int output_width = input_width * block_width - crops_left - crops_right;
  QCHECK_EQ(input_shape.dims(0) % (block_height * block_width), 0);

  model->GetArray(op->outputs[0])
      .copy_shape(Shape({input_shape.dims(0) / (block_height * block_width),
                         output_height, output_width, input_shape.dims(3)}));
}

void ProcessGatherOperator(Model* model, GatherOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  const auto& indices_array = model->GetArray(op->inputs[1]);
  auto& output_array = model->GetArray(op->outputs[0]);

  // Bail if we already know the output shape.
  if (output_array.has_shape()) {
    return;
  }

  // Yield until input dims have been resolved.
  if (!input_array.has_shape() || !indices_array.has_shape()) {
    return;
  }

  // Yield until the axis has been resolved.
  if (!op->axis) {
    return;
  }
  int axis = op->axis.value();

  const auto& input_shape = input_array.shape();
  const auto& indices_shape = indices_array.shape();
  QCHECK_GE(input_shape.dimensions_count(), 1);
  op->input_rank = input_shape.dimensions_count();
  QCHECK_LT(axis, op->input_rank);

  // Copy the input dimensions to the output except for the axis dimensions
  // where the dimension of indices_shape is used.
  auto output_dims = output_array.mutable_shape()->mutable_dims();
  for (int dim = 0; dim < axis; ++dim) {
    output_dims->push_back(input_shape.dims(dim));
  }
  for (int dim = 0; dim < indices_shape.dimensions_count(); ++dim) {
    output_dims->push_back(indices_shape.dims(dim));
  }
  for (int dim = axis + 1; dim < input_shape.dimensions_count(); ++dim) {
    output_dims->push_back(input_shape.dims(dim));
  }
}

void ProcessGatherNdOperator(Model* model, GatherNdOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  const auto& indices_array = model->GetArray(op->inputs[1]);
  auto& output_array = model->GetArray(op->outputs[0]);

  // Bail if we already know the output shape.
  if (output_array.has_shape()) {
    return;
  }

  // Yield until input dims have been resolved.
  if (!input_array.has_shape() || !indices_array.has_shape()) {
    return;
  }

  const auto& input_shape = input_array.shape();
  const auto& indices_shape = indices_array.shape();
  QCHECK_GE(input_shape.dimensions_count(), 1);
  QCHECK_GE(indices_shape.dimensions_count(), 1);
  const int indices_nd =
      indices_shape.dims(indices_shape.dimensions_count() - 1);
  QCHECK_LE(indices_nd, input_shape.dimensions_count());

  auto output_dims = output_array.mutable_shape()->mutable_dims();
  for (int dim = 0; dim < indices_shape.dimensions_count() - 1; ++dim) {
    output_dims->push_back(indices_shape.dims(dim));
  }
  for (int dim = indices_nd; dim < input_shape.dimensions_count(); ++dim) {
    output_dims->push_back(input_shape.dims(dim));
  }
}

void ProcessTopkV2Operator(Model* model, TopKV2Operator* op) {
  const auto& input_values = model->GetArray(op->inputs[0]);
  const auto& input_k = model->GetArray(op->inputs[1]);
  auto& output_values = model->GetArray(op->outputs[0]);
  auto& output_indexes = model->GetArray(op->outputs[1]);

  // Bail if we already know the output shape.
  if (output_indexes.has_shape()) {
    QCHECK(output_values.has_shape());
    return;
  }

  // Yield until input dims have been resolved.
  if (!input_values.has_shape() || !input_k.has_shape()) {
    return;
  }

  // If the value is initialized, we can specify the last dimension, otherwise
  // unknown.
  if (input_k.buffer) {
    const auto& input_values_shape = input_values.shape();
    auto output_indexes_dims = output_indexes.mutable_shape()->mutable_dims();
    auto output_values_dims = output_values.mutable_shape()->mutable_dims();
    for (int dim = 0; dim < input_values_shape.dimensions_count() - 1; dim++) {
      output_indexes_dims->push_back(input_values_shape.dims(dim));
      output_values_dims->push_back(input_values_shape.dims(dim));
    }
    const int32_t k_value = input_k.GetBuffer<ArrayDataType::kInt32>().data[0];
    output_indexes_dims->push_back(k_value);
    output_values_dims->push_back(k_value);
  }
}

void ProcessPadOperator(Model* model, PadOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  const auto& input_array = model->GetArray(op->inputs[0]);

  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) return;

  if (op->left_padding.empty()) return;
  CHECK_EQ(op->left_padding.size(), op->right_padding.size());

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) return;

  Shape output_shape = input_array.shape();
  std::vector<int>& dims = *output_shape.mutable_dims();
  CHECK_EQ(op->left_padding.size(), dims.size());

  for (int i = 0; i < op->left_padding.size(); ++i) {
    dims[i] += op->left_padding[i] + op->right_padding[i];
  }

  output_array.copy_shape(output_shape);
}

void ProcessPadV2Operator(Model* model, PadV2Operator* op) {
  CHECK_EQ(op->inputs.size(), 3);
  CHECK_EQ(op->outputs.size(), 1);

  const auto& input_array = model->GetArray(op->inputs[0]);

  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) return;

  if (op->left_padding.empty()) return;
  CHECK_EQ(op->left_padding.size(), op->right_padding.size());

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) return;

  Shape output_shape = input_array.shape();
  std::vector<int>& dims = *output_shape.mutable_dims();
  CHECK_EQ(op->left_padding.size(), dims.size());

  for (int i = 0; i < op->left_padding.size(); ++i) {
    dims[i] += op->left_padding[i] + op->right_padding[i];
  }

  output_array.copy_shape(output_shape);
}

void ProcessRankOperator(Model* model, TensorFlowRankOperator* op) {
  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }

  // Only set the output shape. Array contents are set by
  // ResolveConstantShapeOrRank.
  Shape* output_shape = output_array.mutable_shape();
  output_shape->ReplaceDims({});
}

void ProcessShapeOperator(Model* model, TensorFlowShapeOperator* op) {
  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }

  // Only set the output shape. Array contents are set by
  // ResolveConstantShapeOrRank.
  Shape* output_shape = output_array.mutable_shape();
  output_shape->ReplaceDims({input_array.shape().dimensions_count()});
}

void ProcessPackOperator(Model* model, PackOperator* op) {
  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  std::unique_ptr<Shape> packed_shape;
  for (const auto& input : op->inputs) {
    const auto& input_array = model->GetArray(input);
    if (!input_array.has_shape()) {
      // Yield until all input dims have been resolved.
      return;
    }

    Shape shape = input_array.shape();
    if (!packed_shape) {
      packed_shape.reset(new Shape(shape));
    } else {
      CHECK(*packed_shape == shape) << "All input arrays to Pack operators "
                                       "must have the same shape. Input \""
                                    << input << "\" is different.";
    }
  }

  int axis = op->axis;
  if (axis < 0) {
    // Handle negative axis
    axis += packed_shape->dims().size() + 1;
  }
  packed_shape->mutable_dims()->insert(
      packed_shape->mutable_dims()->begin() + axis, op->inputs.size());
  output_array.copy_shape(*packed_shape);
}

void ProcessStridedSliceOperator(Model* model, StridedSliceOperator* op) {
  CHECK_GE(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  if (op->start_indices.empty() || op->stop_indices.empty() ||
      op->strides.empty()) {
    // ResolveStridedSliceAttributes has not run yet.
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }

  if (op->ellipsis_mask != 0) {
    // Something like LOG_FIRST_N(WARNING, 10) would be prefferable to reduce
    // log noise. However, the TensorFlow logging library does not appear to
    // support this.
    LOG(WARNING) << "Skipping StridedSlice op with output \"" << op->outputs[0]
                 << "\". ellipsis_mask is not supported (mask="
                 << op->ellipsis_mask << ")";
    return;
  }
  if (op->new_axis_mask != 0) {
    LOG(WARNING) << "Skipping StridedSlice op with output \"" << op->outputs[0]
                 << "\". new_axis_mask is not supported (mask="
                 << op->new_axis_mask << ")";
    return;
  }

  int num_input_axes = input_array.shape().dimensions_count();
  CHECK_LE(op->start_indices.size(), num_input_axes)
      << "StridedSlice op with output \"" << op->outputs[0]
      << "\", requires no more than " << num_input_axes << " start indices";
  CHECK_LE(op->stop_indices.size(), num_input_axes)
      << "StridedSlice op with output \"" << op->outputs[0]
      << "\", requires no more than " << num_input_axes << " stop indices";
  CHECK_LE(op->strides.size(), num_input_axes)
      << "StridedSlice op with output \"" << op->outputs[0]
      << "\", requires no more than " << num_input_axes << " strides";
  for (int i = 0; i < op->strides.size(); i++) {
    CHECK_NE(op->strides[i], 0) << "Strides must be non-zero. Axis " << i
                                << " has stride=" << op->strides[i] << ".";
  }

  // Create output shape
  std::vector<int>* dims = output_array.mutable_shape()->mutable_dims();

  // Compute output shape
  for (int axis = 0; axis < num_input_axes; ++axis) {
    const auto strided_slice_params =
        tflite::strided_slice::BuildStridedSliceParams(
            op->begin_mask, op->end_mask, op->shrink_axis_mask,
            op->start_indices, op->stop_indices, op->strides);
    int start_index = tflite::strided_slice::StartForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis);
    int stop_index = tflite::strided_slice::StopForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis,
        start_index);

    int dim_size = std::ceil(static_cast<float>(stop_index - start_index) /
                             op->strides[axis]);

    CHECK_GT(dim_size, 0)
        << "Output size for an axis must be greater than 0. Axis " << axis
        << " computes to size " << dim_size
        << " for StridedSlice op with output \"" << op->outputs[0] << "\".";
    if (op->shrink_axis_mask & (1 << axis)) {
      CHECK_EQ(dim_size, 1)
          << "Output size for an axis must compute to 1 when shrinking an "
             "axis. Axis "
          << axis << " computes to size " << dim_size
          << " for StridedSlice op with output \"" << op->outputs[0] << "\".";
    } else {
      dims->push_back(dim_size);
    }
  }
}

void ProcessSqueezeOperator(Model* model, SqueezeOperator* op) {
  CHECK_EQ(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);

  const auto& input_array = model->GetArray(op->inputs[0]);

  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) return;

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) return;

  const std::vector<int>& input_dims = input_array.shape().dims();
  std::vector<int> output_dims;

  std::vector<int> squeeze_dims;
  const int input_num_dims = input_dims.size();
  for (int i : op->squeeze_dims) {
    squeeze_dims.push_back(i < 0 ? i + input_num_dims : i);
  }
  for (int i = 0; i < input_num_dims; ++i) {
    if (input_dims[i] != 1 ||
        (!squeeze_dims.empty() &&
         std::find(squeeze_dims.begin(), squeeze_dims.end(), i) ==
             squeeze_dims.end())) {
      output_dims.push_back(input_dims[i]);
    }
  }
  *output_array.mutable_shape()->mutable_dims() = output_dims;
}

void ProcessSvdfOperator(Model* model, SvdfOperator* op) {
  CHECK(op->inputs.size() == 3 || op->inputs.size() == 4);
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) return;

  auto& weights_feature_array = model->GetArray(op->inputs[1]);
  if (!weights_feature_array.has_shape()) return;

  const auto& weights_time_array = model->GetArray(op->inputs[2]);
  if (!weights_time_array.has_shape()) return;

  const bool has_bias = (op->inputs.size() == 4);
  if (has_bias) {
    const auto& bias_array = model->GetArray(op->inputs[3]);
    if (!bias_array.has_shape()) return;
  }

  const int batch_size = input_array.shape().dims()[0];
  const int num_units = weights_feature_array.shape().dims()[0];
  const int memory_size = weights_time_array.shape().dims()[1];

  auto& state_array = model->GetArray(op->outputs[0]);
  state_array.mutable_shape()->ReplaceDims(
      {batch_size, memory_size * num_units});

  auto& output_array = model->GetArray(op->outputs[1]);
  output_array.mutable_shape()->ReplaceDims({batch_size, num_units});
}

void ProcessTransposeOperator(Model* model, TransposeOperator* op) {
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // We have already run
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }
  const auto& input_shape = input_array.shape();

  auto& perm_array = model->GetArray(op->inputs[1]);
  if (!perm_array.has_shape()) {
    // Yield until permutation shape been resolved.
    return;
  }
  if (!perm_array.buffer) {
    // Yield until the permutation is constant
    return;
  }
  CHECK(perm_array.data_type == ArrayDataType::kInt32)
      << "Transpose permutation input must be int32";

  std::vector<int32> const& perm =
      perm_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(perm.size(), input_shape.dimensions_count())
      << "Transpose permutation input " << op->inputs[1]
      << " must be same length as input dimensions";
  std::vector<int>* output_dims = output_array.mutable_shape()->mutable_dims();
  for (int i = 0; i < perm.size(); i++) {
    int axis = perm[i];
    CHECK_GE(axis, 0);
    CHECK_LT(axis, input_shape.dimensions_count());
    output_dims->push_back(input_shape.dims(axis));
  }
}

template <typename Op>
void ProcessArgMinMaxOperator(Model* model, Op* op) {
  CHECK_EQ(op->inputs.size(), 2);
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }

  const Array& axis_array = model->GetArray(op->inputs[1]);
  // Yield until input axis array shape has been resolved.
  if (!axis_array.has_shape()) {
    return;
  }

  const std::vector<int>& input_dims = input_array.shape().dims();

  CHECK(axis_array.data_type == ArrayDataType::kInt32 ||
        axis_array.data_type == ArrayDataType::kInt64)
      << "axis_array must be int32, int64";

  CHECK_EQ(RequiredBufferSizeForShape(axis_array.shape()), 1)
      << "Axis array must be scalar.";

  int64 axis;
  if (axis_array.data_type == ArrayDataType::kInt32) {
    axis = axis_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  } else {
    axis = axis_array.GetBuffer<ArrayDataType::kInt64>().data[0];
  }

  std::vector<int> output_dims;

  output_dims.reserve(input_dims.size() - 1);
  for (int i = 0; i < input_dims.size(); ++i) {
    if (i != axis) {
      output_dims.push_back(input_dims[i]);
    }
  }

  const string& output_name = op->outputs[0];
  auto& output_array = model->GetArray(output_name);
  if (output_array.has_shape()) {
    return;
  }
  *output_array.mutable_shape()->mutable_dims() = output_dims;
}

void ProcessSparseToDenseOperator(Model* model, SparseToDenseOperator* op) {
  CHECK_EQ(op->inputs.size(), 4);

  const Array& output_shape_array = model->GetArray(op->inputs[1]);
  if (!output_shape_array.has_shape()) return;
  CHECK_EQ(output_shape_array.shape().dimensions_count(), 1);

  // Output should not go over four dimensions.
  CHECK_LE(output_shape_array.shape().dims(0), 4);

  const string& output_name = op->outputs[0];
  Array& output_array = model->GetArray(output_name);
  if (output_array.has_shape()) return;

  CHECK(output_shape_array.data_type == ArrayDataType::kInt32 ||
        output_shape_array.data_type == ArrayDataType::kInt64);
  if (output_shape_array.data_type == ArrayDataType::kInt32) {
    *output_array.mutable_shape()->mutable_dims() =
        output_shape_array.GetBuffer<ArrayDataType::kInt32>().data;
  } else {
    const std::vector<int64>& output_shape_data =
        output_shape_array.GetBuffer<ArrayDataType::kInt64>().data;
    std::copy(
        output_shape_data.begin(), output_shape_data.end(),
        std::back_inserter(*output_array.mutable_shape()->mutable_dims()));
  }
}

void ProcessTileOperator(Model* model, TensorFlowTileOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);

  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // We have already run.
    return;
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until input dims have been resolved.
    return;
  }
  const auto& input_shape = input_array.shape();

  auto& multiples_array = model->GetArray(op->inputs[1]);
  if (!multiples_array.has_shape()) {
    // Yield until multiples shape been resolved.
    return;
  }
  if (!multiples_array.buffer) {
    // Yield until the multiples is constant.
    return;
  }
  CHECK(multiples_array.data_type == ArrayDataType::kInt32)
      << "Tile multiples input must be int32";

  std::vector<int32> const& multiples =
      multiples_array.GetBuffer<ArrayDataType::kInt32>().data;
  CHECK_EQ(multiples.size(), input_shape.dimensions_count())
      << "Tile multiples input " << op->inputs[1]
      << " must be same length as input dimensions";

  auto* mutable_dims = output_array.mutable_shape()->mutable_dims();
  mutable_dims->resize(multiples.size());
  for (int i = 0; i < mutable_dims->size(); ++i) {
    (*mutable_dims)[i] = input_shape.dims(i) * multiples[i];
  }
}

void ProcessOneHotOperator(Model* model, OneHotOperator* op) {
  CHECK_EQ(op->inputs.size(), 4);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.has_shape()) {
    // Shape already propagated
    return;
  }

  // Yield until indices dims have been resolved.
  const auto& indices_array =
      model->GetArray(op->inputs[OneHotOperator::INDICES_INPUT]);
  if (!indices_array.has_shape()) {
    return;
  }

  // Yield until depth is constant and dims have been resolved.
  if (!IsConstantParameterArray(*model,
                                op->inputs[OneHotOperator::DEPTH_INPUT])) {
    return;
  }
  const auto& depth_array =
      model->GetArray(op->inputs[OneHotOperator::DEPTH_INPUT]);
  if (!depth_array.has_shape()) {
    return;
  }

  CHECK(depth_array.data_type == ArrayDataType::kInt32)
      << "Depth array must be int32.";
  CHECK_EQ(RequiredBufferSizeForShape(depth_array.shape()), 1)
      << "Depth array must be scalar.";

  const int depth = depth_array.GetBuffer<ArrayDataType::kInt32>().data[0];
  CHECK_GE(depth, 0) << "Depth must be non-negative.";

  const int indices_dims = indices_array.shape().dimensions_count();
  const int output_dims = indices_dims + 1;
  const int axis = op->axis == -1 ? indices_dims : op->axis;
  CHECK_GE(axis, 0) << "Resolved axis must be non-negative.";

  auto* mutable_dims = output_array.mutable_shape()->mutable_dims();
  mutable_dims->resize(output_dims);
  for (int i = 0; i < output_dims; ++i) {
    int dim = 0;
    if (i < axis) {
      dim = indices_array.shape().dims(i);
    } else if (i == axis) {
      dim = depth;
    } else {
      dim = indices_array.shape().dims(i - 1);
    }
    (*mutable_dims)[i] = dim;
  }
}

void ProcessUnpackOperator(Model* model, UnpackOperator* op) {
  CHECK_EQ(op->inputs.size(), 1);
  const auto& input_array = model->GetArray(op->inputs[0]);
  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }

  const std::vector<int>& input_dims = input_array.shape().dims();
  std::vector<int> output_dims;

  output_dims.reserve(input_dims.size() - 1);
  for (int i = 0; i < input_dims.size(); ++i) {
    if (i != op->axis) {
      output_dims.push_back(input_dims[i]);
    }
  }
  for (const string& output_name : op->outputs) {
    auto& output_array = model->GetArray(output_name);
    if (output_array.has_shape()) {
      return;
    }
    *output_array.mutable_shape()->mutable_dims() = output_dims;
  }
}

void ProcessMirrorPadOperator(Model* model, MirrorPadOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  const auto& input_array = model->GetArray(op->inputs[0]);
  const auto& padding_matrix = model->GetArray(op->inputs[1]);

  // Yield until input dims have been resolved.
  if (!input_array.has_shape()) {
    return;
  }

  auto& output_array = model->GetArray(op->outputs[0]);
  // If output already computed or padding matrix is non
  // const then return.
  if (output_array.has_shape() ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return;
  }
  Shape output_shape = input_array.shape();
  std::vector<int>& dims = *output_shape.mutable_dims();

  std::vector<int64_t> padding;
  if (padding_matrix.data_type == ArrayDataType::kInt32) {
    const auto& data = padding_matrix.GetBuffer<ArrayDataType::kInt32>().data;
    for (auto elem : data) {
      padding.push_back(static_cast<int64_t>(elem));
    }
  } else if (padding_matrix.data_type == ArrayDataType::kInt64) {
    const auto& data = padding_matrix.GetBuffer<ArrayDataType::kInt64>().data;
    for (auto elem : data) {
      padding.push_back(elem);
    }
  } else {
    CHECK(padding_matrix.data_type == ArrayDataType::kInt64 ||
          padding_matrix.data_type == ArrayDataType::kInt32);
  }
  CHECK_EQ(padding_matrix.shape().dimensions_count(), 2);
  CHECK_EQ(input_array.shape().dimensions_count(),
           padding_matrix.shape().dims(0));
  for (int i = 0; i < input_array.shape().dimensions_count(); ++i) {
    dims[i] += padding[i * 2] + padding[i * 2 + 1];
  }

  output_array.copy_shape(output_shape);
}

void ProcessUniqueOperator(Model* model, UniqueOperator* op) {
  const auto& input_array = model->GetArray(op->inputs[0]);
  // We have 2 outputs, the shape of the index tensor, is the same size
  // as the input array. The unique values tensor, is unknown until runtime.
  CHECK_EQ(op->outputs.size(), 2);
  auto& idx_output_array = model->GetArray(op->outputs[1]);

  // Yield until input dims have been resolved, or output already computed
  if (!input_array.has_shape() || idx_output_array.has_shape()) {
    return;
  }
  idx_output_array.copy_shape(input_array.shape());
}

void ProcessMatrixDiagOperator(Model* model, MatrixDiagOperator* op) {
  CHECK_EQ(op->inputs.size(), 1);
  CHECK_EQ(op->outputs.size(), 1);
  auto& input_array = model->GetArray(op->inputs[0]);
  auto& output_array = model->GetArray(op->outputs[0]);
  // The input array must have a shape in order to proceed. Also,
  // bail out if the output shape has already been calculated.
  if (!input_array.has_shape() || output_array.has_shape()) {
    // We have already run
    return;
  }
  // Get the input_shape
  Shape* mutable_shape = input_array.mutable_shape();
  std::vector<int>* dims = mutable_shape->mutable_dims();
  int dims_size = dims->size();
  // Scalars are not allowed.
  CHECK_GT(dims_size, 0);
  int last_dim = (*dims)[dims_size - 1];
  dims->push_back(last_dim);
  output_array.copy_shape(*mutable_shape);
}

void ProcessMatrixSetDiagOperator(Model* model, MatrixSetDiagOperator* op) {
  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);
  auto& input_array = model->GetArray(op->inputs[0]);
  auto& output_array = model->GetArray(op->outputs[0]);
  // The shape of the input array must be known because that will
  // be the shape of the output array.
  if (!input_array.has_shape() || !output_array.has_shape()) {
    // We have already run
    return;
  }

  output_array.copy_shape(input_array.shape());
}

}  // namespace

::tensorflow::Status PropagateFixedSizes::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  std::unordered_map<string, std::vector<int>> old_output_dims;
  for (const auto& output : op->outputs) {
    if (model->GetArray(output).has_shape()) {
      old_output_dims[output] = model->GetArray(output).shape().dims();
    }
  }

  switch (op->type) {
    case OperatorType::kAbs:
    case OperatorType::kBatchNormalization:
    case OperatorType::kL2Normalization:
    case OperatorType::kDequantize:
    case OperatorType::kElu:
    case OperatorType::kRelu:
    case OperatorType::kRelu1:
    case OperatorType::kRelu6:
    case OperatorType::kPRelu:
    case OperatorType::kLeakyRelu:
    case OperatorType::kSoftmax:
    case OperatorType::kLogSoftmax:
    case OperatorType::kLog:
    case OperatorType::kLogistic:
    case OperatorType::kTanh:
    case OperatorType::kLocalResponseNormalization:
    case OperatorType::kIdentity:
    case OperatorType::kFakeQuant:
    case OperatorType::kNeg:
    case OperatorType::kRsqrt:
    case OperatorType::kSqrt:
    case OperatorType::kSquare:
    case OperatorType::kAll:
    case OperatorType::kAssert:
    case OperatorType::kCast:
    case OperatorType::kFloor:
    case OperatorType::kCeil:
    case OperatorType::kExp:
    case OperatorType::kSin:
    case OperatorType::kCos:
    case OperatorType::kLogicalAnd:
    case OperatorType::kLogicalNot:
    case OperatorType::kLogicalOr:
    case OperatorType::kZerosLike:
    case OperatorType::kReverseV2:
    case OperatorType::kReverseSequence:
      ProcessSimpleOperator(model, op, 0);
      break;
    case OperatorType::kGather:
      ProcessGatherOperator(model, static_cast<GatherOperator*>(op));
      break;
    case OperatorType::kGatherNd:
      ProcessGatherNdOperator(model, static_cast<GatherNdOperator*>(op));
      break;
    case OperatorType::kTopK_V2:
      ProcessTopkV2Operator(model, static_cast<TopKV2Operator*>(op));
      break;
    case OperatorType::kAdd:
    case OperatorType::kSub:
    case OperatorType::kMul:
    case OperatorType::kDiv:
    case OperatorType::kFloorDiv:
    case OperatorType::kFloorMod:
    case OperatorType::kLess:
    case OperatorType::kLessEqual:
    case OperatorType::kGreater:
    case OperatorType::kMaximum:  //  Element-wise Maximum
    case OperatorType::kMinimum:  //  Element-wise Minimum
    case OperatorType::kGreaterEqual:
    case OperatorType::kEqual:
    case OperatorType::kNotEqual:
    case OperatorType::kPow:
    case OperatorType::kSquaredDifference:
      ProcessSimpleBinaryOperator(model, op);
      break;
    case OperatorType::kAddN:
      ProcessAddNOperator(model, op);
      break;
    case OperatorType::kConv:
      ProcessConvOperator(model, static_cast<ConvOperator*>(op));
      break;
    case OperatorType::kTransposeConv:
      ProcessTransposeConvOperator(model,
                                   static_cast<TransposeConvOperator*>(op));
      break;
    case OperatorType::kDepthwiseConv:
      ProcessDepthwiseConvOperator(model,
                                   static_cast<DepthwiseConvOperator*>(op));
      break;
    case OperatorType::kDepthToSpace:
      ProcessDepthToSpaceOperator(model,
                                  static_cast<DepthToSpaceOperator*>(op));
      break;
    case OperatorType::kSpaceToDepth:
      ProcessSpaceToDepthOperator(model,
                                  static_cast<SpaceToDepthOperator*>(op));
      break;
    case OperatorType::kFill:
      CHECK_EQ(op->inputs.size(), 2);
      ProcessOpWithShapeInput(model, op);
      break;
    case OperatorType::kFullyConnected:
      ProcessFullyConnectedOperator(model,
                                    static_cast<FullyConnectedOperator*>(op));
      break;
    case OperatorType::kReshape:
      ProcessTensorFlowReshapeOperator(
          model, static_cast<TensorFlowReshapeOperator*>(op));
      break;
    case OperatorType::kAveragePool:
      ProcessAveragePoolOperator(model, static_cast<AveragePoolOperator*>(op));
      break;
    case OperatorType::kMaxPool:
      ProcessMaxPoolOperator(model, static_cast<MaxPoolOperator*>(op));
      break;
    case OperatorType::kL2Pool:
      ProcessL2PoolOperator(model, static_cast<L2PoolOperator*>(op));
      break;
    case OperatorType::kReduceMin:  //  Reduction Min
    case OperatorType::kReduceMax:  //  Reduction Max
    case OperatorType::kSum:
    case OperatorType::kReduceProd:
    case OperatorType::kMean:
    case OperatorType::kAny:
      ProcessTensorFlowReductionOperator(model, op);
      break;
    case OperatorType::kSelect:
      ProcessSelectOperator(model, static_cast<SelectOperator*>(op));
      break;
    case OperatorType::kSlice:
      ProcessSliceOperator(model, static_cast<SliceOperator*>(op));
      break;

    case OperatorType::kSwitch:
      // We can't know the sizes of the outputs until we have resolved the
      // predicate, and once we have resolved the predicate, the whole
      // Switch node will get resolved away.
      // See ResolveTensorFlowSwitch.
      break;
    case OperatorType::kMerge:
      // No need to bother resolving TensorFlow Merge ops: other graph
      // transformations will remove them anyway.
      // See ResolveTensorFlowMerge.
      break;
    case OperatorType::kSplit:
      ProcessTensorFlowSplitOperator(model,
                                     static_cast<TensorFlowSplitOperator*>(op));
      break;
    case OperatorType::kSplitV:
      ProcessTensorFlowSplitVOperator(
          model, static_cast<TensorFlowSplitVOperator*>(op));
      break;
    case OperatorType::kSqueeze:
      ProcessSqueezeOperator(model, static_cast<SqueezeOperator*>(op));
      break;
    case OperatorType::kConcat:
    case OperatorType::kConcatV2:
      // Unimplemented, hopefully another graph transformation will
      // drop it or rewrite it. Concretely, either ResolveTensorFlowConcat
      // will resolve this node to a DepthConcatenation, or else we have
      // a more general non-depth concatenation that will hopefully be dropped,
      // or else at the moment we will abort.
      break;
    case OperatorType::kExpandDims:
      // Yield until ExpandDims is converted to Reshape
      break;
    case OperatorType::kRange:
      ProcessRangeOperator(model, static_cast<RangeOperator*>(op));
      break;
    case OperatorType::kRank:
      ProcessRankOperator(model, static_cast<TensorFlowRankOperator*>(op));
      break;
    case OperatorType::kShape:
      ProcessShapeOperator(model, static_cast<TensorFlowShapeOperator*>(op));
      break;
    case OperatorType::kPack:
      ProcessPackOperator(model, static_cast<PackOperator*>(op));
      break;
    case OperatorType::kReorderAxes:
      ProcessReorderAxesOperator(model, static_cast<ReorderAxesOperator*>(op));
      break;
    case OperatorType::kConcatenation:
      ProcessConcatenationOperator(model,
                                   static_cast<ConcatenationOperator*>(op));
      break;
    case OperatorType::kResizeBilinear:
      ProcessResizeBilinearOperator(model,
                                    static_cast<ResizeBilinearOperator*>(op));
      break;
    case OperatorType::kResizeNearestNeighbor:
      ProcessResizeNearestNeighborOperator(
          model, static_cast<ResizeNearestNeighborOperator*>(op));
      break;
    case OperatorType::kUnidirectionalSequenceLstm:
      ProcessUnidirectionalSequenceLstmOperator(
          model, static_cast<UnidirectionalSequenceLstmOperator*>(op));
      break;
    case OperatorType::kUnidirectionalSequenceRnn:
      ProcessUnidirectionalSequenceRnnOperator(
          model, static_cast<UnidirectionalSequenceRnnOperator*>(op));
      break;
    case OperatorType::kBidirectionalSequenceLstm:
      ProcessBidirectionalSequenceLstmOperator(
          model, static_cast<BidirectionalSequenceLstmOperator*>(op));
      break;
    case OperatorType::kBidirectionalSequenceRnn:
      ProcessBidirectionalSequenceRnnOperator(
          model, static_cast<BidirectionalSequenceRnnOperator*>(op));
      break;
    case OperatorType::kLstmCell:
      ProcessLstmCellOperator(model, static_cast<LstmCellOperator*>(op));
      break;
    case OperatorType::kBatchMatMul:
    case OperatorType::kMatMul:
      // MatMul operators are converted to FullyConnected, after which their
      // shapes are propagated.
      break;
    case OperatorType::kSpaceToBatchND:
      ProcessSpaceToBatchNDOperator(model,
                                    static_cast<SpaceToBatchNDOperator*>(op));
      break;
    case OperatorType::kBatchToSpaceND:
      ProcessBatchToSpaceNDOperator(model,
                                    static_cast<BatchToSpaceNDOperator*>(op));
      break;
    case OperatorType::kPad:
      ProcessPadOperator(model, static_cast<PadOperator*>(op));
      break;
    case OperatorType::kPadV2:
      ProcessPadV2Operator(model, static_cast<PadV2Operator*>(op));
      break;
    case OperatorType::kStridedSlice:
      ProcessStridedSliceOperator(model,
                                  static_cast<StridedSliceOperator*>(op));
      break;
    case OperatorType::kArgMax:
      ProcessArgMinMaxOperator<ArgMaxOperator>(
          model, static_cast<ArgMaxOperator*>(op));
      break;
    case OperatorType::kArgMin:
      ProcessArgMinMaxOperator<ArgMinOperator>(
          model, static_cast<ArgMinOperator*>(op));
      break;
    case OperatorType::kUnsupported: {
      const auto* unsupported_op =
          static_cast<TensorFlowUnsupportedOperator*>(op);
      // Attribute can be not specified, ignore it.
      if (unsupported_op->output_shapes.size() < op->outputs.size()) {
        return ::tensorflow::Status::OK();
      }
      for (int i = 0; i < op->outputs.size(); ++i) {
        const string& output = op->outputs[i];
        model->GetArray(output).copy_shape(unsupported_op->output_shapes.at(i));
      }
      break;
    }
    case OperatorType::kSvdf:
      ProcessSvdfOperator(model, static_cast<SvdfOperator*>(op));
      break;
    case OperatorType::kTranspose:
      ProcessTransposeOperator(model, static_cast<TransposeOperator*>(op));
      break;
    case OperatorType::kDynamicPartition:
    case OperatorType::kDynamicStitch:
      // DynamicPartition/DynamicStitch are currently only supported for
      // transforms that remove them, so we avoid propagating shapes through
      // them and let things settle once they've been removed.
      break;
    case OperatorType::kRandomUniform:
      CHECK_EQ(op->inputs.size(), 1);
      ProcessOpWithShapeInput(model, op);
      break;
    case OperatorType::kSparseToDense:
      ProcessSparseToDenseOperator(model,
                                   static_cast<SparseToDenseOperator*>(op));
      break;
    case OperatorType::kTile:
      ProcessTileOperator(model, static_cast<TensorFlowTileOperator*>(op));
      break;
      break;
    case OperatorType::kOneHot:
      ProcessOneHotOperator(model, static_cast<OneHotOperator*>(op));
      break;
    case OperatorType::kUnpack:
      ProcessUnpackOperator(model, static_cast<UnpackOperator*>(op));
      break;
    case OperatorType::kMirrorPad:
      ProcessMirrorPadOperator(model, static_cast<MirrorPadOperator*>(op));
      break;
    case OperatorType::kUnique:
      ProcessUniqueOperator(model, static_cast<UniqueOperator*>(op));
      break;
    case OperatorType::kWhere:
      // The size of the output can only be known after evaluating the cond
      // tensor. Ignore shape propagation here and defer that to the
      // interpreter.
      break;
    case OperatorType::kMatrixDiag:
      ProcessMatrixDiagOperator(model, static_cast<MatrixDiagOperator*>(op));
      break;
    case OperatorType::kMatrixSetDiag:
      ProcessMatrixSetDiagOperator(model,
                                   static_cast<MatrixSetDiagOperator*>(op));
      break;
    default:
      // Unimplemented, another graph transformation should drop it.
      LOG(FATAL) << "Unhandled operator type " << OperatorTypeName(op->type);
  }

  // Return true if any output dim changed, false if none changed.
  // Assumption: no transformation clears an output shape, they only add shapes.
  for (const auto& output : op->outputs) {
    if (model->GetArray(output).has_shape() &&
        (old_output_dims[output] != model->GetArray(output).shape().dims())) {
      AddMessageF("Set shape of %s to [%s]", output,
                  absl::StrJoin(model->GetArray(output).shape().dims(), ","));
      *modified = true;
      return ::tensorflow::Status::OK();
    }
  }
  return ::tensorflow::Status::OK();
}

}  // namespace toco
