/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <string>
#include <vector>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

// Sometimes, user may choose to use broadcast mul to perform nearest neighbor
// upsampling, how it works?
//
// If you want to upsample [batch, height, width, channel] to
// [batch, height * scale1, width * scale2, channel],
// you can do a broadcast mul like following:
// [batch, height, 1, width, 1 channel] with
// ones [1, 1, scale1, 1, scale2, channel].
//
// However, there's a 6-d tensor broadcasting mul operation which is not
// supported by the current runtime, also, it's less efficient for the quantized
// case if we can just do some memcpy.
// So we can transform this broadcast mul pattern to pack:
//
//  [batch, height, 1, width, 1, channel]
//                ||    Reshape
//               \/
//  [batch, height, width, channel]
//              ||   Pack at axis 3, with scale2
//             \/
//  [batch, height, width, scale2, channel]
//             ||    Pack at axis 2, with scale1
//            \/
//  [batch, height, scale1, width, scale2, channel]
namespace toco {
namespace {

void CopyShape(const std::vector<int>& new_shape, Array* array) {
  for (int dim : new_shape) {
    array->mutable_shape()->mutable_dims()->push_back(dim);
  }
}

template <ArrayDataType DataType, typename T>
bool HasSameValues(const Array& input_array, T value) {
  auto& buffer = input_array.GetBuffer<DataType>();
  for (int i = 0; i < buffer.Length(); ++i) {
    if (buffer.data.at(i) != value) {
      return false;
    }
  }
  return true;
}

std::vector<std::unique_ptr<Operator>>::iterator FindOperator(
    Model* model, const Operator& op) {
  return std::find_if(
      model->operators.begin(), model->operators.end(),
      [&op](const std::unique_ptr<Operator>& ptr) { return ptr.get() == &op; });
}

}  // namespace

// It's possible the model uses mul-broadcast to implement nearest neighbor
// upsample which may involve 5-d, 6-d tensors. We can actually change this
// pattern to be pack-based which is easier for us to handle.
::tensorflow::Status IdentifyNearestUpsample::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
  *modified = false;
  auto op_it = model->operators.begin() + op_index;
  auto* op = op_it->get();
  if (op->type != OperatorType::kMul) {
    return ::tensorflow::Status::OK();
  }

  // We only support one operand being constant.
  const std::string& lhs = op->inputs.at(0);
  const std::string& rhs = op->inputs.at(1);
  const std::string& output = op->outputs.at(0);

  Operator* next_op = GetOpWithOutput(*model, output);
  if (next_op == nullptr) {
    return ::tensorflow::Status::OK();
  }

  if (IsConstantParameterArray(*model, lhs) ==
      IsConstantParameterArray(*model, rhs)) {
    return ::tensorflow::Status::OK();
  }

  Array& const_array = IsConstantParameterArray(*model, lhs)
                           ? model->GetArray(lhs)
                           : model->GetArray(rhs);
  Array& nonconst_array = IsConstantParameterArray(*model, lhs)
                              ? model->GetArray(rhs)
                              : model->GetArray(lhs);
  Array& output_array = model->GetArray(output);

  // Wait for shape propogation finished.
  if (!const_array.has_shape() || !nonconst_array.has_shape() ||
      !output_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }

  // We need to make sure they have same dimension count & the const parameter
  // only contain ones.
  if (const_array.shape().dimensions_count() !=
      nonconst_array.shape().dimensions_count()) {
    return ::tensorflow::Status::OK();
  }

  if (const_array.data_type == ArrayDataType::kFloat) {
    if (!HasSameValues<ArrayDataType::kFloat, float>(const_array, 1))
      return ::tensorflow::Status::OK();
  } else if (const_array.data_type == ArrayDataType::kInt32) {
    if (!HasSameValues<ArrayDataType::kInt32, int>(const_array, 1))
      return ::tensorflow::Status::OK();
  } else if (const_array.data_type == ArrayDataType::kInt8) {
    if (!HasSameValues<ArrayDataType::kInt8, int8_t>(const_array, 127))
      return ::tensorflow::Status::OK();
  } else if (const_array.data_type == ArrayDataType::kUint8) {
    if (!HasSameValues<ArrayDataType::kUint8, uint8_t>(const_array, 255))
      return ::tensorflow::Status::OK();
  } else {
    return ::tensorflow::Status::OK();
  }

  // We're recognizing the following patterns:
  // non-const shape: [..., M, 1, ..., N, 1, ...]
  // const shape: [..., 1, scale_m, ..., 1, scale_n, ...]
  // and base on that, we're imaging the original shape, which is
  // [..., M, ... N, ...], then we're a serious of packing starting with the
  // lowest axis.

  // Scale_factors will store [scale_m, scale_n, ...].
  std::vector<int> scale_factors;
  // Pack_axis with store [axis_for_M + 1, axis_for_N + 1,...]
  std::vector<int> pack_axis;
  std::vector<int> imagined_original_shape;

  const auto& current_const_shape = const_array.shape();
  const auto& current_nonconst_shape = nonconst_array.shape();
  int imaged_original_shape_current_axis = 0;
  int i = 0;
  for (; i < current_const_shape.dimensions_count() - 1;
       ++i, ++imaged_original_shape_current_axis) {
    if (current_const_shape.dims(i) == 1 &&
        current_nonconst_shape.dims(i + 1) == 1 &&
        current_nonconst_shape.dims(i) != 1) {
      pack_axis.push_back(imaged_original_shape_current_axis + 1);
      scale_factors.push_back(current_const_shape.dims(i + 1));
      imagined_original_shape.push_back(current_nonconst_shape.dims(i));
      ++i;
    } else {
      imagined_original_shape.push_back(current_nonconst_shape.dims(i));
    }
  }
  // Push the rest dim.
  for (; i < current_nonconst_shape.dimensions_count(); ++i) {
    imagined_original_shape.push_back(current_nonconst_shape.dims(i));
  }

  if (pack_axis.empty()) {
    return ::tensorflow::Status::OK();
  }

  std::vector<Operator*> to_be_inserted_ops;

  // First put the reshape op.
  // This reshape op will reshape the input tensor to what we imagined as the
  // original shape.
  auto* reshape_op = new TensorFlowReshapeOperator;
  to_be_inserted_ops.push_back(reshape_op);
  const std::string& original_array_name =
      IsConstantParameterArray(*model, lhs) ? rhs : lhs;
  reshape_op->inputs.push_back(original_array_name);

  // Create shape param.
  const std::string shape_array_name = AvailableArrayName(*model, "new_shape");
  reshape_op->inputs.push_back(shape_array_name);
  Array& shape_array = model->GetOrCreateArray(shape_array_name);
  const int dim_size = imagined_original_shape.size();
  *(shape_array.mutable_shape()->mutable_dims()) = {dim_size};
  shape_array.data_type = ArrayDataType::kInt32;
  auto& shape_buffer = shape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  // This is what imagined as the original shape.
  for (size_t i = 0; i < imagined_original_shape.size(); ++i) {
    shape_buffer.data.push_back(imagined_original_shape.at(i));
  }

  const std::string& reshape_output_name =
      AvailableArrayName(*model, "reshape_output");
  Array& reshape_output_array = model->GetOrCreateArray(reshape_output_name);
  reshape_output_array.data_type = output_array.data_type;
  CopyShape(imagined_original_shape, &reshape_output_array);

  // Copy the quantization/minmax params if applicable.
  if (output_array.minmax) {
    reshape_output_array.GetOrCreateMinMax().min = output_array.minmax->min;
    reshape_output_array.GetOrCreateMinMax().max = output_array.minmax->max;
  }
  if (output_array.quantization_params) {
    reshape_output_array.GetOrCreateQuantizationParams().scale =
        output_array.quantization_params->scale;
    reshape_output_array.GetOrCreateQuantizationParams().zero_point =
        output_array.quantization_params->zero_point;
  }
  reshape_op->outputs.push_back(reshape_output_name);

  // Place the pack op as described in the file comment.
  std::string current_pack_input_name = reshape_output_name;
  std::vector<int> current_shape = imagined_original_shape;
  for (int i = pack_axis.size() - 1; i >= 0; --i) {
    auto* pack_op = new PackOperator;
    int scale = scale_factors.at(i);
    for (int j = 0; j < scale; ++j) {
      pack_op->inputs.push_back(current_pack_input_name);
    }
    // The axis is computed before, the values count is actually the scale.
    int axis = pack_axis.at(i);
    pack_op->axis = axis;
    pack_op->values_count = scale;
    const std::string& pack_output_array_name =
        AvailableArrayName(*model, absl::StrCat("pack_at_", axis));

    // We need to copy the quantization/minmax params if applicable.
    Array& pack_output_array = model->GetOrCreateArray(pack_output_array_name);
    if (output_array.minmax) {
      pack_output_array.GetOrCreateMinMax().min = output_array.minmax->min;
      pack_output_array.GetOrCreateMinMax().max = output_array.minmax->max;
    }
    if (output_array.quantization_params) {
      pack_output_array.GetOrCreateQuantizationParams().scale =
          output_array.quantization_params->scale;
      pack_output_array.GetOrCreateQuantizationParams().zero_point =
          output_array.quantization_params->zero_point;
    }

    pack_output_array.data_type = nonconst_array.data_type;
    current_shape.insert(current_shape.begin() + axis, scale);
    CopyShape(current_shape, &pack_output_array);
    pack_op->outputs.push_back(pack_output_array_name);

    // The output is actually the input to the next pack op.
    current_pack_input_name = pack_output_array_name;
    to_be_inserted_ops.push_back(pack_op);
  }

  // Rewire the final pack op.
  to_be_inserted_ops.at(to_be_inserted_ops.size() - 1)->outputs.clear();
  to_be_inserted_ops.at(to_be_inserted_ops.size() - 1)
      ->outputs.push_back(op->outputs.at(0));

  // Insert all added ops in a reverse order.
  for (int i = to_be_inserted_ops.size() - 1; i >= 0; --i) {
    op_it = model->operators.emplace(op_it, to_be_inserted_ops.at(i));
  }

  // Delete the mul op.
  model->operators.erase(FindOperator(model, *op));

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
