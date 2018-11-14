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
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// NOTE: the Tile implementation here is taken from tflite's Tile kernel.

template <typename T>
void CopyMultipleTimes(const T* in_data, int32_t in_size, int32_t multiplier,
                       T* out_data) {
  for (int i = 0; i < multiplier; ++i) {
    const T* in_end = in_data + in_size;
    T* new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const Shape& in_dimensions,
                                     const T* in_data, const M* multipliers,
                                     T* out_data, int dimension) {
  const int dimension_size = in_dimensions.dims(dimension);
  if (dimension == in_dimensions.dimensions_count() - 1) {
    CopyMultipleTimes(in_data, dimension_size, multipliers[dimension],
                      out_data);
    return std::make_pair(
        dimension_size,
        dimension_size * static_cast<int>(multipliers[dimension]));
  }
  int total_stride_size = 0, total_tiled_stride_size = 0;
  const T* copy_from_data = in_data;
  T* copy_to_data = out_data;
  for (int i = 0; i < dimension_size; ++i) {
    int stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
        TileOneDimension(in_dimensions, copy_from_data, multipliers,
                         copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size,
                    multipliers[dimension] - 1,
                    out_data + total_tiled_stride_size);
  return std::make_pair(total_stride_size,
                        total_tiled_stride_size * multipliers[dimension]);
}

template <ArrayDataType Type>
inline void Tile(const Array& input_array, const Array& multiples_array,
                 Array* output_array) {
  // Allocate output storage.
  auto& output_data = output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));

  switch (multiples_array.data_type) {
    case ArrayDataType::kInt32:
      TileOneDimension(
          input_array.shape(), input_array.GetBuffer<Type>().data.data(),
          multiples_array.GetBuffer<ArrayDataType::kInt32>().data.data(),
          output_array->GetMutableBuffer<Type>().data.data(), 0);
      break;
    case ArrayDataType::kInt64:
      TileOneDimension(
          input_array.shape(), input_array.GetBuffer<Type>().data.data(),
          multiples_array.GetBuffer<ArrayDataType::kInt64>().data.data(),
          output_array->GetMutableBuffer<Type>().data.data(), 0);
      break;
    default:
      CHECK(false);
      break;
  }
}

}  // namespace

// Resolves a constant Tile operation.
::tensorflow::Status ResolveConstantTile::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kTile) {
    return ::tensorflow::Status::OK();
  }
  const auto* op = static_cast<const TensorFlowTileOperator*>(base_op);

  CHECK_GE(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return ::tensorflow::Status::OK();
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return ::tensorflow::Status::OK();
  }

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return ::tensorflow::Status::OK();
  }
  const Array& input_array = model->GetArray(op->inputs[0]);
  const Array& multiples_array = model->GetArray(op->inputs[1]);
  CHECK(multiples_array.data_type == ArrayDataType::kInt32 ||
        multiples_array.data_type == ArrayDataType::kInt64)
      << "Only int32/int64 indices are supported";

  CopyMinMaxAndQuantizationRelatedFields(input_array, &output_array);

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Tile<ArrayDataType::kFloat>(input_array, multiples_array, &output_array);
      break;
    case ArrayDataType::kUint8:
      Tile<ArrayDataType::kUint8>(input_array, multiples_array, &output_array);
      break;
    case ArrayDataType::kInt16:
      Tile<ArrayDataType::kInt16>(input_array, multiples_array, &output_array);
      break;
    case ArrayDataType::kInt32:
      Tile<ArrayDataType::kInt32>(input_array, multiples_array, &output_array);
      break;
    case ArrayDataType::kInt64:
      Tile<ArrayDataType::kInt64>(input_array, multiples_array, &output_array);
      break;
    case ArrayDataType::kComplex64:
      Tile<ArrayDataType::kComplex64>(input_array, multiples_array,
                                      &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Tile op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  // Erase input arrays if no longer used after we remove the op.
  DeleteArrayIfUsedOnce(op->inputs[0], model);
  DeleteArrayIfUsedOnce(op->inputs[1], model);

  // Erase the operator.
  model->operators.erase(it);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
