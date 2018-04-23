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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// Gathers data from axis 0.
template <ArrayDataType Type>
inline void Gather(const Array& input_array, int input_rank,
                   const Array& coords_array, Array* output_array) {
  const Shape& input_shape = input_array.shape();
  const std::vector<DataType<Type>>& input_data =
      input_array.GetBuffer<Type>().data;
  const Shape& coords_shape = coords_array.shape();
  const std::vector<int32>& coords_data =
      coords_array.GetBuffer<ArrayDataType::kInt32>().data;

  const Shape& output_shape = output_array->shape();
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_shape));

  int rev_input_rank = input_shape.dimensions_count() - 1 - (input_rank - 1);
  CHECK_EQ(coords_shape.dims(0), output_array->shape().dims(rev_input_rank));

  int stride = 1;
  for (int i = input_shape.dimensions_count() - 1; i >= input_rank - 1; --i) {
    stride *= input_shape.dims(i);
  }

  for (int i = 0; i < coords_shape.dims(0); ++i) {
    DCHECK_GE(coords_data[i], 0);
    DCHECK_LT(coords_data[i], input_shape.dims(rev_input_rank));
    DataType<Type>* out = output_data.data() + i * stride;
    const DataType<Type>* in = input_data.data() + coords_data[i] * stride;
    memcpy(out, in, sizeof(DataType<Type>) * stride);
  }
}

}  // namespace

// Resolves a constant Gather operation.
// This simply performs the gather and produces the output array with the
// appropriate values.
bool ResolveConstantGather::Run(Model* model, std::size_t op_index) {
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kGather) {
    return false;
  }
  const auto* op = static_cast<const GatherOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return false;
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return false;
  }

  // Only handling axis=0 for now.
  if (op->axis != 0) {
    AddMessageF("%s has axis %d; only axis=0 is supported", LogName(*op),
                op->axis);
    return false;
  }

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return false;
  }
  const Array& input_array = model->GetArray(op->inputs[0]);
  const Array& coords_array = model->GetArray(op->inputs[1]);
  CHECK(coords_array.data_type == ArrayDataType::kInt32)
      << "Only int32 indices are supported";

  // Copy min/max info if present. The ranges of the selected values may be
  // a subset of the original range but we want to ensure the quantization
  // params stay the same.
  if (input_array.minmax) {
    const auto& input_minmax = input_array.GetMinMax();
    auto& output_minmax = output_array.GetOrCreateMinMax();
    output_minmax.min = input_minmax.min;
    output_minmax.max = input_minmax.max;
  }

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Gather<ArrayDataType::kFloat>(input_array, op->input_rank, coords_array,
                                    &output_array);
      break;
    case ArrayDataType::kUint8:
      Gather<ArrayDataType::kUint8>(input_array, op->input_rank, coords_array,
                                    &output_array);
      break;
    case ArrayDataType::kInt32:
      Gather<ArrayDataType::kInt32>(input_array, op->input_rank, coords_array,
                                    &output_array);
      break;
    case ArrayDataType::kInt64:
      Gather<ArrayDataType::kInt64>(input_array, op->input_rank, coords_array,
                                    &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Gather op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  // Erase input arrays if no longer used after we remove the op.
  DeleteArrayIfUsedOnce(op->inputs[0], model);
  DeleteArrayIfUsedOnce(op->inputs[1], model);

  // Erase the operator.
  model->operators.erase(it);
  return true;
}

}  // namespace toco
