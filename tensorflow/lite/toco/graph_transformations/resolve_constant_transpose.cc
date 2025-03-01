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
#include <cstddef>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

// Transposes an array up to rank 4.
// This is ShuffleArrayTemplate with non-enum permutation.
template <ArrayDataType Type>
void Transpose(Model* model, const Array& input_array,
               const std::vector<int>& perm, Array* output_array) {
  const Shape& input_shape = input_array.shape();
  const std::vector<DataType<Type>>& input_data =
      input_array.GetBuffer<Type>().data;

  const Shape& output_shape = output_array->shape();
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_shape));

  CHECK(input_shape.dimensions_count() == output_shape.dimensions_count());
  const int dim = input_shape.dimensions_count();
  CHECK_LE(dim, 4);
  CHECK(static_cast<int>(perm.size()) >= dim);
  for (int i = 0; i < dim; i++) {
    CHECK(perm[i] >= 0 && perm[i] < dim);
    CHECK(input_shape.dims(perm[i]) == output_shape.dims(i));
  }
  Shape extended_input_shape = input_shape;
  ExtendShape(&extended_input_shape, 4);
  Shape extended_output_shape = output_shape;
  ExtendShape(&extended_output_shape, 4);
  std::vector<int> extended_perm;
  ExtendShuffle(perm, 4, &extended_perm);

  const std::vector<int>& extended_input_dims = extended_input_shape.dims();
  const std::vector<int>& extended_output_dims = extended_output_shape.dims();

  // TODO(starka): Rework to handle different numbers of dimensions.
  int input_strides[4];
  input_strides[3] = 1;
  input_strides[2] = extended_input_dims[3];
  input_strides[1] = input_strides[2] * extended_input_dims[2];
  input_strides[0] = input_strides[1] * extended_input_dims[1];
  const int input_stride_0 = input_strides[extended_perm[3]];
  const int input_stride_1 = input_strides[extended_perm[2]];
  const int input_stride_2 = input_strides[extended_perm[1]];
  const int input_stride_3 = input_strides[extended_perm[0]];

  const int output_size_0 = extended_output_dims[3];
  const int output_size_1 = extended_output_dims[2];
  const int output_size_2 = extended_output_dims[1];
  const int output_size_3 = extended_output_dims[0];
  const int output_stride_0 = 1;
  const int output_stride_1 = output_size_0;
  const int output_stride_2 = output_stride_1 * output_size_1;
  const int output_stride_3 = output_stride_2 * output_size_2;

  for (int i3 = 0; i3 < output_size_3; i3++) {
    const DataType<Type>* const input_ptr_3 =
        input_data.data() + i3 * input_stride_3;
    DataType<Type>* const output_ptr_3 =
        output_data.data() + i3 * output_stride_3;
    for (int i2 = 0; i2 < output_size_2; i2++) {
      const DataType<Type>* const input_ptr_2 =
          input_ptr_3 + i2 * input_stride_2;
      DataType<Type>* const output_ptr_2 = output_ptr_3 + i2 * output_stride_2;
      for (int i1 = 0; i1 < output_size_1; i1++) {
        const DataType<Type>* input_ptr = input_ptr_2 + i1 * input_stride_1;
        DataType<Type>* output_ptr = output_ptr_2 + i1 * output_stride_1;
        DataType<Type>* const output_ptr_end =
            output_ptr + output_size_0 * output_stride_0;
        while (output_ptr != output_ptr_end) {
          *output_ptr = *input_ptr;
          input_ptr += input_stride_0;
          output_ptr += output_stride_0;
        }
      }
    }
  }
}

}  // namespace

absl::Status ResolveConstantTranspose::Run(Model* model, std::size_t op_index,
                                           bool* modified) {
  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kTranspose) {
    return absl::OkStatus();
  }
  const auto* op = static_cast<const TransposeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return absl::OkStatus();
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return absl::OkStatus();
  }

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return absl::OkStatus();
  }
  const Array& input_array = model->GetArray(op->inputs[0]);

  CopyMinMaxAndQuantizationRelatedFields(input_array, &output_array);

  if (op->perm.empty()) {
    // Yield until perm has been populated by ResolveTransposeAttributes.
    return absl::OkStatus();
  }

  // We currently only support 1-4 dimensions.
  CHECK_LE(op->perm.size(), 4);

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Transpose<ArrayDataType::kFloat>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kUint8:
      Transpose<ArrayDataType::kUint8>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kInt32:
      Transpose<ArrayDataType::kInt32>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kInt64:
      Transpose<ArrayDataType::kInt64>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kComplex64:
      Transpose<ArrayDataType::kComplex64>(model, input_array, op->perm,
                                           &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Transpose op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  AddMessageF("Resolving constant transpose of %s", LogName(*op));

  DeleteOpAndArrays(model, op);
  *modified = true;
  return absl::OkStatus();
}

}  // namespace toco
