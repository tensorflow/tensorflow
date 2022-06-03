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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// Copies data from multiple source arrays to a destination array based on a
// concatenation dimension. From each array in input_arrays, it copies chunk
// sizes provided in array_copy_size vector (per array). It uses the buffer
// in concatenated_array as destination buffer.
template <ArrayDataType A, typename T>
void CopyTensorSegments(const std::vector<Array*>& input_arrays,
                        const std::vector<int>& array_copy_size,
                        const int num_elements_concatenated_array,
                        Array* concatenated_array) {
  for (Array* input_array : input_arrays) {
    if (!input_array->buffer) {
      return;
    }
  }

  auto& concatenated_array_buffer =
      concatenated_array->GetMutableBuffer<A>().data;
  concatenated_array_buffer.resize(num_elements_concatenated_array);

  // It does not matter which array to use to find the value for the total
  // number of copy steps.
  CHECK(!input_arrays.empty());
  CHECK_NE(array_copy_size[0], 0);
  const int total_copy_steps =
      input_arrays[0]->GetBuffer<A>().data.size() / array_copy_size[0];

  // Initialize the source pointers to point to beginning of the array buffers.
  std::vector<const T*> src_ptr;
  src_ptr.reserve(input_arrays.size());
  for (Array* input_array : input_arrays) {
    src_ptr.push_back(input_array->GetBuffer<A>().data.data());
  }

  // Copy the data from input_arrays to concatenated_array_buffer.
  T* dest_ptr = concatenated_array_buffer.data();
  for (int s = 0; s < total_copy_steps; s++) {
    for (size_t i = 0; i < input_arrays.size(); i++) {
      std::copy(src_ptr[i], src_ptr[i] + array_copy_size[i], dest_ptr);
      src_ptr[i] += array_copy_size[i];
      dest_ptr += array_copy_size[i];
    }
  }
}

// Receives a series of input arrays of type Array and an integer showing the
// axis on which those arrays will be concatenated. It returns the concatenated
// array.
template <ArrayDataType A>
void ConcatenateTensorBuffers(const std::vector<Array*>& input_arrays,
                              int concatenation_axis,
                              Array* concatenated_array) {
  int num_elements_concatenated_array = 1;
  for (int i = 0; i < concatenated_array->shape().dimensions_count(); i++) {
    num_elements_concatenated_array *= concatenated_array->shape().dims()[i];
  }
  // Prepare the data needed for segmented copy from multiple source arrays to
  // a destination array based on a oncatenation dimension.
  std::vector<int> array_copy_size(input_arrays.size());
  int count = 0;
  for (Array* input_array : input_arrays) {
    const Shape array_shape = input_array->shape();
    array_copy_size[count] = 1;
    for (int i = concatenation_axis; i < array_shape.dimensions_count(); i++) {
      array_copy_size[count] *= array_shape.dims()[i];
    }
    count++;
  }

  // Do the actual data copy.
  CopyTensorSegments<A, DataType<A>>(input_arrays, array_copy_size,
                                     num_elements_concatenated_array,
                                     concatenated_array);
}

// Sets the minimum and maximum values for the concatenated array. If it's
// already set (e.g. because of previous pass in TOCO), it doesn't change it and
// returns. Otherwise it uses the input arrays min and max values to compute the
// concatenated array min and max.
void SetMinMaxForConcatenedArray(GraphTransformation* transformation,
                                 const std::vector<Array*>& input_arrays,
                                 Array* concatenated_array) {
  CHECK(concatenated_array->data_type == ArrayDataType::kFloat);
  // If the minmax is already set, use it
  if (concatenated_array->minmax) return;

  double concat_min = std::numeric_limits<double>::infinity();
  double concat_max = -std::numeric_limits<double>::infinity();

  for (Array* input_array : input_arrays) {
    // If any of the input arrays minmax is not set,  return.
    // TODO(ghodrat): shall we add the logic to compute the minmax?
    if (!input_array->minmax) return;
    const MinMax& input_minmax = input_array->GetMinMax();
    concat_min = std::min(concat_min, input_minmax.min);
    concat_max = std::max(concat_max, input_minmax.max);
  }
  MinMax& minmax = concatenated_array->GetOrCreateMinMax();
  minmax.min = concat_min;
  minmax.max = concat_max;

  transformation->AddMessageF("Setting concatenated array min/max to %g,%g",
                              concat_min, concat_max);
}

}  // namespace

// Resolves the concatenation operator if all its inputs are constant arrays.
::tensorflow::Status ResolveConstantConcatenation::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
  *modified = false;
  const auto concat_it = model->operators.begin() + op_index;
  const auto* concat_base_op = concat_it->get();
  if (concat_base_op->type != OperatorType::kConcatenation) {
    return ::tensorflow::OkStatus();
  }
  const auto* concat_op =
      static_cast<const ConcatenationOperator*>(concat_base_op);

  for (const std::string& input_name : concat_op->inputs) {
    // We only expect constant unquantized arrays as input, otherwise we return.
    // We  also make sure the shapes of the input arrays are known and they are
    // all discardable.
    const Operator* input_op = GetOpWithOutput(*model, input_name);
    if (input_op) return ::tensorflow::OkStatus();
    if (!IsConstantParameterArray(*model, input_name))
      return ::tensorflow::OkStatus();
    if (!model->GetArray(input_name).has_shape())
      return ::tensorflow::OkStatus();
    if (model->GetArray(input_name).quantization_params)
      return ::tensorflow::OkStatus();
    if (!IsDiscardableArray(*model, input_name))
      return ::tensorflow::OkStatus();
  }

  const int concatenation_axis = concat_op->axis;

  CHECK_EQ(concat_op->outputs.size(), 1);
  std::string concatenated_array_name = concat_op->outputs[0];
  Array& concatenated_array = model->GetOrCreateArray(concatenated_array_name);
  std::vector<Array*> input_arrays;
  for (const std::string& input_name : concat_op->inputs) {
    input_arrays.push_back(&model->GetArray(input_name));
  }

  AddMessageF("Performing constant concat of %s into %s",
              absl::StrJoin(concat_op->inputs, ", "), concatenated_array_name);

  switch (concatenated_array.data_type) {
    case ArrayDataType::kFloat:
      ConcatenateTensorBuffers<ArrayDataType::kFloat>(
          input_arrays, concatenation_axis, &concatenated_array);
      SetMinMaxForConcatenedArray(this, input_arrays, &concatenated_array);
      break;
    case ArrayDataType::kUint8:
      ConcatenateTensorBuffers<ArrayDataType::kUint8>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kInt32:
      ConcatenateTensorBuffers<ArrayDataType::kInt32>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kInt64:
      ConcatenateTensorBuffers<ArrayDataType::kInt64>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kString:
      ConcatenateTensorBuffers<ArrayDataType::kString>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kComplex64:
      ConcatenateTensorBuffers<ArrayDataType::kComplex64>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    default:
      LOG(FATAL) << "ArrayDataType not supported";
  }

  DeleteOpAndArrays(model, concat_op);
  *modified = true;
  return ::tensorflow::OkStatus();
}

}  // namespace toco
