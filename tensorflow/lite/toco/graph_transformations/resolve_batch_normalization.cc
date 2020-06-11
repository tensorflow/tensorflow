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

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ResolveBatchNormalization::Run(Model* model,
                                                    std::size_t op_index,
                                                    bool* modified) {
  *modified = false;
  auto bn_it = model->operators.begin() + op_index;
  if (bn_it->get()->type != OperatorType::kBatchNormalization) {
    return ::tensorflow::Status::OK();
  }
  const auto* bn_op =
      static_cast<const BatchNormalizationOperator*>(bn_it->get());

  auto& mean_array = model->GetArray(bn_op->inputs[1]);
  const auto& multiplier_array = model->GetArray(bn_op->inputs[2]);
  const auto& offset_array = model->GetArray(bn_op->inputs[3]);

  // This graph transformation needs to address constant buffers below, so
  // we need to exit early if these buffers don't exist yet (i.e. if the params
  // haven't yet been resolved as constants) and will process it once they have.
  if (!mean_array.buffer || !multiplier_array.buffer || !offset_array.buffer) {
    return ::tensorflow::Status::OK();
  }

  CHECK(IsConstantParameterArray(*model, bn_op->inputs[1]) &&
        IsConstantParameterArray(*model, bn_op->inputs[2]) &&
        IsConstantParameterArray(*model, bn_op->inputs[3]))
      << "Batch normalization resolution requires that mean, multiplier and "
         "offset arrays be constant.";

  // We should only have *float* BatchNormalizations... let's guard this
  // assumption by CHECK's.
  CHECK(mean_array.data_type == ArrayDataType::kFloat);
  CHECK(multiplier_array.data_type == ArrayDataType::kFloat);
  CHECK(offset_array.data_type == ArrayDataType::kFloat);

  // Create the new Mul, Add operators
  auto* mul_op = new MulOperator;
  auto* add_op = new AddOperator;
  const string mul_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_mul");
  const string add_name =
      AvailableArrayName(*model, bn_op->outputs[0] + "_add");
  const string mul_param_name = AvailableArrayName(*model, mul_name + "_param");
  const string add_param_name = AvailableArrayName(*model, add_name + "_param");
  mul_op->inputs = {bn_op->inputs[0], mul_param_name};
  mul_op->outputs = {mul_name};
  add_op->inputs = {mul_name, add_param_name};
  add_op->outputs = {bn_op->outputs[0]};
  AddMessageF("Splitting %s into %s and %s", LogName(*bn_op), LogName(*mul_op),
              LogName(*add_op));

  // Create the intermediate activation array (output of mul, input of add)
  auto& intermediate_array = model->GetOrCreateArray(mul_op->outputs[0]);
  intermediate_array.data_type = model->GetArray(bn_op->inputs[0]).data_type;

  // Insert the new operators in the graph
  auto add_it = model->operators.emplace(bn_it, add_op);
  auto mul_it = model->operators.emplace(add_it, mul_op);
  // update invalidated iterators.
  DCHECK_EQ(mul_it->get(), mul_op);
  add_it = mul_it + 1;
  DCHECK_EQ(add_it->get(), add_op);
  bn_it = add_it + 1;
  DCHECK_EQ(bn_it->get(), bn_op);

  // Create the new param arrays
  auto& mean_shape = *mean_array.mutable_shape();
  const auto& multiplier_shape = multiplier_array.shape();
  const auto& offset_shape = offset_array.shape();
  if (mean_shape.dims().empty()) {
    *mean_shape.mutable_dims() = multiplier_shape.dims();
    auto& data = mean_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
    CHECK_EQ(data.size(), 1);
    data.resize(RequiredBufferSizeForShape(mean_shape), data[0]);
  }
  CHECK(mean_shape.dims() == multiplier_shape.dims());
  CHECK(mean_shape.dims() == offset_shape.dims());
  const auto& param_shape = mean_shape;
  const int buffer_size = RequiredBufferSizeForShape(param_shape);
  auto& mul_param_array = model->GetOrCreateArray(mul_param_name);
  auto& add_param_array = model->GetOrCreateArray(add_param_name);
  DropMinMax(model, mul_param_name);
  DropMinMax(model, add_param_name);
  mul_param_array.copy_shape(param_shape);
  add_param_array.copy_shape(param_shape);
  mul_param_array.data_type = ArrayDataType::kFloat;
  add_param_array.data_type = ArrayDataType::kFloat;
  auto& mul_float_data =
      mul_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  auto& add_float_data =
      add_param_array.GetMutableBuffer<ArrayDataType::kFloat>().data;
  mul_float_data.resize(buffer_size);
  add_float_data.resize(buffer_size);
  const auto& mean_float_data =
      mean_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& multiplier_float_data =
      multiplier_array.GetBuffer<ArrayDataType::kFloat>().data;
  const auto& offset_float_data =
      offset_array.GetBuffer<ArrayDataType::kFloat>().data;

  CHECK(static_cast<int>(mul_float_data.size()) == buffer_size);
  CHECK(static_cast<int>(add_float_data.size()) == buffer_size);
  CHECK(static_cast<int>(mean_float_data.size()) == buffer_size);
  CHECK(static_cast<int>(multiplier_float_data.size()) == buffer_size);
  CHECK(static_cast<int>(offset_float_data.size()) == buffer_size);

  for (int i = 0; i < buffer_size; i++) {
    mul_float_data[i] = multiplier_float_data[i];
    add_float_data[i] =
        offset_float_data[i] - mean_float_data[i] * multiplier_float_data[i];
  }

  DeleteOpAndArrays(model, bn_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
