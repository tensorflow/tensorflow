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
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void UnrollBatchMatMul3D(
    const string& input_lhs, const string& input_rhs,
    const BatchMatMulOperator* batch_op, const std::vector<int> batch,
    Model* model, std::vector<std::unique_ptr<Operator>>::iterator* tail_it,
    std::vector<string>* pack_inputs) {
  const std::string batch_name =
      absl::StrCat(batch_op->outputs[0], "_b", absl::StrJoin(batch, "-"));
  const auto& input_array_a = model->GetArray(input_lhs);
  const auto& input_array_b = model->GetArray(input_rhs);
  const int dims_count = input_array_a.shape().dimensions_count();

  // tf.slice(a, ...).
  std::vector<int> begin_indices_a = batch;
  begin_indices_a.resize(dims_count);
  std::vector<int> slice_size_a = input_array_a.shape().dims();
  for (int i = 0; i < batch.size(); ++i) {
    slice_size_a[i] = 1;
  }
  auto* slice_a_op = new SliceOperator;
  slice_a_op->inputs = {
      input_lhs,
      CreateInt32Array(model, batch_name + "/slice_a/slice/begin",
                       begin_indices_a),
      CreateInt32Array(model, batch_name + "/slice_a/slice/size", slice_size_a),
  };
  slice_a_op->outputs = {AvailableArrayName(*model, batch_name + "/slice_a")};
  auto& slice_a_op_output = model->GetOrCreateArray(slice_a_op->outputs[0]);
  slice_a_op_output.data_type = input_array_a.data_type;
  *tail_it = model->operators.emplace(*tail_it, slice_a_op) + 1;

  // Reshape to remove the first dimension ([1,M,N] -> [M,N]).
  auto* slice_a_reshape_op = new TensorFlowReshapeOperator;
  slice_a_reshape_op->inputs = {
      slice_a_op->outputs[0],
      CreateInt32Array(model, batch_name + "/slice_a/reshape/shape",
                       {-1, input_array_a.shape().dims(dims_count - 1)})};
  slice_a_reshape_op->outputs = {
      AvailableArrayName(*model, batch_name + "/slice_a/reshape")};
  auto& slice_a_reshape_op_output =
      model->GetOrCreateArray(slice_a_reshape_op->outputs[0]);
  slice_a_reshape_op_output.data_type = input_array_a.data_type;
  *tail_it = model->operators.emplace(*tail_it, slice_a_reshape_op) + 1;

  // tf.slice(b, ...).
  std::vector<int> begin_indices_b = batch;
  begin_indices_b.resize(dims_count);
  std::vector<int> slice_size_b = input_array_b.shape().dims();
  for (int i = 0; i < batch.size(); ++i) {
    slice_size_b[i] = 1;
  }
  auto* slice_b_op = new SliceOperator;
  slice_b_op->inputs = {
      input_rhs,
      CreateInt32Array(model, batch_name + "/slice_b/slice/begin",
                       begin_indices_b),
      CreateInt32Array(model, batch_name + "/slice_b/slice/size", slice_size_b),
  };
  slice_b_op->outputs = {AvailableArrayName(*model, batch_name + "/slice_b")};
  auto& slice_b_op_output = model->GetOrCreateArray(slice_b_op->outputs[0]);
  slice_b_op_output.data_type = input_array_b.data_type;
  *tail_it = model->operators.emplace(*tail_it, slice_b_op) + 1;

  // Reshape to remove the first dimension ([1,M,N] -> [M,N]).
  auto* slice_b_reshape_op = new TensorFlowReshapeOperator;
  slice_b_reshape_op->inputs = {
      slice_b_op->outputs[0],
      CreateInt32Array(model, batch_name + "/slice_b/reshape/shape",
                       {-1, input_array_b.shape().dims(dims_count - 1)})};
  slice_b_reshape_op->outputs = {
      AvailableArrayName(*model, batch_name + "/slice_b/reshape")};
  auto& slice_b_reshape_op_output =
      model->GetOrCreateArray(slice_b_reshape_op->outputs[0]);
  slice_b_reshape_op_output.data_type = input_array_b.data_type;
  *tail_it = model->operators.emplace(*tail_it, slice_b_reshape_op) + 1;

  // tf.matmul(slice_a, slice_b).
  auto* matmul_op = new TensorFlowMatMulOperator;
  matmul_op->inputs = {slice_a_reshape_op->outputs[0],
                       slice_b_reshape_op->outputs[0]};
  matmul_op->outputs = {AvailableArrayName(*model, batch_name)};
  auto& matmul_op_output = model->GetOrCreateArray(matmul_op->outputs[0]);
  matmul_op_output.data_type = input_array_a.data_type;
  *tail_it = model->operators.emplace(*tail_it, matmul_op) + 1;

  // Add to stack.
  pack_inputs->push_back(matmul_op->outputs[0]);
}

std::vector<string> UnrollBatchMatMulRecursion(
    const string& input_lhs, const string& input_rhs,
    const BatchMatMulOperator* batch_op, Model* model,
    std::vector<std::unique_ptr<Operator>>::iterator* tail_it,
    const std::vector<int>& batch_prefix) {
  const auto& input_array_a = model->GetArray(input_lhs);
  const auto& dims_vec = input_array_a.shape().dims();
  const int current_dim_size = dims_vec[batch_prefix.size()];
  std::vector<string> batch_pack_inputs;

  if (batch_prefix.size() + 3 == dims_vec.size()) {
    // Base case
    for (int batch = 0; batch < current_dim_size; ++batch) {
      std::vector<int> new_batch_prefix = batch_prefix;
      new_batch_prefix.emplace_back(batch);
      UnrollBatchMatMul3D(input_lhs, input_rhs, batch_op, new_batch_prefix,
                          model, tail_it, &batch_pack_inputs);
    }
  } else {
    // Recursion
    for (int batch = 0; batch < current_dim_size; ++batch) {
      std::vector<int> new_batch_prefix = batch_prefix;
      new_batch_prefix.emplace_back(batch);
      std::vector<string> pack_inputs = UnrollBatchMatMulRecursion(
          input_lhs, input_rhs, batch_op, model, tail_it, new_batch_prefix);

      // The pack that will join all the individual matmul results together.
      auto* pack_op = new PackOperator;
      std::string batch_name = absl::StrCat(
          batch_op->outputs[0], "_b", absl::StrJoin(new_batch_prefix, "-"));
      pack_op->inputs = pack_inputs;
      pack_op->outputs = {AvailableArrayName(*model, batch_name + "/pack")};
      auto& pack_op_output = model->GetOrCreateArray(pack_op->outputs[0]);
      pack_op_output.data_type = input_array_a.data_type;
      pack_op->axis = 0;
      pack_op->values_count = pack_inputs.size();
      *tail_it = model->operators.emplace(*tail_it, pack_op) + 1;

      batch_pack_inputs.push_back(pack_op->outputs[0]);
    }
  }
  return batch_pack_inputs;
}

std::vector<int32> GetTransposePerm(const Array& input_array) {
  const int32 dims = input_array.shape().dimensions_count();
  std::vector<int32> perm_array_val(dims);
  for (int i = 0; i < dims; ++i) {
    perm_array_val[i] = i;
  }
  perm_array_val[dims - 2] = dims - 1;
  perm_array_val[dims - 1] = dims - 2;
  return perm_array_val;
}

std::vector<int32> GetTransposeShape(const Shape& input_shape,
                                     const std::vector<int32>& perm_array_val) {
  const int32 dims = input_shape.dimensions_count();
  std::vector<int32> output_shape(dims);
  for (int i = 0; i < dims; ++i) {
    output_shape[i] = input_shape.dims(perm_array_val[i]);
  }
  return output_shape;
}

TransposeOperator* TransposeInput(const string& input, Model* model) {
  const auto& input_array = model->GetArray(input);
  const auto perm_array = GetTransposePerm(input_array);
  const string perm_array_name = CreateInt32Array(
      model, AvailableArrayName(*model, input + "/transpose/perm"), perm_array);
  auto* transpose_op = new TransposeOperator;
  transpose_op->inputs = {input, perm_array_name};
  transpose_op->outputs = {AvailableArrayName(*model, input + "/transpose")};
  auto& transpose_array = model->GetOrCreateArray(transpose_op->outputs[0]);
  *transpose_array.mutable_shape()->mutable_dims() =
      GetTransposeShape(input_array.shape(), perm_array);
  model->GetOrCreateArray(transpose_op->outputs[0]);
  return transpose_op;
}

}  // namespace

// Unrolls a BatchMatMul on the batch dimension.
// We need to slice each batch out of the inputs, matmul them individually, then
// stack them all back together at the end.
//
// This transform effectively looks like:
//  result_slices = []
//  for bat in B:
//    slice_a = tf.reshape(tf.slice(a, [bat, 0, 0], [1, M, N]), [M, N])
//    slice_b = tf.reshape(tf.slice(b, [bat, 0, 0], [1, M, N]), [M, N])
//    slice_c = tf.matmul(slice_a, slice_b)
//    result_slices[bat] = slice_c
//  result = tf.stack(result_slices)
::tensorflow::Status UnrollBatchMatMul::Run(Model* model, std::size_t op_index,
                                            bool* modified) {
  *modified = false;
  auto batch_op_it = model->operators.begin() + op_index;
  if (batch_op_it->get()->type != OperatorType::kBatchMatMul) {
    return ::tensorflow::Status::OK();
  }
  const auto* batch_op =
      static_cast<const BatchMatMulOperator*>(batch_op_it->get());

  auto& tail_it = batch_op_it;

  string input_lhs = batch_op->inputs[0];
  string input_rhs = batch_op->inputs[1];
  const auto& input_lhs_array = model->GetArray(input_lhs);
  const auto& input_rhs_array = model->GetArray(input_rhs);
  if (!input_lhs_array.has_shape() || !input_rhs_array.has_shape())
    return ::tensorflow::Status::OK();

  // Transpose LHS input if necessary.
  if (batch_op->adj_x) {
    TransposeOperator* transpose_op = TransposeInput(input_lhs, model);
    tail_it = model->operators.emplace(tail_it, transpose_op) + 1;
    input_lhs = transpose_op->outputs[0];
  }
  const auto& input_array_a = model->GetArray(input_lhs);

  // Transpose RHS input if necessary.
  if (batch_op->adj_y) {
    TransposeOperator* transpose_op = TransposeInput(input_rhs, model);
    tail_it = model->operators.emplace(tail_it, transpose_op) + 1;
    input_rhs = transpose_op->outputs[0];
  }
  const auto& input_array_b = model->GetArray(input_rhs);

  const int dims = input_array_a.shape().dimensions_count();
  for (int i = 0; i < dims - 2; ++i) {
    CHECK_EQ(input_array_a.shape().dims(i), input_array_b.shape().dims(i))
        << "input array not consistent at index " << i;
  }
  CHECK_EQ(input_array_a.shape().dims(dims - 1),
           input_array_b.shape().dims(dims - 2))
      << "Input dimensions must be compatible for multipication. shape a = ["
      << absl::StrJoin(input_array_a.shape().dims(), ", ") << "], shape b = ["
      << absl::StrJoin(input_array_b.shape().dims(), ", ") << "]";

  if (dims == 2) {
    // This is really just a MatMul. This likely means that someone hand-crafted
    // a graphdef with a BatchMatMul when they really wanted a MatMul.
    AddMessageF("Replacing non-batch BatchMatMul %s by a MatMul operator",
                LogName(*batch_op));
    auto* matmul_op = new TensorFlowMatMulOperator;
    matmul_op->inputs = {input_lhs, input_rhs};
    matmul_op->outputs = batch_op->outputs;
    tail_it = model->operators.emplace(tail_it, matmul_op) + 1;
    CHECK_EQ(tail_it->get(), batch_op);
    model->operators.erase(tail_it);
    *modified = true;
    return ::tensorflow::Status::OK();
  }

  CHECK_GE(input_array_a.shape().dimensions_count(), 3)
      << "Input arrays must have rank >= 3";

  const auto& dims_vec = input_array_a.shape().dims();
  AddMessageF("Unrolling BatchMatMul %s %d times", LogName(*batch_op),
              std::accumulate(dims_vec.begin(), dims_vec.end() - 2, 1,
                              std::multiplies<int>()));

  std::vector<string> pack_inputs = UnrollBatchMatMulRecursion(
      input_lhs, input_rhs, batch_op, model, &tail_it, {});
  auto* pack_op = new PackOperator;
  pack_op->inputs = pack_inputs;
  pack_op->outputs = {batch_op->outputs[0]};
  pack_op->axis = 0;
  pack_op->values_count = pack_inputs.size();
  model->operators.emplace(tail_it, pack_op);

  // Remove the old batch matmul now that we've unrolled.
  batch_op_it = model->operators.begin();
  for (; batch_op_it != model->operators.end(); ++batch_op_it) {
    if (batch_op_it->get() == batch_op) {
      break;
    }
  }
  CHECK(batch_op_it != model->operators.end());
  CHECK(batch_op_it->get() == batch_op);
  model->operators.erase(batch_op_it);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
