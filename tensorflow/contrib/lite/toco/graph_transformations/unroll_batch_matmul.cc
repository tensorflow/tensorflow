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

#include "tensorflow/contrib/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/contrib/lite/toco/model.h"
#include "tensorflow/contrib/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

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
bool UnrollBatchMatMul::Run(Model* model, std::size_t op_index) {
  auto batch_op_it = model->operators.begin() + op_index;
  if (batch_op_it->get()->type != OperatorType::kBatchMatMul) {
    return false;
  }
  const auto* batch_op =
      static_cast<const BatchMatMulOperator*>(batch_op_it->get());

  // We must have the shape of at least one input to know our batch size.
  const auto& input_array_a = model->GetArray(batch_op->inputs[0]);
  const auto& input_array_b = model->GetArray(batch_op->inputs[1]);
  if (!input_array_a.has_shape() || !input_array_b.has_shape()) return false;

  // We only support the rank 3 case. If you are batching on rank > 3 you'll
  // have to figure that out.
  CHECK_EQ(input_array_a.shape().dimensions_count(),
           input_array_b.shape().dimensions_count())
      << "Input dimensions must have the same rank";
  if (input_array_a.shape().dimensions_count() == 2) {
    // This is really just a MatMul. This likely means that someone hand-crafted
    // a graphdef with a BatchMatMul when they really wanted a MatMul.
    AddMessageF("Replacing non-batch BatchMatMul %s by a MatMul operator",
                LogName(*batch_op));
    auto* matmul_op = new TensorFlowMatMulOperator;
    matmul_op->inputs = batch_op->inputs;
    matmul_op->outputs = batch_op->outputs;
    const auto matmul_op_it = model->operators.emplace(batch_op_it, matmul_op);
    batch_op_it = matmul_op_it + 1;
    CHECK_EQ(batch_op_it->get(), batch_op);
    model->operators.erase(batch_op_it);
    return true;
  }
  CHECK_EQ(input_array_a.shape().dimensions_count(), 3)
      << "Input arrays must have rank 3";

  // Perform the matmul for each slice of the batch.
  int batch_count = input_array_a.shape().dims(0);
  AddMessageF("Unrolling BatchMatMul %s %d times", LogName(*batch_op),
              batch_count);
  auto tail_it = batch_op_it;
  std::vector<string> pack_inputs;
  for (int batch = 0; batch < batch_count; ++batch) {
    std::string batch_name =
        std::string(batch_op->outputs[0]) + "_b" + std::to_string(batch);

    // tf.slice(a, ...).
    auto* slice_a_op = new SliceOperator;
    slice_a_op->inputs = {
        batch_op->inputs[0],
        CreateInt32Array(model, batch_name + "/slice_a/slice/begin",
                         {batch, 0, 0}),
        CreateInt32Array(
            model, batch_name + "/slice_a/slice/size",
            {1, input_array_a.shape().dims(1), input_array_a.shape().dims(2)}),
    };
    slice_a_op->outputs = {AvailableArrayName(*model, batch_name + "/slice_a")};
    auto& slice_a_op_output = model->GetOrCreateArray(slice_a_op->outputs[0]);
    slice_a_op_output.data_type = input_array_a.data_type;
    tail_it = model->operators.emplace(tail_it, slice_a_op) + 1;

    // Reshape to remove the first dimension ([1,M,N] -> [M,N]).
    auto* slice_a_reshape_op = new TensorFlowReshapeOperator;
    slice_a_reshape_op->inputs = {
        slice_a_op->outputs[0],
        CreateInt32Array(model, batch_name + "/slice_a/reshape/shape",
                         {-1, input_array_a.shape().dims(2)})};
    slice_a_reshape_op->outputs = {
        AvailableArrayName(*model, batch_name + "/slice_a/reshape")};
    auto& slice_a_reshape_op_output =
        model->GetOrCreateArray(slice_a_reshape_op->outputs[0]);
    slice_a_reshape_op_output.data_type = input_array_a.data_type;
    tail_it = model->operators.emplace(tail_it, slice_a_reshape_op) + 1;

    // tf.slice(b, ...).
    auto* slice_b_op = new SliceOperator;
    slice_b_op->inputs = {
        batch_op->inputs[1],
        CreateInt32Array(model, batch_name + "/slice_b/slice/begin", {0, 0, 0}),
        CreateInt32Array(
            model, batch_name + "/slice_b/slice/size",
            {1, input_array_b.shape().dims(1), input_array_b.shape().dims(2)}),
    };
    slice_b_op->outputs = {AvailableArrayName(*model, batch_name + "/slice_b")};
    auto& slice_b_op_output = model->GetOrCreateArray(slice_b_op->outputs[0]);
    slice_b_op_output.data_type = input_array_b.data_type;
    tail_it = model->operators.emplace(tail_it, slice_b_op) + 1;

    // Reshape to remove the first dimension ([1,M,N] -> [M,N]).
    auto* slice_b_reshape_op = new TensorFlowReshapeOperator;
    slice_b_reshape_op->inputs = {
        slice_b_op->outputs[0],
        CreateInt32Array(model, batch_name + "/slice_b/reshape/shape",
                         {-1, input_array_b.shape().dims(2)})};
    slice_b_reshape_op->outputs = {
        AvailableArrayName(*model, batch_name + "/slice_b/reshape")};
    auto& slice_b_reshape_op_output =
        model->GetOrCreateArray(slice_b_reshape_op->outputs[0]);
    slice_b_reshape_op_output.data_type = input_array_b.data_type;
    tail_it = model->operators.emplace(tail_it, slice_b_reshape_op) + 1;

    // tf.matmul(slice_a, slice_b).
    auto* matmul_op = new TensorFlowMatMulOperator;
    matmul_op->inputs = {slice_a_reshape_op->outputs[0],
                         slice_b_reshape_op->outputs[0]};
    matmul_op->outputs = {AvailableArrayName(*model, batch_name)};
    auto& matmul_op_output = model->GetOrCreateArray(matmul_op->outputs[0]);
    matmul_op_output.data_type = input_array_a.data_type;
    tail_it = model->operators.emplace(tail_it, matmul_op) + 1;

    // Add to stack.
    pack_inputs.push_back(matmul_op->outputs[0]);
  }

  // The pack that will join all the individual matmul results together.
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
  return true;
}

}  // namespace toco
