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

#include "tensorflow/c/eager/tape.h"

namespace tensorflow {
namespace eager {

bool GradientTape::ShouldRecord(gtl::ArraySlice<int64> tensor_ids) {
  for (int64 i : tensor_ids) {
    if (tensor_tape_.find(i) != tensor_tape_.end()) {
      return true;
    }
  }
  return false;
}

void GradientTape::Watch(int64 tensor_id) {
  tensor_tape_.emplace(tensor_id, -1);
}

void GradientTape::RecordOperation(
    const string& op_type, gtl::ArraySlice<TapeTensor> output_tensors,
    gtl::ArraySlice<int64> input_tensor_id, void* backward_function,
    const std::function<void()>& backward_function_deleter) {
  if (!ShouldRecord(input_tensor_id)) {
    backward_function_deleter();
    return;
  }
  std::vector<int64> ids;
  ids.reserve(input_tensor_id.size());
  for (int64 i : input_tensor_id) {
    tensor_usage_[i]++;
    ids.push_back(i);
  }
  const int64 op_id = next_op_id_++;
  std::vector<TapeTensor> tensors;
  tensors.reserve(output_tensors.size());
  for (const TapeTensor& o : output_tensors) {
    // Note: the tensor can have already been watched and hence be in the tape,
    // so we cannot check that we're inserting it here.
    tensor_tape_[o.id] = op_id;
    tensor_usage_[o.id] = 1;
    tensors.push_back(o);
  }
  op_tape_[op_id] = OpTapeEntry{op_type, tensors, ids, backward_function,
                                backward_function_deleter};
}

void GradientTape::DeleteTrace(int64 tensor_id) {
  auto it = tensor_usage_.find(tensor_id);
  if (it == tensor_usage_.end()) {
    return;
  }
  it->second--;
  if (it->second != 0) {
    return;
  }
  tensor_usage_.erase(it);
  auto tensor_op_it = tensor_tape_.find(tensor_id);
  if (tensor_op_it == tensor_tape_.end()) {
    return;
  }
  const int64 op_id = tensor_op_it->second;
  if (op_id == -1) {
    // Do not delete watched tensors.
    return;
  }
  tensor_tape_.erase(tensor_op_it);
  auto op_it = op_tape_.find(op_id);
  CHECK(op_it != op_tape_.end());
  for (const auto& output : op_it->second.output_tensor_info) {
    if (tensor_usage_.find(output.id) != tensor_usage_.end()) {
      // Found a usage for an output, so cannot delete the op.
      return;
    }
  }
  for (int64 id : op_it->second.input_tensor_id) {
    DeleteTrace(id);
  }
  op_it->second.backward_function_deleter();
  op_tape_.erase(op_it);
}

std::pair<TensorTape, OpTape> GradientTape::Export() {
  return {std::move(tensor_tape_), std::move(op_tape_)};
}

}  // namespace eager
}  // namespace tensorflow
