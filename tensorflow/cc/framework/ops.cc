/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {
namespace ops {

Operation::Operation(Node* n) : inputs_(GetInputs(n)), node_(n) {}

Output Operation::input(int i) const {
  CHECK_NOTNULL(node_);
  CHECK_GE(i, 0);
  CHECK_LT(i, node_->num_inputs());
  // Handle the case where the input was unknown at the time this
  // Operation was constructed.
  if (inputs_[i].first == nullptr && inputs_[i].second == -1) {
    for (const Edge* e : node_->in_edges()) {
      if (e->IsControlEdge()) continue;
      if (e->dst_input() == i) {
        return Output(e->src(), e->src_output());
      }
    }
  }
  return Output(inputs_[i].first, inputs_[i].second);
}

Output Operation::output(int i) const {
  CHECK_NOTNULL(node_);
  CHECK_GE(i, 0);
  CHECK_LT(i, node_->num_outputs());
  return Output(node_, i);
}

uint64 Operation::hash(int64 index) const {
  return ::tensorflow::Hash64(reinterpret_cast<const char*>(&node_),
                              sizeof(Node*), index);
}

Operation::Inputs Operation::GetInputs(Node* node) {
  Operation::Inputs inputs;
  if (node != nullptr) {
    inputs.resize(node->num_inputs(), {nullptr, -1});
    for (const Edge* e : node->in_edges()) {
      if (e->IsControlEdge()) continue;
      inputs[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  return inputs;
}

Input::Initializer::Initializer(
    const std::initializer_list<Input::Initializer>& v) {
  if (v.size() < 1) {
    // Empty initializer list defaults to float tensor with shape (0,)
    tensor = Tensor(DT_FLOAT, TensorShape{0});
    return;
  }
  auto const& first = *v.begin();
  // Check to make sure that the constituent Initializers are all the same
  // type and same shape.
  for (auto const& e : v) {
    if (e.tensor.dtype() != first.tensor.dtype()) {
      status = errors::InvalidArgument(
          "Initializer list components should all have the same type");
      return;
    }
    if (!TensorShape{e.tensor.shape()}.IsSameSize(
            TensorShape{first.tensor.shape()})) {
      status = errors::InvalidArgument(
          "Initializer list components should all have the same shape");
      return;
    }
  }

  // Form the new shape.
  TensorShape shape{static_cast<int64>(v.size())};
  shape.AppendShape(TensorShape{first.tensor.shape()});

  Tensor t(first.tensor.dtype(), shape);

  // Collate the constituent Tensors.
  size_t offset = 0;
  for (auto const& e : v) {
    Tensor elem = e.tensor;
    if (first.tensor.dtype() == DT_STRING) {
      for (int i = 0; i < elem.NumElements(); ++i) {
        t.flat<string>()(offset + i) = elem.flat<string>()(i);
      }
      offset += elem.NumElements();
    } else {
      std::copy_n(elem.tensor_data().data(), elem.TotalBytes(),
                  const_cast<char*>(t.tensor_data().data()) + offset);
      offset += elem.TotalBytes();
    }
  }
  tensor = t;
}

}  // namespace ops
}  // namespace tensorflow
