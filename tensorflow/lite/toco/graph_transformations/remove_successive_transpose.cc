/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

bool TransformsToIdentity(std::vector<int> const& perm1,
                          std::vector<int> const& perm2) {
  if (perm2.size() != perm1.size() || perm1.empty()) {
    return false;
  }
  // perm1 is the order of the indices after first transpose. When perm1 is
  // reordered according to perm2, if the result is simple increasing sequence
  // i.e., range(0, perm1.size()), then the two transposes cancel each other.
  for (int i = 0; i < perm1.size(); ++i) {
    if (perm1[i] < 0 || perm1[i] >= perm1.size() || perm2[i] < 0 ||
        perm2[i] >= perm1.size()) {
      return false;
    }
    if (perm1[perm2[i]] != i) {
      return false;
    }
  }
  return true;
}

void ReplaceOpInputsWith(Model* model, const string& lookfor,
                         const string& replacewith) {
  for (const auto& op : model->operators) {
    for (int i = 0; i < op->inputs.size(); ++i) {
      if (op->inputs[i] == lookfor) {
        op->inputs[i] = replacewith;
      }
    }
  }
}

}  // namespace

::tensorflow::Status RemoveSuccesiveTranspose::Run(Model* model,
                                                   std::size_t op_index,
                                                   bool* modified) {
  *modified = false;
  auto op = model->operators.begin() + op_index;
  if (op->get()->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }

  TransposeOperator* t_op = static_cast<TransposeOperator*>(op->get());
  if (CountOpsWithInput(*model, t_op->outputs[0]) != 1) {
    return ::tensorflow::Status::OK();
  }
  Operator* next = GetOpWithInput(*model, t_op->outputs[0]);
  if (!next || next->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }

  TransposeOperator* t_next = static_cast<TransposeOperator*>(next);
  if (!CountOpsWithInput(*model, t_next->outputs[0])) {
    return ::tensorflow::Status::OK();
  }

  if (TransformsToIdentity(t_op->perm, t_next->perm)) {
    // Find the input tensor that uses the results of transpose t_next, then
    // make it point to the input of t_op, effectively isolating both the
    // transposes from the graph.
    ReplaceOpInputsWith(model, t_next->outputs[0], t_op->inputs[0]);
    DeleteOpAndArrays(model, t_next);
    DeleteOpAndArrays(model, t_op);
    *modified = true;
  }

  return ::tensorflow::Status::OK();
}

}  // namespace toco
