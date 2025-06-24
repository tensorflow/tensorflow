/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/function_body.h"

#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/common_runtime/arg_ret_placement.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/hash.h"
#include "tensorflow/core/platform/refcount.h"

namespace tensorflow {

FunctionBody::FunctionBody(core::RefCountPtr<FunctionRecord>&& record,
                           DataTypeSlice arg_t, DataTypeSlice ret_t, Graph* g)
    : record(std::move(record)),
      graph(g),
      arg_types(arg_t.begin(), arg_t.end()),
      ret_types(ret_t.begin(), ret_t.end()) {
  // 1. Find regular Arg/Ret nodes.
  this->arg_nodes.resize(arg_types.size());
  this->ret_nodes.resize(ret_types.size());
  for (Node* n : this->graph->op_nodes()) {
    absl::InlinedVector<Node*, 4UL>* node_vec;
    if (n->type_string() == FunctionLibraryDefinition::kRetOp ||
        n->type_string() == FunctionLibraryDefinition::kDeviceRetOp) {
      node_vec = &this->ret_nodes;
    } else if (n->type_string() == FunctionLibraryDefinition::kArgOp ||
               n->type_string() == FunctionLibraryDefinition::kDeviceArgOp) {
      node_vec = &this->arg_nodes;
    } else {
      continue;
    }
    int index;
    TF_CHECK_OK(GetNodeAttr(n->attrs(), "index", &index));
    CHECK_LE(0, index);
    CHECK_LT(index, node_vec->size());
    (*node_vec)[index] = n;
  }
  // 2. Find ControlRet nodes that must be always executed.
  std::unordered_set<absl::string_view, StringPieceHasher>
      control_ret_node_names;
  for (const auto& control_ret : this->record->fdef().control_ret()) {
    control_ret_node_names.insert(control_ret.second);
  }
  this->control_ret_nodes.reserve(control_ret_node_names.size());
  for (Node* n : this->graph->op_nodes()) {
    if (control_ret_node_names.count(n->name()) > 0) {
      this->control_ret_nodes.push_back(n);
    }
  }
}

FunctionBody::~FunctionBody() { delete this->graph; }

absl::Status FunctionBody::Finalize() {
  // Get the allocator attributes for the function body args and rets first to
  // avoid mutating the struct in case of an error.
  std::vector<AllocatorAttributes> args_alloc_attrs;
  std::vector<AllocatorAttributes> rets_alloc_attrs;
  TF_RETURN_IF_ERROR(full_type::SetAllocAttrsForArgs(
      this->arg_nodes, this->arg_types, args_alloc_attrs));
  TF_RETURN_IF_ERROR(full_type::SetAllocAttrsForRets(
      this->ret_nodes, this->ret_types, rets_alloc_attrs));
  // Move them to the struct.
  this->args_alloc_attrs.clear();
  this->rets_alloc_attrs.clear();
  std::move(args_alloc_attrs.begin(), args_alloc_attrs.end(),
            std::back_inserter(this->args_alloc_attrs));
  std::move(rets_alloc_attrs.begin(), rets_alloc_attrs.end(),
            std::back_inserter(this->rets_alloc_attrs));

  // Unreference the function record.
  this->record.reset();

  // Destruct the owned graph.
  if (this->graph != nullptr) {
    delete this->graph;
    this->graph = nullptr;
  }

  // Clear the vectors holding the pointers to the nodes in the destructed
  // graph.
  this->arg_nodes.clear();
  this->ret_nodes.clear();
  this->control_ret_nodes.clear();

  return absl::OkStatus();
}

}  // end namespace tensorflow
