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

#include "tensorflow/compiler/tf2xla/graph_compiler.h"

#include <deque>
#include <numeric>
#include <vector>

#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

Status GraphCompiler::Compile() {
  std::vector<NodeBinding> bindings(graph_->num_node_ids());
  std::vector<Node*> topo_sorted_nodes;
  // XLA requires determinism, generate a stable ordering from DFS.
  GetReversePostOrder(*graph_, &topo_sorted_nodes,
                      /*stable_comparator=*/NodeComparatorID());

  OpKernelContext::Params params;
  PartiallySetupParams(&params);

  for (Node* n : topo_sorted_nodes) {
    // Set up bindings.
    NodeBinding& binding = bindings[n->id()];
    binding.node = n;
    Status s = flib_->CreateKernel(n->def(), &binding.op_kernel);
    binding.output_attrs.resize(n->num_outputs());
    if (!s.ok()) {
      binding.op_kernel = nullptr;
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }
  }

  // Bindings are initialized by the size of graph_->num_node_ids. However, the
  // graph may contain dead nodes that still hold a valid node id. Thus
  // graph_->num_node_ids could be larger than number of topo sorted nodes.
  TF_RET_CHECK(bindings.size() >= topo_sorted_nodes.size());

  for (Node* n : topo_sorted_nodes) {
    TF_RET_CHECK(!n->IsRecv() && !n->IsSend() && !n->IsSwitch())
        << "Not supported node: " << n->DebugString();
    NodeBinding& binding = bindings[n->id()];
    params.op_kernel = binding.op_kernel;
    params.output_attr_array = binding.output_attrs.data();

    // tensor_inputs_ is a buffer reused across graph traversal. We clean up and
    // reinitialize the buffer before we visit a new node.
    tensor_inputs_.clear();
    tensor_inputs_.resize(n->num_inputs());

    // Set up inputs from outputs of previous nodes.
    for (auto* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      Node* src = e->src();
      tensor_inputs_[e->dst_input()] =
          bindings[src->id()].tensor_values[e->src_output()];
    }

    OpKernelContext op_context(&params, n->num_outputs());
    if (IsFunctional(n)) {
      TF_RETURN_IF_ERROR(CompileFunctionalNode(n, &op_context));
    } else {
      device_->Compute(CHECK_NOTNULL(params.op_kernel), &op_context);
      Status s = op_context.status();
      TF_RETURN_IF_ERROR(s);
    }

    // Set up outputs. Also check if outputs from the previous computation is
    // valid.
    for (int o = 0; o < n->num_outputs(); ++o) {
      const auto tensor_val = op_context.release_output(o);
      if (*op_context.is_output_dead() || tensor_val.tensor == nullptr) {
        return errors::Internal("Missing xla_context ", o, "-th output from ",
                                (*op_context.is_output_dead() ? "(dead)" : ""),
                                SummarizeNode(*n));
      }
      binding.tensor_values.push_back(tensor_val);
    }
  }

  // Clean up tensor data and op kernels.
  for (NodeBinding& binding : bindings) {
    delete binding.op_kernel;
    for (auto& t : binding.tensor_values) {
      if (!t.is_ref()) {
        delete t.tensor;
      }
    }
  }
  return Status::OK();
}

bool GraphCompiler::IsFunctional(Node* n) {
  return n->type_string() == FunctionLibraryDefinition::kGradientOp ||
         (flib_->GetFunctionLibraryDefinition()->Find(n->def().op()) !=
          nullptr);
}

Status GraphCompiler::CompileFunctionalNode(Node* n,
                                            OpKernelContext* op_context) {
  TF_RET_CHECK(IsFunctional(n));
  // For functional nodes, compile them using compiler_ and call into the
  // functions.
  XlaOpKernelContext xla_op_context(op_context);

  std::vector<XlaCompiler::Argument> arguments;
  XlaCompiler::CompilationResult result;
  NameAttrList func;
  if (flib_->GetFunctionLibraryDefinition()->Find(n->def().op())) {
    func.set_name(n->def().op());
  } else {
    func.set_name(FunctionLibraryDefinition::kGradientOp);
  }
  *func.mutable_attr() = n->def().attr();

  // Compile the graph using the function compiler.
  TF_ASSIGN_OR_RETURN(auto computation, compiler_(func, &xla_op_context));
  XlaContext& context = XlaContext::Get(op_context);
  auto* b = context.builder();

  // Graph data handles from the inputs.
  std::vector<xla::ComputationDataHandle> handles;
  for (auto tensor : tensor_inputs_) {
    auto expression =
        reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
    // TODO(yunxing): Support two rare cases below where input is a resource or
    // contains a null handle.
    TF_RET_CHECK(expression->resource() == nullptr)
        << "Input with resource is not supported.";
    TF_RET_CHECK(expression->handle().handle() != 0)
        << "Invalid computation handle.";
    handles.push_back(expression->handle());
  }
  auto output_handle = b->Call(*computation, handles);
  // The output handle of `Call` computation is a tuple type. Unzip it so
  // that it can into fit future computations.
  for (int64 idx = 0; idx < n->num_outputs(); ++idx) {
    xla_op_context.SetOutput(idx, b->GetTupleElement(output_handle, idx));
  }
  return b->first_error();
}

void GraphCompiler::PartiallySetupParams(OpKernelContext::Params* params) {
  params->device = device_;
  params->inputs = &tensor_inputs_;
  params->step_container = step_container_;
  params->resource_manager = device_->resource_manager();
}

}  // namespace tensorflow
