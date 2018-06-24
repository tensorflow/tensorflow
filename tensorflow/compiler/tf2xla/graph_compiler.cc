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
#include "tensorflow/compiler/tf2xla/const_analysis.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/functionalize_control_flow.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

namespace {
Status PrepareArguments(XlaOpKernelContext* ctx, Graph* graph,
                        const std::vector<const XlaExpression*>& expressions,
                        std::vector<XlaCompiler::Argument>* args) {
  auto builder = ctx->builder();
  auto client = ctx->compiler()->client();
  std::vector<bool> compile_time_constant_flags(expressions.size());

  TF_RETURN_IF_ERROR(
      BackwardsConstAnalysis(*graph, &compile_time_constant_flags));

  args->resize(expressions.size());
  for (int i = 0; i < args->size(); ++i) {
    XlaCompiler::Argument& arg = (*args)[i];
    arg.type = ctx->input_type(i);
    arg.shape = ctx->InputShape(i);

    if (arg.type == DT_RESOURCE) {
      return errors::InvalidArgument(
          "Resource as function argument is not yet implemented.");
    } else if (expressions[i]->has_constant_value()) {
      arg.kind = XlaCompiler::Argument::kConstant;
      arg.constant_value = expressions[i]->constant_value();
    } else if (compile_time_constant_flags[i]) {
      arg.kind = XlaCompiler::Argument::kConstant;
      TF_RET_CHECK(expressions[i]->resource() == nullptr)
          << "Input with resource is not yet implemented.";
      TF_ASSIGN_OR_RETURN(auto constant_graph, builder->BuildConstantSubGraph(
                                                   expressions[i]->handle()));
      TF_ASSIGN_OR_RETURN(auto literal,
                          client->ComputeConstant(constant_graph));
      TF_RETURN_IF_ERROR(
          LiteralToHostTensor(*literal, arg.type, &arg.constant_value));
    } else {
      arg.kind = XlaCompiler::Argument::kParameter;
    }
  }
  return Status::OK();
}
}  // namespace
Status GraphCompiler::Compile() {
  // Maintain a mapping from node id to node outputs.
  using NodeOutputs = std::vector<TensorValue>;
  std::vector<NodeOutputs> output_registry(graph_->num_node_ids());
  auto output_registry_cleanup = gtl::MakeCleanup([&output_registry] {
    for (const NodeOutputs& outputs : output_registry) {
      for (const TensorValue& value : outputs) {
        CHECK(!value.is_ref());
        delete value.tensor;
      }
    }
  });

  // XLA requires determinism, generate a stable ordering from DFS.
  std::vector<Node*> topo_sorted_nodes;
  GetReversePostOrder(*graph_, &topo_sorted_nodes,
                      /*stable_comparator=*/NodeComparatorName());

  OpKernelContext::Params params;
  PartiallySetupParams(&params);

  for (Node* n : topo_sorted_nodes) {
    OpKernel* op_kernel_raw = nullptr;
    Status s = flib_->CreateKernel(n->def(), &op_kernel_raw);
    // Transfer ownership of the kernel to a local smart pointer.
    std::unique_ptr<OpKernel> op_kernel(op_kernel_raw);

    if (!s.ok()) {
      s = AttachDef(s, *n);
      LOG(ERROR) << "Executor failed to create kernel. " << s;
      return s;
    }

    TF_RET_CHECK(!n->IsRecv() && !n->IsSend() && !n->IsSwitch())
        << "Not supported node: " << n->DebugString();
    params.op_kernel = op_kernel.get();
    gtl::InlinedVector<AllocatorAttributes, 4> output_attr(n->num_outputs());
    params.output_attr_array = output_attr.data();

    // tensor_inputs_ is a buffer reused across graph traversal. We clean up and
    // reinitialize the buffer before we visit a new node.
    tensor_inputs_.clear();
    tensor_inputs_.resize(n->num_inputs());

    // Set up inputs from outputs of previous nodes.
    for (auto* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      const Node* src = e->src();
      TF_RET_CHECK(src->id() < output_registry.size());
      const NodeOutputs& src_outputs = output_registry[src->id()];

      tensor_inputs_.at(e->dst_input()) = src_outputs.at(e->src_output());
    }

    OpKernelContext op_context(&params, n->num_outputs());
    if (IsFunctional(n)) {
      TF_RETURN_IF_ERROR(CompileFunctionalNode(n, &op_context));
    } else {
      device_->Compute(CHECK_NOTNULL(params.op_kernel), &op_context);
      Status s = op_context.status();
      if (!s.ok()) {
        return AttachDef(s, n->def());
      }
    }

    // Set up outputs. Also check if outputs from the previous computation is
    // valid.
    NodeOutputs& outputs = output_registry[n->id()];
    outputs.resize(n->num_outputs());
    for (int o = 0; o < n->num_outputs(); ++o) {
      outputs[o] = op_context.release_output(o);
      if (*op_context.is_output_dead() || outputs[o].tensor == nullptr) {
        return errors::Internal("Missing xla_context ", o, "-th output from ",
                                (*op_context.is_output_dead() ? "(dead)" : ""),
                                SummarizeNode(*n));
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
  // For functional nodes, compile them using compiler from the context and call
  // into the functions.
  XlaOpKernelContext xla_op_context(op_context);

  XlaCompiler* compiler = xla_op_context.compiler();

  NameAttrList func;
  if (flib_->GetFunctionLibraryDefinition()->Find(n->def().op())) {
    func.set_name(n->def().op());
  } else {
    func.set_name(FunctionLibraryDefinition::kGradientOp);
  }
  *func.mutable_attr() = n->def().attr();

  std::vector<const XlaExpression*> expressions;

  for (auto tensor : tensor_inputs_) {
    auto expression =
        reinterpret_cast<const XlaExpression*>(tensor->tensor_data().data());
    expressions.push_back(expression);
  }

  // Prepare the arguments and compile the function.
  std::vector<XlaCompiler::Argument> arguments;
  const FunctionBody* fbody;
  TF_RETURN_IF_ERROR(compiler->FindFunctionBody(func, &fbody));

  auto graph = compiler->GetGraph(fbody);

  TF_RETURN_IF_ERROR(
      PrepareArguments(&xla_op_context, graph.get(), expressions, &arguments));

  XlaCompiler::CompileOptions compile_options;
  compile_options.is_entry_computation = false;
  XlaCompiler::CompilationResult result;
  TF_RETURN_IF_ERROR(
      compiler->CompileFunction(compile_options, func, arguments, &result));

  TF_RET_CHECK(arguments.size() == expressions.size());

  std::vector<xla::XlaOp> handles;
  for (int64 i = 0; i < expressions.size(); ++i) {
    if (arguments[i].kind == XlaCompiler::Argument::kConstant) {
      continue;
    }
    handles.push_back(expressions[i]->handle());
  }

  XlaContext& context = XlaContext::Get(op_context);
  auto* b = context.builder();

  auto output_handle = b->Call(*result.computation, handles);
  // The output handle of `Call` computation is a tuple type. Unzip it so
  // that it can fit into future computations.
  int computation_output = 0;
  for (int64 i = 0; i < n->num_outputs(); ++i) {
    if (result.outputs[i].is_constant) {
      xla_op_context.SetConstantOutput(i, result.outputs[i].constant_value);
    } else {
      xla_op_context.SetOutput(
          i, b->GetTupleElement(output_handle, computation_output));
      ++computation_output;
    }
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
