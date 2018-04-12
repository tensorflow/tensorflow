/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/function_optimizer.h"
#include <unordered_map>
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"

namespace tensorflow {
namespace grappler {
namespace {

class FunctionInliningContext {
 public:
  explicit FunctionInliningContext(const GrapplerItem& item,
                                   RewriterConfig::Toggle opt_level)
      : library_(&item.graph.library()),
        opt_level_(opt_level),
        functions_(InliningCandidates(item)) {}

  const FunctionDefLibrary& Library() const { return *library_; }

  bool HasInlinedFunctions() const { return !functions_.empty(); }

  // Find inlining candidate by name. Return nullptr if not found.
  const FunctionDef* FindInlinedFunction(const string& name) const {
    auto it = functions_.find(name);
    if (it != functions_.end()) {
      return it->second;
    } else {
      return nullptr;
    }
  }

 private:
  std::unordered_map<string, const FunctionDef*> InliningCandidates(
      const GrapplerItem& item) const {
    std::unordered_map<string, const FunctionDef*> functions;
    for (const FunctionDef& func : item.graph.library().function()) {
      // Don't inline functions marked as noinline
      if (func.attr().count("_noinline") != 0 &&
          func.attr().at("_noinline").b() &&
          opt_level_ != RewriterConfig::AGGRESSIVE) {
        continue;
      }
      // Can't create IdentityN nodes with no input or output: skip these
      // functions for now.
      if (func.signature().input_arg_size() == 0 ||
          func.signature().output_arg_size() == 0) {
        continue;
      }
      functions[func.signature().name()] = &func;
    }
    return functions;
  }

  const FunctionDefLibrary* library_;
  RewriterConfig::Toggle opt_level_;
  std::unordered_map<string, const FunctionDef*> functions_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionInliningContext);
};

// Copy input/output argument type to the type_list. Return error if argument
// type is not explicitly defined, and not specified in function attributes.
Status CopyArgType(const NodeDef& func_node,
                   const std::unordered_map<string, AttrValue>& func_attr,
                   const string& arg_kind, const OpDef::ArgDef& arg,
                   AttrValue::ListValue* type_list) {
  if (arg.type() != DT_INVALID) {
    type_list->add_type(arg.type());
  } else {
    auto it = func_attr.find(arg.type_attr());
    if (it == func_attr.end() || it->second.type() == DT_INVALID) {
      return errors::InvalidArgument(
          "Invalid ", arg_kind, " argument ", arg.name(), " for function ",
          func_node.op(), " instantiated by ", func_node.name());
    }
    type_list->add_type(it->second.type());
  }
  return Status::OK();
}

// Add an IdentityN op to hook the function inputs to: this ensures that
// they're all evaluated before the evaluation of the function body starts.
Status HookInlinedFunctionInputs(
    const NodeDef& func_node, const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr, NodeDef* inputs) {
  inputs->set_name(strings::StrCat(func_node.name(), "/", "inlined_inputs"));
  inputs->set_op("IdentityN");
  inputs->set_device(func_node.device());
  *inputs->mutable_input() = func_node.input();
  AttrValue::ListValue* type_list =
      (*inputs->mutable_attr())["T"].mutable_list();
  for (const OpDef::ArgDef& arg : func.signature().input_arg()) {
    TF_RETURN_IF_ERROR(
        CopyArgType(func_node, func_attr, "input", arg, type_list));
  }
  return Status::OK();
}

// Add an IdentityN op to hook the function outputs to: this ensures that the
// function body is fully evaluated before its fanout gets scheduled.
Status HookInlinedFunctionOutputs(
    const NodeDef& func_node, const FunctionDef& func,
    const std::unordered_map<string, AttrValue>& func_attr,
    const gtl::ArraySlice<string> fetch, NodeDef* outputs) {
  outputs->set_name(func_node.name());
  outputs->set_op("IdentityN");
  outputs->set_device(func_node.device());
  AttrValue::ListValue* type_list =
      (*outputs->mutable_attr())["T"].mutable_list();
  for (int i = 0; i < func.signature().output_arg_size(); ++i) {
    const OpDef::ArgDef& arg = func.signature().output_arg(i);
    TF_RETURN_IF_ERROR(
        CopyArgType(func_node, func_attr, "output", arg, type_list));
    // Use the fetch names since they take into account the output mapping.
    outputs->add_input(strings::StrCat(func_node.name(), "/", fetch[i]));
  }
  return Status::OK();
}

Status InlineFunction(const NodeDef& func_node, const FunctionDef& func,
                      const FunctionInliningContext& ctx,
                      GraphDef* optimized_graph) {
  const std::unordered_map<string, AttrValue> func_attr(
      func_node.attr().begin(), func_node.attr().end());

  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, ctx.Library());
  if (!item) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name());
  }

  std::unordered_map<string, int> input_nodes;
  for (int i = 0; i < func.signature().input_arg_size(); ++i) {
    const OpDef::ArgDef& arg = func.signature().input_arg(i);
    input_nodes[arg.name()] = i;
  }

  // Hook inlined function inputs to IdentityN node
  NodeDef* func_inputs = optimized_graph->add_node();
  TF_RETURN_IF_ERROR(
      HookInlinedFunctionInputs(func_node, func, func_attr, func_inputs));

  for (NodeDef& func_body_node : *item->graph.mutable_node()) {
    if (input_nodes.find(func_body_node.name()) != input_nodes.end()) {
      CHECK_EQ(0, func_body_node.input_size());
      // Turn input placeholders into identity nodes
      if (IsPlaceholder(func_body_node)) {
        func_body_node.set_op("Identity");
      }
      int input_id = input_nodes[func_body_node.name()];
      func_body_node.add_input(
          strings::StrCat(func_inputs->name(), ":", input_id));
    } else {
      // Update the input names if any.
      for (string& input : *func_body_node.mutable_input()) {
        input = AddPrefixToNodeName(input, /*prefix=*/func_node.name());
      }
      // If the node has no input, make hook it up to the func_inputs node to
      // ensure it runs in the same frame as the other nodes of the function
      // body.
      if (func_body_node.input_size() == 0) {
        *func_body_node.add_input() = AsControlDependency(func_inputs->name());
      }
    }

    // Add the node name as a prefix to avoid collisions after inlining
    func_body_node.set_name(
        strings::StrCat(func_node.name(), "/", func_body_node.name()));

    // Make sure the node is placed
    func_body_node.set_device(func_node.device());

    // Check if a body node is itself a function
    const FunctionDef* func_body_node_func =
        ctx.FindInlinedFunction(func_body_node.op());
    if (func_body_node_func != nullptr) {
      // Recursively inline function calls
      TF_RETURN_IF_ERROR(InlineFunction(func_body_node, *func_body_node_func,
                                        ctx, optimized_graph));
    } else {
      // Annotate the node with the function attributes.
      for (const auto& attr : func.attr()) {
        func_body_node.mutable_attr()->insert(attr);
      }
      // Move the node to the main graph
      optimized_graph->add_node()->Swap(&func_body_node);
    }
  }

  // Hook inlined function outputs to IdentityN node
  NodeDef* func_outputs = optimized_graph->add_node();
  TF_RETURN_IF_ERROR(HookInlinedFunctionOutputs(func_node, func, func_attr,
                                                item->fetch, func_outputs));

  return Status::OK();
}

class FakeCPUDevice : public Device {
 public:
  FakeCPUDevice(Env* env, const DeviceAttributes& attr) : Device(env, attr) {}
  Status Sync() override { return Status::OK(); }
};

class SymbolicGradientEnv {
 public:
  SymbolicGradientEnv(int graph_version, const FunctionDefLibrary& library)
      : graph_version_(graph_version), library_(library) {}

  FunctionLibraryDefinition* function_library() {
    InitializeIfNeeded();
    return fld_.get();
  }
  FunctionLibraryRuntime* function_library_runtime() {
    InitializeIfNeeded();
    return flr_;
  }

 private:
  // This initialization is expensive. Do it lazily to avoid paying for it
  // unless it's needed.
  void InitializeIfNeeded() {
    if (flr_) {
      return;
    }
    Env* env = Env::Default();
    DeviceAttributes attr;
    attr.set_name("/device:CPU:0");
    attr.set_device_type("CPU");
    FakeCPUDevice* dev = new FakeCPUDevice(env, attr);
    std::vector<Device*> devices;
    devices.push_back(dev);
    dvc_mgr_.reset(new DeviceMgr(devices));
    fld_.reset(new FunctionLibraryDefinition(OpRegistry::Global(), library_));
    OptimizerOptions optimizer_opts;
    optimizer_opts.set_do_function_inlining(true);
    pflr_.reset(new ProcessFunctionLibraryRuntime(
        dvc_mgr_.get(), env, graph_version_, fld_.get(), optimizer_opts));
    flr_ = pflr_->GetFLR(dev->name());
  }

  const int graph_version_;
  const FunctionDefLibrary& library_;
  std::unique_ptr<DeviceMgr> dvc_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> fld_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  FunctionLibraryRuntime* flr_ = nullptr;
};

Status InlineSymbolicGradient(const NodeDef& node, SymbolicGradientEnv* env,
                              GraphDef* inlined_graph) {
  GraphDef graph_def;

  // Create a node to anchor the gradient inputs
  NodeDef* inlined_input = graph_def.add_node();
  inlined_input->set_name("FunctionInputs");
  inlined_input->set_op("IdentityN");
  AttrValue::ListValue* type_list =
      (*inlined_input->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tin").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }

  // Add the gradient node
  NodeDef* inlined = graph_def.add_node();
  *inlined = node;
  inlined->clear_input();
  for (int i = 0; i < node.attr().at("Tin").list().type_size(); ++i) {
    inlined->add_input(strings::StrCat(inlined_input->name(), ":", i));
  }

  // Create a node to anchor the gradient outputs
  NodeDef* inlined_output = graph_def.add_node();
  inlined_output->set_name("FunctionOutputs");
  inlined_output->set_op("IdentityN");
  type_list = (*inlined_output->mutable_attr())["T"].mutable_list();
  for (const auto& type : node.attr().at("Tout").list().type()) {
    type_list->add_type(static_cast<DataType>(type));
  }
  for (int i = 0; i < node.attr().at("Tout").list().type_size(); ++i) {
    inlined_output->add_input(strings::StrCat(inlined->name(), ":", i));
  }

  // Convert the graphdef to a graph
  GraphConstructorOptions graph_ctor_opts;
  graph_ctor_opts.allow_internal_ops = true;
  graph_ctor_opts.expect_device_spec = false;
  Graph graph(env->function_library());
  TF_RETURN_IF_ERROR(
      ConvertGraphDefToGraph(graph_ctor_opts, graph_def, &graph));

  // Recursively inline the functions until there is nothing more to inline. We
  // should at least expand one function.
  int counter = 0;
  while (counter < 50 &&
         ExpandInlineFunctions(env->function_library_runtime(), &graph)) {
    ++counter;
  }

  GraphDef inlined_graph_def;
  graph.ToGraphDef(&inlined_graph_def);

  // Add the default values of attributes to the nodes that have been inlined.
  TF_RETURN_IF_ERROR(AddDefaultAttrsToGraphDef(&inlined_graph_def,
                                               *graph.op_registry(), 0, true));

  // Add the inlined nodes to the graph
  for (NodeDef& inlined_node : *inlined_graph_def.mutable_node()) {
    if (inlined_node.name() == "FunctionOutputs") {
      inlined_node.set_name(node.name());
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
    } else if (inlined_node.name() == "FunctionInputs") {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      inlined_node.clear_input();
      for (int i = 0; i < node.input_size(); ++i) {
        inlined_node.add_input(node.input(i));
      }
    } else {
      inlined_node.set_name(
          AddPrefixToNodeName(inlined_node.name(), node.name()));
      for (int i = 0; i < inlined_node.input_size(); ++i) {
        inlined_node.set_input(
            i, AddPrefixToNodeName(inlined_node.input(i), node.name()));
      }
      // If the node has no input, hook it up to the function input node to make
      // sure it runs in the same frame as the other nodes of the function body.
      if (inlined_node.input_size() == 0) {
        *inlined_node.add_input() = AsControlDependency(
            AddPrefixToNodeName("FunctionInputs", node.name()));
      }
    }
    inlined_node.set_device(node.device());
    inlined_graph->add_node()->Swap(&inlined_node);
  }

  return Status::OK();
}

}  // namespace

Status FunctionOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                   GraphDef* optimized_graph) {
  FunctionInliningContext function_inlining_ctx(item, opt_level_);

  // Nothing to do here.
  if (!function_inlining_ctx.HasInlinedFunctions()) {
    *optimized_graph = item.graph;
    return Status::OK();
  }

  SymbolicGradientEnv env(item.graph.versions().producer(),
                          item.graph.library());

  for (const NodeDef& node : item.graph.node()) {
    if (node.op() == "SymbolicGradient") {
      TF_RETURN_IF_ERROR(InlineSymbolicGradient(node, &env, optimized_graph));
      continue;
    }

    const FunctionDef* func =
        function_inlining_ctx.FindInlinedFunction(node.op());
    if (func != nullptr) {
      TF_RETURN_IF_ERROR(
          InlineFunction(node, *func, function_inlining_ctx, optimized_graph));
    } else {
      *optimized_graph->add_node() = node;
    }
  }

  // TODO(bsteiner): specialize the implementation of functions that can't be
  // inlined based on the context in which they're instantiated.

  // TODO(bsteiner): trim the library to remove unused function definitions
  *optimized_graph->mutable_versions() = item.graph.versions();
  *optimized_graph->mutable_library() = item.graph.library();

  return Status::OK();
}

void FunctionOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimized_graph,
                                 double result) {
  // Nothing to do for FunctionOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
