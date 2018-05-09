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
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace {

// Mark functions that were created as a result of function specialization.
constexpr char kGrapplerSpecializedFuncAttr[] = "_GrapplerSpecializedFunc";

constexpr char kNoInlineAttr[] = "_noinline";

bool AttrIsTrue(const FunctionDef& func, const string& attr) {
  return func.attr().count(attr) != 0 && func.attr().at(attr).b();
}

bool MarkedSpecialized(const FunctionDef& func) {
  return AttrIsTrue(func, kGrapplerSpecializedFuncAttr);
}

bool MarkedNoInline(const FunctionDef& func) {
  return AttrIsTrue(func, kNoInlineAttr);
}

// Find unique name for the specialized function. Collision can happen if
// specialized function is instantiated for the nodes with the same name (e.g.
// inside function body of two different functions).
string UniqueSpecializedFunctionName(const FunctionDef& func,
                                     const NodeDef& func_node,
                                     const FunctionLibraryDefinition& flib) {
  using str_util::StringReplace;
  using strings::StrCat;

  string specialized_name = StrCat(func.signature().name(), "_specialized_for_",
                                   StringReplace(func_node.name(), "/", "_",
                                                 /*replace_all*/ true));
  string unique_name = specialized_name;

  int idx = 0;
  while (flib.Find(unique_name)) {
    unique_name = strings::StrCat(specialized_name, "_", ++idx);
  }
  return unique_name;
}

// Specialized function instantiation type parameters, body parameters, and
// const inputs.
struct FunctionSpecializationSignature {
  string func_name;
  std::unordered_map<string, DataType> type_parameters;
  std::unordered_map<string, AttrValue> body_parameters;
  std::unordered_map<int, string> const_inputs;

  bool operator==(const FunctionSpecializationSignature& other) const {
    bool equals = func_name == other.func_name &&
                  type_parameters == other.type_parameters &&
                  const_inputs == other.const_inputs;

    if (!equals) return false;

    // Equality is not defined for AttrValue.
    if (body_parameters.size() != other.body_parameters.size()) return false;

    for (const auto& lhs : body_parameters) {
      auto it = other.body_parameters.find(lhs.first);
      if (it == other.body_parameters.end()) return false;
      if (!AreAttrValuesEqual(lhs.second, (*it).second)) return false;
    }

    return true;
  }

  struct Hash {
    uint64 operator()(FunctionSpecializationSignature const& s) const {
      uint64 h = Hash64(s.func_name);

      // Use std::map for deterministic iteration order.

      std::map<string, DataType> types(s.type_parameters.begin(),
                                       s.type_parameters.end());
      for (const auto& pair : types) {
        AttrValue attr_value;
        attr_value.set_type(pair.second);
        h = Hash64Combine(Hash64(pair.first), h);
        h = Hash64Combine(AttrValueHash(attr_value), h);
      }

      std::map<string, AttrValue> body(s.body_parameters.begin(),
                                       s.body_parameters.end());
      for (const auto& pair : body) {
        h = Hash64Combine(Hash64(pair.first), h);
        h = Hash64Combine(AttrValueHash(pair.second), h);
      }

      std::map<int, string> inputs(s.const_inputs.begin(),
                                   s.const_inputs.end());
      for (const auto& pair : inputs) {
        h = Hash64Combine(std::hash<int>()(pair.first), h);
        h = Hash64Combine(Hash64(pair.second), h);
      }

      return h;
    }
  };
};

struct FunctionSpecialization {
  string specialized_func_name;
  std::unordered_set<string> const_inputs;
  std::unordered_set<string> control_deps;
};

class FunctionOptimizerContext {
 public:
  explicit FunctionOptimizerContext(RewriterConfig::Toggle opt_level,
                                    const GrapplerItem& item)
      : function_library_(OpRegistry::Global(), item.graph.library()) {
    InitializeTrulyConstNodes(item);
    InitializeInlinedFunctions(opt_level, item);
  }

  const FunctionLibraryDefinition& function_library() const {
    return function_library_;
  }

  FunctionLibraryDefinition* mutable_function_library() {
    return &function_library_;
  }

  bool IsInlinedFunction(const string& name) const {
    return inlined_functions_.count(name) > 0;
  }

  bool IsTrulyConst(const string& name) const {
    return TrulyConstNode(name) != nullptr;
  }

  const NodeDef* TrulyConstNode(const string& name) const {
    return gtl::FindWithDefault(truly_const_nodes_, name, nullptr);
  }

  // Find inlining candidate by name. Return nullptr if not found.
  const FunctionDef* FindInlinedFunction(const string& name) const {
    return gtl::FindWithDefault(inlined_functions_, name, nullptr);
  }

  const FunctionSpecialization* FindFunctionSpecialization(
      const FunctionSpecializationSignature& sig) const {
    return gtl::FindOrNull(specialized_functions_, sig);
  }

  void AddSpecializedFunction(const FunctionSpecializationSignature& sig,
                              const FunctionSpecialization& specialized_func) {
    specialized_functions_.emplace(sig, specialized_func);
  }

 private:
  void InitializeTrulyConstNodes(const GrapplerItem& item) {
    std::unordered_set<string> feed_nodes;
    for (const auto& feed : item.feed) {
      feed_nodes.insert(NodeName(feed.first));
    }

    for (const NodeDef& node : item.graph.node()) {
      if (IsConstant(node) && feed_nodes.count(node.name()) == 0) {
        truly_const_nodes_[node.name()] = &node;
      }
    }
  }

  void InitializeInlinedFunctions(RewriterConfig::Toggle opt_level,
                                  const GrapplerItem& item) {
    bool aggressive = opt_level == RewriterConfig::AGGRESSIVE;

    for (const FunctionDef& func : item.graph.library().function()) {
      // Can't create IdentityN nodes with no input or output: skip these
      // functions for now.
      if (func.signature().input_arg_size() == 0 ||
          func.signature().output_arg_size() == 0) {
        continue;
      }
      bool marked_noinline = MarkedNoInline(func);
      bool marked_specialized = MarkedSpecialized(func);

      if (!marked_specialized && (!marked_noinline || aggressive)) {
        inlined_functions_[func.signature().name()] = &func;
      }
    }
  }

  FunctionLibraryDefinition function_library_;
  // Functions that can be inlined into optimized graph.
  std::unordered_map<string, const FunctionDef*> inlined_functions_;
  // Nodes that are Const and not in feed.
  std::unordered_map<string, const NodeDef*> truly_const_nodes_;

  // Specialized functions.
  std::unordered_map<FunctionSpecializationSignature,
                     const FunctionSpecialization,
                     FunctionSpecializationSignature::Hash>
      specialized_functions_;

  TF_DISALLOW_COPY_AND_ASSIGN(FunctionOptimizerContext);
};

bool HasTrulyConstInputs(const NodeDef& node,
                         const FunctionOptimizerContext& ctx) {
  const auto is_truly_const = [&ctx](const string& input) {
    return ctx.IsTrulyConst(NodeName(input));
  };
  return std::any_of(node.input().begin(), node.input().end(), is_truly_const);
}

// Return trimmed FunctionDefLibrary with functions that are reachable from
// the optimized graph.
FunctionDefLibrary TrimFunctionLibrary(const FunctionLibraryDefinition& flib,
                                       const GraphDef& optimized_graph) {
  // Functions that are reachable from the optimized graph.
  std::unordered_set<string> keep_funcs;

  std::vector<const FunctionDef*> func_queue;
  func_queue.reserve(flib.num_functions());

  // Add registered and not already processed functions to the queue by name.
  const auto add_to_func_queue = [&](const string& func_name) {
    const FunctionDef* func = flib.Find(func_name);
    if (func && keep_funcs.find(func_name) == keep_funcs.end()) {
      func_queue.push_back(func);
    }
  };

  // Find all the functions that are reachable from the given node.
  const auto add_node_to_func_queue = [&](const NodeDef& node) {
    // Node itself can be a call to the function.
    add_to_func_queue(node.op());

    // Or node can have an attribute referencing a function.
    for (const auto& attr : node.attr()) {
      const auto& attr_value = attr.second;

      // 1. AttrValue.func
      if (attr_value.has_func()) {
        add_to_func_queue(attr_value.func().name());
      }

      // 2. AttrValue.ListValue.func
      if (attr_value.has_list()) {
        for (const auto& func : attr_value.list().func()) {
          add_to_func_queue(func.name());
        }
      }
    }
  };

  // Add all functions that are directly called from the optimized graph.
  const auto& graph_nodes = optimized_graph.node();
  std::for_each(graph_nodes.begin(), graph_nodes.end(), add_node_to_func_queue);

  // Process all reachable functions.
  while (!func_queue.empty()) {
    const FunctionDef* func = func_queue.back();
    func_queue.pop_back();

    const string& func_name = func->signature().name();
    keep_funcs.insert(func_name);

    // Find all the functions called from the function body.
    const auto& func_body = func->node_def();
    std::for_each(func_body.begin(), func_body.end(), add_node_to_func_queue);

    // Check if the function has a registered gradient.
    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) add_to_func_queue(grad_func_name);
  }

  FunctionDefLibrary lib;
  for (const string& func_name : keep_funcs) {
    const FunctionDef* func = CHECK_NOTNULL(flib.Find(func_name));
    *lib.add_function() = *func;

    const string grad_func_name = flib.FindGradient(func_name);
    if (!grad_func_name.empty()) {
      GradientDef* gd = lib.add_gradient();
      gd->set_function_name(func_name);
      gd->set_gradient_func(grad_func_name);
    }
  }

  VLOG(3) << "Trimmed function library: " << keep_funcs.size() << " functions ("
          << static_cast<int>(keep_funcs.size() - flib.num_functions()) << ")";

  return lib;
}

// Push all constant inputs of an instantiating node into the function body.
Status PushDownConstInputs(const NodeDef& func_node,
                           const FunctionOptimizerContext& ctx,
                           GrapplerFunctionItem* item,
                           std::unordered_set<string>* const_inputs,
                           std::unordered_set<string>* control_deps) {
  // Record node control dependencies in the control_deps set.
  const auto record_control_deps = [&](const NodeDef* const_input) {
    for (int i = const_input->input_size() - 1; i >= 0; --i) {
      const string& input = const_input->input(i);
      if (IsControlInput(input))
        control_deps->insert(input);
      else
        break;
    }
  };

  for (int i = func_node.input_size() - 1; i >= 0; --i) {
    const string& input = func_node.input(i);
    if (IsControlInput(input)) continue;

    const string node_name = NodeName(input);
    if (ctx.IsTrulyConst(node_name)) {
      VLOG(3) << "Push const into function body: input=" << input;
      const auto* const_input = CHECK_NOTNULL(ctx.TrulyConstNode(node_name));
      const_inputs->insert(input);
      record_control_deps(const_input);
      TF_RETURN_IF_ERROR(ReplaceInputWithConst(*const_input, i, item));
    }
  }

  return Status::OK();
}

// Remove inputs that were pushed into the function body, and attach their
// control dependencies to the function caller node.
void RemovePushedDownConstInputs(const std::unordered_set<string>& const_inputs,
                                 const std::unordered_set<string>& control_deps,
                                 NodeDef* specialized_func_node) {
  // Nothing to do if it was no const inputs to the function node.
  if (const_inputs.empty()) return;

  // Keep only non-const inputs.
  std::vector<string> keep_inputs;
  const auto& inputs = specialized_func_node->input();
  std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(keep_inputs),
               [&](const string& input) {
                 return const_inputs.find(input) == const_inputs.end();
               });

  specialized_func_node->clear_input();
  for (const auto& keep : keep_inputs) specialized_func_node->add_input(keep);

  // Attach control dependencies of pushed down const input to the caller node.
  if (!control_deps.empty()) {
    std::unordered_set<string> existing_control_deps;

    for (const string& input : keep_inputs) {
      existing_control_deps.insert(AsControlDependency(NodeName(input)));
    }

    for (const string& ctrl : control_deps) {
      if (existing_control_deps.find(ctrl) == existing_control_deps.end()) {
        VLOG(3) << "Forward control dependency: input=" << ctrl;
        specialized_func_node->add_input(ctrl);
      }
    }
  }
}

Status InitializeFunctionSpecializationSignature(
    const NodeDef& func_node, const FunctionDef& func,
    const AttrValueMap& func_attr, const FunctionOptimizerContext& ctx,
    FunctionSpecializationSignature* sig) {
  sig->func_name = func.signature().name();

  TF_RETURN_IF_ERROR(
      InstantiationTypeParameters(func, func_attr, &sig->type_parameters));
  TF_RETURN_IF_ERROR(
      InstantiationBodyParameters(func, func_attr, &sig->body_parameters));

  for (int i = 0; i < func_node.input_size(); ++i) {
    const string& input = func_node.input(i);
    if (ctx.IsTrulyConst(input)) {
      sig->const_inputs.emplace(i, input);
    }
  }

  return Status::OK();
}

Status SpecializeFunction(const NodeDef& func_node, const FunctionDef& func,
                          FunctionOptimizerContext* ctx,
                          GraphDef* optimized_graph) {
  VLOG(2) << "Specialize function instantiation: "
          << SummarizeNodeDef(func_node);

  const std::unordered_map<string, AttrValue> func_attr(
      func_node.attr().begin(), func_node.attr().end());

  FunctionSpecializationSignature signature;
  TF_RETURN_IF_ERROR(InitializeFunctionSpecializationSignature(
      func_node, func, func_attr, *ctx, &signature));

  // Check if function was already specialized for identical context.
  const FunctionSpecialization* already_specialized =
      ctx->FindFunctionSpecialization(signature);

  if (already_specialized) {
    VLOG(2) << "Function was already specialized in identical context: "
               "specialized_name="
            << already_specialized->specialized_func_name;

    // Add a function call node for the specialized function.
    NodeDef* specialized_func_node = optimized_graph->add_node();
    *specialized_func_node = func_node;
    specialized_func_node->set_op(already_specialized->specialized_func_name);

    RemovePushedDownConstInputs(already_specialized->const_inputs,
                                already_specialized->control_deps,
                                specialized_func_node);

    return Status::OK();
  }

  // Add a new specialized function definition to the library.
  const auto& flib = ctx->function_library();

  // Make a GrapplerFunctionItem and convert it back to FunctionDef after
  // pushing all constant inputs into the function body.
  GrapplerFunctionItem item;
  TF_RETURN_IF_ERROR(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  // Push const inputs into the function body, and keep track of their control
  // dependencies.
  std::unordered_set<string> const_inputs;
  std::unordered_set<string> control_deps;
  TF_RETURN_IF_ERROR(PushDownConstInputs(func_node, *ctx, &item, &const_inputs,
                                         &control_deps));

  // TODO(ezhulenev): Push down known input shapes.
  FunctionDef specialized_func;
  TF_RETURN_IF_ERROR(MakeFunctionDef(item, flib, &specialized_func));

  // Find a name for specialized function.
  const string specialized_func_name =
      UniqueSpecializedFunctionName(func, func_node, flib);

  specialized_func.mutable_signature()->set_name(specialized_func_name);
  auto* specialized_attr = specialized_func.mutable_attr();
  (*specialized_attr)[kGrapplerSpecializedFuncAttr].set_b(true);

  // Add specialized function to the library.
  TF_RETURN_IF_ERROR(
      ctx->mutable_function_library()->AddFunctionDef(specialized_func));

  // Add a function call node for the specialized function.
  NodeDef* specialized_func_node = optimized_graph->add_node();
  *specialized_func_node = func_node;
  specialized_func_node->set_op(specialized_func_name);

  // Update specialized node to remove inputs for pushed down consts.
  RemovePushedDownConstInputs(const_inputs, control_deps,
                              specialized_func_node);

  ctx->AddSpecializedFunction(
      signature, {specialized_func_name, const_inputs, control_deps});

  return Status::OK();
}

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
                      const FunctionOptimizerContext& ctx,
                      GraphDef* optimized_graph) {
  VLOG(2) << "Inline function instantiation: " << SummarizeNodeDef(func_node);

  const std::unordered_map<string, AttrValue> func_attr(
      func_node.attr().begin(), func_node.attr().end());

  GrapplerFunctionItem item;
  Status item_status =
      MakeGrapplerFunctionItem(func, func_attr, ctx.function_library(), &item);

  if (!item_status.ok()) {
    return errors::InvalidArgument("Failed to inline function ", func_node.op(),
                                   " instantiated by ", func_node.name(),
                                   ". Error: ", item_status.error_message());
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

  for (NodeDef& func_body_node : *item.mutable_function_body().mutable_node()) {
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
  std::vector<string> fetch = OutputTensors(item);
  TF_RETURN_IF_ERROR(HookInlinedFunctionOutputs(func_node, func, func_attr,
                                                fetch, func_outputs));

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
  VLOG(2) << "Inline symbolic gradient: " << SummarizeNodeDef(node);

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
  VLOG(1) << "Optimize Grappler item: id=" << item.id;

  // Nothing to do here.
  if (item.graph.library().function_size() == 0) {
    VLOG(3) << "Skip Grappler item with empty function library";
    *optimized_graph = item.graph;
    return Status::OK();
  }

  FunctionOptimizerContext ctx(opt_level_, item);
  SymbolicGradientEnv env(item.graph.versions().producer(),
                          item.graph.library());

  bool inline_gradients = options_.enable_symbolic_gradient_inlining;
  bool inline_func = options_.enable_function_inlining;
  bool specialize_func = options_.enable_function_specialization;

  for (const NodeDef& node : item.graph.node()) {
    const string func_name = node.op();

    if (func_name == "SymbolicGradient" && inline_gradients) {
      // Inline symbolic gradients only if the corresponding function is inlined
      const auto* f_attr = gtl::FindOrNull(node.attr(), "f");
      string f_name = f_attr != nullptr ? f_attr->func().name() : "";
      if (ctx.IsInlinedFunction(f_name)) {
        TF_RETURN_IF_ERROR(InlineSymbolicGradient(node, &env, optimized_graph));
        continue;
      }
    }

    const FunctionDef* func = ctx.function_library().Find(func_name);
    if (func != nullptr) {
      if (inline_func && ctx.IsInlinedFunction(func_name)) {
        // Inline function body into the optimized graph}
        TF_RETURN_IF_ERROR(InlineFunction(node, *func, ctx, optimized_graph));
        continue;
      }

      // Do not specialize if function has custom gradient.
      const string grad_func = ctx.function_library().FindGradient(func_name);

      if (specialize_func && grad_func.empty() &&
          (IsParametrized(*func) || HasTrulyConstInputs(node, ctx))) {
        // TODO(ezhulenev): Specialize function call if input has a known shape.
        // Specialize function body for its instantiation attributes and inputs.
        TF_RETURN_IF_ERROR(
            SpecializeFunction(node, *func, &ctx, optimized_graph));
        continue;
      }
    }

    // If we reached this point, node was not handled by any of the stages
    // (inline, specialize), simply add a copy to the graph.
    *optimized_graph->add_node() = node;
  }

  *optimized_graph->mutable_versions() = item.graph.versions();
  *optimized_graph->mutable_library() =
      options_.enable_trim_function_library
          ? TrimFunctionLibrary(ctx.function_library(), *optimized_graph)
          : ctx.function_library().ToProto();

  return Status::OK();
}

void FunctionOptimizer::Feedback(Cluster* cluster, const GrapplerItem& item,
                                 const GraphDef& optimized_graph,
                                 double result) {
  // Nothing to do for FunctionOptimizer.
}

}  // end namespace grappler
}  // end namespace tensorflow
