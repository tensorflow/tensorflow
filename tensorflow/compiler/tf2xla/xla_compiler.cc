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

#include "tensorflow/compiler/tf2xla/xla_compiler.h"

#include <numeric>
#include <vector>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/dump_graph.h"
#include "tensorflow/compiler/tf2xla/graph_compiler.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace {

// Checks that arguments `args` match types `types`.
Status CheckSignature(const DataTypeVector& types,
                      absl::Span<const XlaCompiler::Argument> args) {
  if (args.size() != types.size()) {
    return errors::Internal("Compilation arguments have ", args.size(),
                            " elements while function has ", types.size());
  }
  for (int i = 0; i < types.size(); ++i) {
    if (types[i] != args[i].type && types[i] != DT_RESOURCE) {
      return errors::Internal(
          "Argument ", i, " has declared type ", DataTypeString(args[i].type),
          " but function parameter has type ", DataTypeString(types[i]));
    }
  }
  return Status::OK();
}

// Uses the _Arg and _Retval nodes in the graph to determine a core assignment
// for each argument and return value.
xla::StatusOr<std::pair<std::map<int, int>, std::map<int, int>>>
ComputeArgAndRetvalCores(const Graph& graph) {
  auto get_sharding_for_node = [](const Node* n) -> xla::StatusOr<int> {
    TF_ASSIGN_OR_RETURN(
        auto sharding,
        ParseShardingFromDevice(*n, std::numeric_limits<int32>::max()));
    if (sharding.has_value()) {
      TF_RET_CHECK(sharding.value().type() ==
                   xla::OpSharding::Type::OpSharding_Type_MAXIMAL);
      return sharding.value().tile_assignment_devices(0);
    } else {
      return -1;
    }
  };
  std::map<int, int> arg_cores;
  std::map<int, int> retval_cores;
  for (const Node* n : graph.nodes()) {
    if (n->type_string() == FunctionLibraryDefinition::kArgOp) {
      TF_ASSIGN_OR_RETURN(int core, get_sharding_for_node(n));
      if (core < 0) continue;
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0) << "Negative _Arg index";
      arg_cores[index] = core;
    } else if (n->type_string() == FunctionLibraryDefinition::kRetOp) {
      TF_ASSIGN_OR_RETURN(int core, get_sharding_for_node(n));
      if (core < 0) continue;
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0) << "Negative _Retval index";
      TF_ASSIGN_OR_RETURN(retval_cores[index], get_sharding_for_node(n));
      retval_cores[index] = core;
    }
  }
  return std::make_pair(std::move(arg_cores), std::move(retval_cores));
}

Status ExecuteGraph(XlaContext* xla_context, std::unique_ptr<Graph> graph,
                    XlaCompilationDevice* device, FunctionLibraryRuntime* flib,
                    int64 step_id) {
  // Resource cleanup is a bit messy. XlaContext is a ref-countd resource; the
  // resource manager takes ownership via Create, and unrefs via Cleanup.  We
  // explicitly add a reference to ensure the refcount at entry is maintained at
  // all exit points; Create and Cleanup are always called in this function.
  //
  // The Executor requires us to use ScopedStepContainer. We wrap it in a
  // unique_ptr so we can capture the cleanup status in the end.
  xla_context->Ref();
  Status status;
  auto step_container = absl::make_unique<ScopedStepContainer>(
      step_id, [&status, device](const string& name) {
        status = device->resource_manager()->Cleanup(name);
      });
  TF_RETURN_IF_ERROR(device->resource_manager()->Create(
      step_container->name(), XlaContext::kXlaContextResourceName,
      xla_context));

  GraphCompiler graph_compiler(device, graph.get(), flib, step_container.get());
  TF_RETURN_IF_ERROR(graph_compiler.Compile());
  // Explicitly clean up the step container, to capture the cleanup status.
  step_container.reset();
  return Status::OK();
}

// Builds the XLA computation.
// - `args` is the list of input arguments
// - `retvals` is the list of retvals produced by _Retval operators, in index
//   order.
// - `args_core` and `retval_cores` are mapping from arg/return indices to core
//   assignments.
// - If `return_updated_values_for_all_resources` is true, all resources will be
//   included in `resource_updates`, regardless of whether their value changed.
// - Sets `*num_nonconst_outputs` to the number of outputs of the `computation`.
// - Sets `*resource_updates` to a description of resources whose values are
//   written by the computation; the variable writes are the last
// - `resource_updates.size()` return values from the computation. Each entry in
//   `resource_updates` is a ResourceUpdate, whose `index` is the index of a
//   resource variable argument to the computation to be updated, and `type` is
//   the type of the final output.
Status BuildComputation(
    const std::vector<XlaCompiler::Argument>& args,
    const std::vector<XlaExpression>& retvals,
    const std::map<int, int>& arg_cores, const std::map<int, int>& retval_cores,
    const std::vector<std::unique_ptr<XlaResource>>& resources,
    std::unique_ptr<xla::XlaOp> token_output,
    const XlaCompiler::ShapeRepresentationFn& shape_representation_fn,
    bool return_updated_values_for_all_resources, bool always_return_tuple,
    xla::XlaBuilder* builder, xla::XlaComputation* computation,
    int* num_computation_outputs, int* num_nonconst_outputs,
    std::vector<XlaCompiler::OutputDescription>* outputs,
    std::vector<XlaCompiler::ResourceUpdate>* resource_updates,
    xla::Shape* output_shape) {
  // Attach a common operator name as metadata. This has no semantic effect — it
  // merely makes the HLO graph more readable when visualized via TensorBoard,
  // since TensorBoard forms groups out of operators with similar names.
  xla::OpMetadata retval_metadata;
  retval_metadata.set_op_name("XLA_Retvals");
  builder->SetOpMetadata(retval_metadata);
  auto cleanup = gtl::MakeCleanup([builder]() { builder->ClearOpMetadata(); });

  // Builds a no-op XLA computation. We need to set the sharding of outputs, but
  // cannot change the sharding of the existing output op. To do this, we build
  // a new identity op to which shardings can be applied.
  auto identity_op = [builder](xla::XlaOp op) {
    return xla::GetTupleElement(xla::Tuple(builder, {op}), 0);
  };

  std::vector<xla::XlaOp> elems;
  elems.reserve(retvals.size());

  // Keeps track of which retvals have layout to update. The first element is
  // the output index, second element is the new layout.
  std::vector<std::pair<int64, xla::Layout>> retval_to_update_layout;
  for (int i = 0; i < retvals.size(); ++i) {
    XlaCompiler::OutputDescription& output = (*outputs)[i];
    const XlaExpression& retval = retvals[i];
    output.type = retval.dtype();
    switch (retval.kind()) {
      case XlaExpression::Kind::kConstant:
        output.is_constant = true;
        output.constant_value = retval.constant_value();
        output.shape = output.constant_value.shape();
        break;

      case XlaExpression::Kind::kTensorList:
        TF_FALLTHROUGH_INTENDED;
      case XlaExpression::Kind::kXlaOp: {
        output.is_constant = false;
        TF_ASSIGN_OR_RETURN(output.shape, retval.GetShape());
        xla::XlaOp value = retval.handle();
        auto it = retval_cores.find(i);
        xla::XlaScopedShardingAssignment assign_sharding(
            builder, it == retval_cores.end()
                         ? absl::optional<xla::OpSharding>()
                         : xla::sharding_builder::AssignDevice(it->second));
        if (shape_representation_fn) {
          // If there is a shape representation function, reshape the output
          // tensor to the shape given by the representation shape function.
          TF_ASSIGN_OR_RETURN(xla::Shape shape, shape_representation_fn(
                                                    output.shape, output.type));
          value = xla::Reshape(value, xla::AsInt64Slice(shape.dimensions()));
          retval_to_update_layout.emplace_back(elems.size(), shape.layout());
        } else if (it != retval_cores.end()) {
          // Apply the sharding to the output, if there is a core assignment.
          value = identity_op(value);
        }

        elems.push_back(value);
        break;
      }

      case XlaExpression::Kind::kResource:
        output.is_constant = false;
        output.input_index = retval.resource()->arg_num();
        output.shape = retval.resource()->shape();
        break;

      case XlaExpression::Kind::kInvalid:
        return errors::InvalidArgument(
            "Invalid expression returned by computation. "
            "This probably means a return value was not set.");
    }
  }
  *num_nonconst_outputs = elems.size();

  // Add return values for resources whose values have changed.
  std::vector<const XlaResource*> arg_resources;
  arg_resources.reserve(resources.size());
  for (const auto& resource : resources) {
    if (resource->arg_num() >= 0) {
      arg_resources.push_back(resource.get());
    }
  }
  std::sort(arg_resources.begin(), arg_resources.end(),
            [](const XlaResource* a, const XlaResource* b) {
              return a->arg_num() < b->arg_num();
            });

  for (const XlaResource* resource : arg_resources) {
    DCHECK_LT(resource->arg_num(), args.size());
    const XlaCompiler::Argument& arg = args[resource->arg_num()];
    auto it = arg_cores.find(resource->arg_num());
    const int core = it == arg_cores.end() ? -1 : it->second;
    bool modified = !resource->value().IsIdenticalTo(resource->initial_value());
    // TensorArray gradients were modified if their values changed or there are
    // any newly created gradients.
    for (const auto& grad : resource->tensor_array_gradients()) {
      modified =
          modified ||
          !grad.second->value().IsIdenticalTo(grad.second->initial_value()) ||
          arg.tensor_array_gradients.count(grad.first) == 0;
    }
    if (return_updated_values_for_all_resources || modified) {
      resource_updates->emplace_back();
      XlaCompiler::ResourceUpdate& update = resource_updates->back();
      update.input_index = resource->arg_num();
      update.type = resource->type();
      update.shape = resource->shape();
      update.modified = modified;
      for (const auto& grad : resource->tensor_array_gradients()) {
        update.tensor_array_gradients_accessed.insert(grad.first);
      }

      // Request that the value be returned on a specific core.
      xla::XlaScopedShardingAssignment assign_sharding(
          builder, core == -1 ? absl::optional<xla::OpSharding>()
                              : xla::sharding_builder::AssignDevice(core));

      xla::XlaOp handle;
      TF_RETURN_IF_ERROR(resource->Pack(&handle, builder));

      // Ensures the correct sharding is applied to the output.
      handle = identity_op(handle);

      elems.push_back(handle);
    }
  }

  // If we have token output, append it as the last one.
  if (token_output) {
    elems.push_back(*token_output);
  }

  *num_computation_outputs = elems.size();

  // Builds the XLA computation. We *always* form a tuple here to ensure that
  // the output value is the last thing added into the XLA computation, even
  // if there is only one output value.
  auto tuple = xla::Tuple(builder, elems);
  if (!always_return_tuple && elems.size() == 1) {
    xla::GetTupleElement(tuple, 0);
  }

  xla::StatusOr<xla::XlaComputation> computation_status = builder->Build();
  if (!computation_status.ok()) {
    return computation_status.status();
  }
  *computation = computation_status.ConsumeValueOrDie();

  TF_ASSIGN_OR_RETURN(const auto& program_shape,
                      computation->GetProgramShape());
  *output_shape = program_shape.result();
  // Update the output layout to the layout of retval.
  for (auto& update : retval_to_update_layout) {
    if (!always_return_tuple && elems.size() == 1) {
      *output_shape->mutable_layout() = update.second;
      continue;
    }

    xla::Shape* output_sub_shape =
        xla::ShapeUtil::GetMutableSubshape(output_shape, {update.first});
    *output_sub_shape->mutable_layout() = update.second;
  }
  return Status::OK();
}

}  // namespace

bool XlaCompiler::Argument::operator==(
    const XlaCompiler::Argument& other) const {
  if (std::tie(kind, resource_kind, type, name, initialized, max_array_size,
               tensor_array_gradients) !=
      std::tie(other.kind, other.resource_kind, other.type, other.name,
               other.initialized, other.max_array_size,
               other.tensor_array_gradients)) {
    return false;
  }
  if (absl::holds_alternative<xla::Shape>(shape)) {
    if (!absl::holds_alternative<xla::Shape>(other.shape)) {
      return false;
    }
    if (!xla::Shape::Equal()(absl::get<xla::Shape>(shape),
                             absl::get<xla::Shape>(other.shape))) {
      return false;
    }
  } else {
    if (!absl::holds_alternative<TensorShape>(other.shape)) {
      return false;
    }
    if (absl::get<TensorShape>(shape) != absl::get<TensorShape>(other.shape)) {
      return false;
    }
  }
  if (constant_value.shape() != other.constant_value.shape()) {
    return false;
  }
  return constant_value.tensor_data() == other.constant_value.tensor_data();
}

string XlaCompiler::Argument::HumanString() const {
  string common;
  if (!name.empty()) {
    common = absl::StrCat(" name=", name);
  }
  absl::StrAppend(&common, " type=", DataTypeString(type),
                  " shape=", ShapeHumanString());
  switch (kind) {
    case kInvalid:
      return "invalid";
    case kConstant:
      return absl::StrCat("kind=constant", common,
                          " value=", constant_value.DebugString());
    case kResource: {
      string output = absl::StrCat("kind=resource", common, " resource_kind=",
                                   XlaResource::KindToString(resource_kind),
                                   " initialized=", initialized);
      if (max_array_size >= 0) {
        absl::StrAppend(&output, " max_array_size=", max_array_size);
      }
      if (!tensor_array_gradients.empty()) {
        absl::StrAppend(&output, " tensor_array_gradients=",
                        absl::StrJoin(tensor_array_gradients, ","));
      }
      return output;
    }
    case kParameter:
      return absl::StrCat("kind=parameter", common);
    case kToken:
      return absl::StrCat("token", common);
  }
}

std::vector<int64> XlaCompiler::Argument::DimensionSizes() const {
  if (absl::holds_alternative<TensorShape>(shape)) {
    return xla::InlinedVectorToVector(
        absl::get<TensorShape>(shape).dim_sizes());
  } else {
    return absl::get<xla::Shape>(shape).dimensions();
  }
}

string XlaCompiler::Argument::ShapeHumanString() const {
  if (absl::holds_alternative<TensorShape>(shape)) {
    return absl::get<TensorShape>(shape).DebugString();
  } else {
    return absl::get<xla::Shape>(shape).DebugString();
  }
}

XlaCompiler::XlaCompiler(XlaCompiler::Options options)
    : options_(options),
      initialization_status_(Status::OK()),
      next_step_id_(1),
      device_(new XlaCompilationDevice(SessionOptions(), options_.device_type)),
      device_mgr_(absl::WrapUnique(device_)) {
  CHECK(!options_.device_type.type_string().empty());
  if (options_.populate_resource_manager) {
    initialization_status_ =
        (*options_.populate_resource_manager)(device_->resource_manager());
  }

  local_flib_def_.reset(new FunctionLibraryDefinition(OpRegistry::Global(),
                                                      FunctionDefLibrary{}));
  local_pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), options.graph_def_version,
      local_flib_def_.get(), OptimizerOptions(),
      nullptr /* custom_kernel_creator */));
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), options.graph_def_version, options.flib_def,
      OptimizerOptions(), nullptr /* custom_kernel_creator */));

  local_flib_runtime_ = local_pflr_->GetFLR(device_->name());
  flib_runtime_ = pflr_->GetFLR(device_->name());

  // The default shape representation function is the identity.
  if (!options_.shape_representation_fn) {
    options_.shape_representation_fn =
        [](const TensorShape& shape,
           DataType dtype) -> xla::StatusOr<xla::Shape> {
      xla::Shape xla_shape;
      TF_RETURN_IF_ERROR(TensorShapeToXLAShape(dtype, shape, &xla_shape));
      return xla_shape;
    };
  }
}

XlaCompiler::~XlaCompiler() = default;

int64 XlaCompiler::NextStepId() { return next_step_id_++; }

uint64 XlaCompiler::SignatureHash::operator()(
    const std::pair<string, std::vector<Argument>>& signature) const {
  return std::hash<string>()(signature.first);
}

static Status GetFunctionBody(const NameAttrList& function,
                              FunctionLibraryRuntime* flib_runtime,
                              const FunctionBody** fbody) {
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(flib_runtime->Instantiate(
      function.name(), AttrSlice(&function.attr()), &handle));

  *fbody = flib_runtime->GetFunctionBody(handle);
  TF_RET_CHECK(*fbody);
  return Status::OK();
}

Status XlaCompiler::FindFunctionBody(const NameAttrList& function,
                                     const FunctionBody** fbody) {
  // The function may be in either the local_flib_runtime_ or flib_runtime_.
  // Look up the function in local first and if it is not found then look up the
  // function in flib_runtime_.
  auto status = GetFunctionBody(function, local_flib_runtime_, fbody);
  if (!status.ok()) {
    if (!errors::IsNotFound(status)) {
      return status;
    }
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        GetFunctionBody(function, flib_runtime_, fbody),
        "Local lookup failed with: ", status.error_message());
    VLOG(4) << "Function " << function.name() << " in flib_runtime_";
  } else {
    VLOG(4) << "Function " << function.name() << " in local_flib_runtime_";
  }
  return Status::OK();
}

std::unique_ptr<Graph> XlaCompiler::GetGraph(const FunctionBody* fbody) {
  std::unique_ptr<Graph> graph(new Graph(options_.flib_def));
  CopyGraph(*fbody->graph, graph.get());
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_common_subexpression_elimination(false);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(true);
  GraphOptimizer optimizer(opts);
  // Do not constant fold nodes that output DT_VARIANT type tensors.
  // XLA does not support Const nodes of Variant type since it needs
  // to know the original ops to be able to compile them to the relevant
  // XLA form.
  // TODO(srbs): This filter is a little conservative. E.g. a subgraph of
  // the form:
  //                          Const
  //                            |
  // EmptyTensorList -> TensorListPushBack -> TensorListPopBack -> Op
  //                                                  |
  //                                        (Discard popped list)
  //
  // Would have been reduced to "Const -> Op" without this filter.
  // However since we are only allowed to specify the filter at the "Node"
  // level there is no good way to allow the above behavior. So we
  // disallow any sort of constant folding on Variant nodes for now.
  auto cf_consider_fn = [](const Node* n) {
    for (const auto& output_arg : n->op_def().output_arg()) {
      if (output_arg.type() == DT_VARIANT) {
        return false;
      }
    }
    return true;
  };
  GraphOptimizer::Options graph_optimizer_options;
  graph_optimizer_options.cf_consider_fn = cf_consider_fn;
  optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                     /*device=*/nullptr, &graph, graph_optimizer_options);

  return graph;
}

Status XlaCompiler::CompileFunction(
    const XlaCompiler::CompileOptions& options, const NameAttrList& function,
    absl::Span<const XlaCompiler::Argument> args,
    XlaCompiler::CompilationResult* result) {
  const string function_id =
      Canonicalize(function.name(), AttrSlice(&function.attr()));
  VLOG(1) << "XlaCompiler::CompileFunction " << function_id;

  const std::vector<XlaCompiler::Argument> arg_vector(args.begin(), args.end());
  auto it = cache_.find({function_id, arg_vector});
  if (it != cache_.end()) {
    *result = it->second;
    return Status::OK();
  }

  const FunctionBody* fbody;
  TF_RETURN_IF_ERROR(FindFunctionBody(function, &fbody));

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckSignature(fbody->arg_types, args),
      "Signature check failure while compiling: ", function.name());

  std::unique_ptr<Graph> graph = GetGraph(fbody);

  // Clear the "_kernel" attribute if it is set to "host". This is used to
  // indicate that a computation should happen on the host instead of the
  // accelerator, but doesn't make sense in XLA.
  const char* const kKernelAttr = "_kernel";
  for (Node* n : graph->nodes()) {
    string value;
    if (GetNodeAttrSimple(n->attrs(), kKernelAttr, &value) && value == "host") {
      n->ClearAttr(kKernelAttr);
    }
  }

  // _Arg and _Retval nodes don't exist in the stored subgraph for the function;
  // they are added by the function body looked up.  Therefore, they don't have
  // core assignments here.
  // Attempt to assign a core to each _Retval and _Arg. Chooses the
  // lowest-numbered core that consumes the argument. We choose the
  // lowest-numbered core so the assignment is deterministic.
  for (Node* n : graph->nodes()) {
    if (absl::string_view(n->type_string()) ==
        FunctionLibraryDefinition::kArgOp) {
      TF_RETURN_IF_ERROR(SetNodeShardingFromNeighbors(n, /*out_edges=*/true));
    }
  }
  // Do _Retval as a second loop, in case the retval's input is an _Arg (which
  // may have gotten a device assignment from the first loop).
  for (Node* n : graph->nodes()) {
    if (absl::string_view(n->type_string()) ==
        FunctionLibraryDefinition::kRetOp) {
      TF_RETURN_IF_ERROR(SetNodeShardingFromNeighbors(n, /*out_edges=*/false));
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileFunction: "
            << dump_graph::DumpGraphToFile(
                   absl::StrCat("xla_compile_function_", function_id), *graph);
  }

  VLOG(1) << "====================================================";
  TF_RETURN_IF_ERROR(
      CompileGraph(options, function_id, std::move(graph), args, result));
  VLOG(1) << "====================================================";

  cache_[{function_id, arg_vector}] = *result;
  return Status::OK();
}

// Computes the XLA shape for argument 'arg'.
Status XlaCompiler::XLAShapeForArgument(const XlaCompiler::Argument& arg,
                                        bool is_entry_computation,
                                        xla::Shape* xla_shape) const {
  switch (arg.kind) {
    case XlaCompiler::Argument::kConstant:
      LOG(FATAL) << "Unreachable case";
    case XlaCompiler::Argument::kParameter: {
      if (is_entry_computation) {
        TensorShape shape;
        if (absl::holds_alternative<TensorShape>(arg.shape)) {
          shape = absl::get<TensorShape>(arg.shape);
        } else {
          TF_RETURN_IF_ERROR(
              XLAShapeToTensorShape(absl::get<xla::Shape>(arg.shape), &shape));
        }
        TF_ASSIGN_OR_RETURN(*xla_shape,
                            options_.shape_representation_fn(shape, arg.type));
      } else {
        if (absl::holds_alternative<xla::Shape>(arg.shape)) {
          *xla_shape = absl::get<xla::Shape>(arg.shape);
        } else {
          TF_RETURN_IF_ERROR(TensorShapeToXLAShape(
              arg.type, absl::get<TensorShape>(arg.shape), xla_shape));
        }
      }
      return Status::OK();
    }
    case XlaCompiler::Argument::kResource: {
      TF_RET_CHECK(arg.initialized);

      switch (arg.resource_kind) {
        case XlaResource::kVariable: {
          TF_RET_CHECK(absl::holds_alternative<TensorShape>(arg.shape));
          TF_ASSIGN_OR_RETURN(*xla_shape,
                              options_.shape_representation_fn(
                                  absl::get<TensorShape>(arg.shape), arg.type));

          return Status::OK();
        }
        case XlaResource::kTensorArray: {
          if (arg.max_array_size < 0) {
            return errors::InvalidArgument(
                "Negative max_array_size in XLAShapeForArgument");
          }
          TF_RET_CHECK(absl::holds_alternative<TensorShape>(arg.shape));
          TensorShape shape;
          shape.AddDim(arg.max_array_size);
          shape.AppendShape(absl::get<TensorShape>(arg.shape));
          TF_RETURN_IF_ERROR(TensorShapeToXLAShape(arg.type, shape, xla_shape));

          if (!arg.tensor_array_gradients.empty()) {
            std::vector<xla::Shape> tuple_shape(
                arg.tensor_array_gradients.size() + 1, *xla_shape);
            *xla_shape = xla::ShapeUtil::MakeTupleShape(tuple_shape);
          }
          return Status::OK();
        }
        case XlaResource::kStack: {
          if (arg.max_array_size < 0) {
            return errors::InvalidArgument(
                "Negative max_array_size in XLAShapeForArgument");
          }
          TF_RET_CHECK(absl::holds_alternative<TensorShape>(arg.shape));
          TensorShape shape;
          shape.AddDim(arg.max_array_size);
          shape.AppendShape(absl::get<TensorShape>(arg.shape));
          xla::Shape buffer_shape;
          TF_RETURN_IF_ERROR(
              TensorShapeToXLAShape(arg.type, shape, &buffer_shape));
          *xla_shape = xla::ShapeUtil::MakeTupleShape(
              {buffer_shape, xla::ShapeUtil::MakeShape(xla::S32, {})});
          return Status::OK();
        }

        case XlaResource::kInvalid:
          return errors::Internal(
              "Invalid resource type in XLAShapeForArgument()");
      }
    }
    case XlaCompiler::Argument::kToken: {
      *xla_shape = xla::ShapeUtil::MakeTokenShape();
      return Status::OK();
    }
    case XlaCompiler::Argument::kInvalid:
      return errors::Internal("Invalid argument type in XLAShapeForArgument()");
  }
}

// Builds XLA computations for each of the arguments to the computation.
// `args` are the arguments to the computation.
Status XlaCompiler::BuildArguments(
    const Graph& graph, const std::vector<XlaCompiler::Argument>& args,
    bool use_tuple_arg, xla::XlaBuilder* builder, XlaContext* context,
    const std::map<int, int>& arg_cores,
    std::vector<XlaExpression>* arg_expressions,
    std::vector<int>* input_to_args, std::vector<xla::Shape>* input_shapes,
    bool is_entry_computation) {
  arg_expressions->resize(args.size());

  // Argument numbers of arguments and resources that are to be passed to the
  // XLA computation as runtime parameters. `input_to_args[a] = b` means that
  // the a'th XLA input corresponds to the b'th original arg indexes.
  input_to_args->clear();
  input_to_args->reserve(args.size());

  // Fills in constant arguments, and computes non-constant argument order.
  for (std::vector<XlaCompiler::Argument>::size_type i = 0; i < args.size();
       ++i) {
    const XlaCompiler::Argument& arg = args[i];
    XlaExpression& arg_expression = (*arg_expressions)[i];
    switch (arg.kind) {
      case XlaCompiler::Argument::kResource: {
        TF_RET_CHECK(arg.resource_kind != XlaResource::kInvalid);
        TF_RET_CHECK(absl::holds_alternative<TensorShape>(arg.shape));
        // TODO(phawkins): this code assumes that resource arguments do not
        // alias.
        XlaResource* resource =
            context->AddResource(absl::make_unique<XlaResource>(
                arg.resource_kind, i, arg.name, arg.type,
                absl::get<TensorShape>(arg.shape), xla::XlaOp(),
                /*max_array_size=*/arg.max_array_size,
                /*tensor_array_gradients=*/arg.tensor_array_gradients,
                /*tensor_array_multiple_writes_aggregate=*/true));
        arg_expression = XlaExpression::Resource(resource);
        if (arg.initialized) {
          input_to_args->push_back(i);
        }
        break;
      }
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kToken: {
        input_to_args->push_back(i);
        break;
      }
      case XlaCompiler::Argument::kConstant:
        arg_expression = XlaExpression::Constant(arg.constant_value);
        break;
      case XlaCompiler::Argument::kInvalid:
        return errors::Internal(
            "Unreachable case in BuildArguments() while filling constant args");
    }
  }

  if (input_to_args->empty()) {
    return Status::OK();
  }

  // `arg_to_inputs[c] = d` means that the c'th original arg index corresponds
  // to the d'th XLA input. Note that the value -1 corresponds to constants, or
  // other args that don't correspond to an input.
  std::vector<int> arg_to_inputs(args.size(), -1);
  for (int i = 0; i < input_to_args->size(); i++) {
    arg_to_inputs[input_to_args->at(i)] = i;
  }

  std::vector<xla::Shape> arg_shapes(input_to_args->size());
  for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
    // Computes the shapes of non-constant arguments.
    TF_RETURN_IF_ERROR(XLAShapeForArgument(
        args[(*input_to_args)[i]], is_entry_computation, &arg_shapes[i]));
  }

  if (use_tuple_arg) {
    input_shapes->push_back(xla::ShapeUtil::MakeTupleShape(arg_shapes));
  } else {
    *input_shapes = arg_shapes;
  }

  // Attach a common operator name as metadata. This has no semantic effect — it
  // merely makes the HLO graph more readable when visualized via TensorBoard,
  // since TensorBoard forms groups out of operators with similar names.
  xla::OpMetadata arg_metadata;
  arg_metadata.set_op_name("XLA_Args");
  builder->SetOpMetadata(arg_metadata);

  // Build parameter handles for non-constant arguments.
  std::vector<xla::XlaOp> arg_handles(input_to_args->size());
  if (use_tuple_arg) {
    xla::XlaOp tuple;
    if (is_entry_computation) {
      xla::OpSharding tuple_sharding;
      tuple_sharding.set_type(xla::OpSharding::Type::OpSharding_Type_TUPLE);
      for (int64 parameter : *input_to_args) {
        auto it = arg_cores.find(parameter);
        const int core = it == arg_cores.end() ? 0 : it->second;
        *tuple_sharding.add_tuple_shardings() =
            xla::sharding_builder::AssignDevice(core);
      }
      xla::XlaScopedShardingAssignment assign_tuple_sharding(builder,
                                                             tuple_sharding);
      tuple = xla::Parameter(builder, 0, (*input_shapes)[0], "arg_tuple");
    } else {
      tuple = xla::Parameter(builder, 0, (*input_shapes)[0], "arg_tuple");
    }

    for (int i = 0; i < input_to_args->size(); ++i) {
      const XlaCompiler::Argument& arg = args[input_to_args->at(i)];
      for (const auto& dim_and_arg_num : arg.dynamic_dim_to_arg_num_map) {
        int dynamic_size_param_index = arg_to_inputs.at(dim_and_arg_num.second);
        TF_RETURN_IF_ERROR(builder->SetDynamicBinding(
            /*dynamic_size_param_num=*/0, {dynamic_size_param_index},
            /*target_param_num=*/0, /*target_param_index=*/{i},
            dim_and_arg_num.first));
      }
    }

    for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
      auto it = arg_cores.find(i);
      const int core = it == arg_cores.end() ? -1 : it->second;
      xla::XlaScopedShardingAssignment assign_sharding(
          builder, core == -1 ? absl::optional<xla::OpSharding>()
                              : xla::sharding_builder::AssignDevice(core));
      arg_handles[i] = xla::GetTupleElement(tuple, i);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
      auto it = arg_cores.find(i);
      const int core = it == arg_cores.end() ? -1 : it->second;
      xla::XlaScopedShardingAssignment assign_sharding(
          builder, core == -1 ? absl::optional<xla::OpSharding>()
                              : xla::sharding_builder::AssignDevice(core));
      arg_handles[i] = xla::Parameter(builder, i, (*input_shapes)[i],
                                      absl::StrCat("arg", i));
    }

    for (int i = 0; i < input_to_args->size(); ++i) {
      const XlaCompiler::Argument& arg = args[input_to_args->at(i)];
      for (const auto& dim_and_arg_num : arg.dynamic_dim_to_arg_num_map) {
        int dynamic_size_param_index = arg_to_inputs.at(dim_and_arg_num.second);
        TF_RETURN_IF_ERROR(builder->SetDynamicBinding(
            /*dynamic_size_param_num=*/dynamic_size_param_index, {},
            /*target_param_num=*/i, /*target_param_index=*/{},
            dim_and_arg_num.first));
      }
    }
  }

  builder->ClearOpMetadata();

  // Fill in the handles in non-constant arguments, and reshape parameters
  // back to their correct shapes.
  VLOG(2) << "XLA computation inputs:";
  for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
    const XlaCompiler::Argument& arg = args[input_to_args->at(i)];
    VLOG(2) << "  XLA arg " << i
            << " shape: " << xla::ShapeUtil::HumanString(arg_shapes[i])
            << " name: " << arg.name << " TF arg " << input_to_args->at(i);
    XlaExpression& arg_expression = (*arg_expressions)[input_to_args->at(i)];
    switch (arg.kind) {
      case XlaCompiler::Argument::kResource: {
        TF_RET_CHECK(arg.initialized);
        XlaResource* resource = arg_expression.resource();
        TF_RETURN_IF_ERROR(resource->SetFromPack(arg.tensor_array_gradients,
                                                 arg_handles[i], builder));
        VLOG(2) << "    resource: num_gradients: "
                << arg.tensor_array_gradients.size();
        break;
      }
      case XlaCompiler::Argument::kParameter:
        // Reshape parameters back to their correct shapes.
        // TODO(b/76097077): propagate device assignments onto arguments and
        // return values of functions, and then reshape unconditionally.
        if (is_entry_computation) {
          arg_expression = XlaExpression::XlaOp(
              xla::Reshape(arg_handles[i], arg.DimensionSizes()), arg.type);
        } else {
          arg_expression = XlaExpression::XlaOp(arg_handles[i], arg.type);
        }
        break;
      case XlaCompiler::Argument::kToken: {
        arg_expression = XlaExpression::XlaOp(arg_handles[i], arg.type);
        break;
      }
      case XlaCompiler::Argument::kConstant:
      case XlaCompiler::Argument::kInvalid:
        return errors::Internal(
            "Unreachable case in BuildArguments() while filling handles");
    }
  }

  return Status::OK();
}

Status XlaCompiler::CompileSingleOp(
    const XlaCompiler::CompileOptions& options, const NodeDef& node_def,
    absl::Span<const XlaCompiler::Argument> args,
    absl::Span<const DataType> result_types, CompilationResult* result) {
  // TODO(b/74182462): We implement this by creating a new dummy Graph including
  // _Arg nodes, and let CompileGraph walk it. This could be optimized.
  std::unique_ptr<Graph> graph(new Graph(OpRegistry::Global()));

  Status status;
  // First create the actual node we care about computing.
  Node* main_node = graph->AddNode(node_def, &status);
  TF_RETURN_IF_ERROR(status);

  // Create dummy _Arg nodes. Link these to `node` and also via a control
  // dependency edge to the _SOURCE node.
  for (int64 i = 0; i < args.size(); ++i) {
    Node* node;
    string arg_name = absl::StrCat("_arg", i);
    Status status =
        NodeBuilder(arg_name, FunctionLibraryDefinition::kArgOp)
            .ControlInput(graph->source_node())
            .Attr("T", args[i].kind == Argument::kResource ? DT_RESOURCE
                                                           : args[i].type)
            .Attr("index", i)
            .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
    graph->AddEdge(node, 0, main_node, i);
  }

  // Similarly with return values, create dummy _Retval nodes fed by `node`.
  for (int64 i = 0; i < result_types.size(); ++i) {
    Node* node;
    string retval_name = absl::StrCat("_retval", i);
    Status status = NodeBuilder(retval_name, FunctionLibraryDefinition::kRetOp)
                        .Input(main_node, i)
                        .Attr("T", result_types[i])
                        .Attr("index", i)
                        .Finalize(graph.get(), &node);
    TF_RETURN_IF_ERROR(status);
  }
  FixupSourceAndSinkEdges(graph.get());

  return CompileGraph(options, node_def.name(), std::move(graph), args, result);
}

namespace {

// Check that the ops of all non-functional nodes have been registered.
Status ValidateFunctionDef(const FunctionDef* fdef,
                           const FunctionLibraryDefinition& flib_def) {
  for (const NodeDef& node : fdef->node_def()) {
    const string& op = node.op();
    if (op == FunctionLibraryDefinition::kGradientOp || flib_def.Find(op)) {
      continue;
    }
    const OpDef* op_def;
    TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(op, &op_def));
  }
  return Status::OK();
}

// Check that the graph doesn't have any invalid nodes (e.g. incompatible with
// given device_type, invalid data type, missing attributes...)
Status ValidateGraph(const Graph* graph,
                     const FunctionLibraryDefinition& flib_def,
                     const DeviceType& device_type, const string& name) {
  auto maybe_error = [&](const Node* node, const Status& s) -> Status {
    if (!s.ok()) {
      return errors::InvalidArgument(absl::StrCat(
          "Detected unsupported operations when trying to compile graph ", name,
          " on ", device_type.type_string(), ": ", node->def().op(), " (",
          s.error_message(), ")", FormatNodeForError(*node)));
    }
    return Status::OK();
  };

  for (const Node* node : graph->nodes()) {
    if (node->type_string() == FunctionLibraryDefinition::kGradientOp) {
      continue;
    }
    const FunctionDef* fdef = flib_def.Find(node->def().op());
    Status s;
    if (fdef) {
      s = ValidateFunctionDef(fdef, flib_def);
      TF_RETURN_IF_ERROR(maybe_error(node, s));
      continue;
    }
    const OpDef* op_def;
    s = OpRegistry::Global()->LookUpOpDef(node->def().op(), &op_def);
    TF_RETURN_IF_ERROR(maybe_error(node, s));
    TF_RETURN_IF_ERROR(ValidateNodeDef(node->def(), *op_def));
    s = FindKernelDef(device_type, node->def(), nullptr, nullptr);
    TF_RETURN_IF_ERROR(maybe_error(node, s));
  }
  return Status::OK();
}

// Converts the value of any expressions whose values are known at compile-time
// to constants.
Status ResolveConstantExpressionsToConstants(
    xla::Client* client, absl::Span<XlaExpression> expressions) {
  for (XlaExpression& expression : expressions) {
    if (expression.kind() == XlaExpression::Kind::kXlaOp) {
      TF_ASSIGN_OR_RETURN(absl::optional<Tensor> constant,
                          expression.ResolveConstant(client));
      if (constant.has_value()) {
        expression = XlaExpression::Constant(*constant);
      }
    }
  }
  return Status::OK();
}

void ConvertConstantsToExpressions(xla::XlaBuilder* builder,
                                   absl::Span<XlaExpression> expressions) {
  for (XlaExpression& expression : expressions) {
    if (expression.kind() == XlaExpression::Kind::kConstant) {
      expression =
          XlaExpression::XlaOp(expression.AsXlaOp(builder), expression.dtype());
    }
  }
}

}  // namespace

Status XlaCompiler::CompileGraph(const XlaCompiler::CompileOptions& options,
                                 string const& name,
                                 std::unique_ptr<Graph> graph,
                                 absl::Span<const XlaCompiler::Argument> args,
                                 CompilationResult* result) {
  VLOG(1) << "Executing graph symbolically to populate XlaBuilder.";

  TF_RETURN_IF_ERROR(PropagateConstIntoFunctionalNodes(
      graph.get(), options_.flib_def, local_flib_def_.get()));
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileGraph: "
            << dump_graph::DumpGraphToFile(
                   absl::StrCat("xla_compile_graph_", name), *graph,
                   flib_runtime_->GetFunctionLibraryDefinition());
  }

  // Report the error here if initialization failed.
  TF_RETURN_IF_ERROR(initialization_status_);

  // Detect invalid nodes.
  // FunctionalizeControlFlow may remove some nodes from the graph.
  TF_RETURN_IF_ERROR(ValidateGraph(graph.get(), *options_.flib_def,
                                   options_.device_type, name));

  xla::XlaBuilder builder(name);
  XlaContext* context = new XlaContext(this, &builder);
  core::ScopedUnref context_unref(context);

  std::vector<XlaCompiler::Argument> real_args(args.begin(), args.end());
  int token_input_index = -1;
  std::unique_ptr<xla::XlaOp> token_output;
  if (options.add_token_input_output) {
    // Add extra token input.
    token_input_index = real_args.size();

    XlaCompiler::Argument token_arg;
    token_arg.kind = XlaCompiler::Argument::kToken;
    real_args.push_back(token_arg);
  }

  std::map<int, int> arg_cores;
  std::map<int, int> retval_cores;
  TF_ASSIGN_OR_RETURN(std::tie(arg_cores, retval_cores),
                      ComputeArgAndRetvalCores(*graph));

  std::vector<XlaExpression> arg_expressions;
  TF_RETURN_IF_ERROR(BuildArguments(
      *graph, real_args, options.use_tuple_arg, &builder, context, arg_cores,
      &arg_expressions, &result->input_mapping, &result->xla_input_shapes,
      options.is_entry_computation));
  context->set_args(std::move(arg_expressions));

  PushNodeTokenMapping();
  // Use std::set instead of std::unordered_set to ensure determinism.
  std::set<std::string> output_node_token_inputs;
  if (token_input_index != -1) {
    // Original token comes from input.
    auto arg_expression = context->args()[token_input_index];
    TF_RETURN_IF_ERROR(
        SetNodeToken(kXlaTokenArgNodeName, arg_expression.handle()));

    // Calculate token inputs for output token.
    output_node_token_inputs = CalculateTokenInputsForOutputToken(*graph);

    // If there's no side-effecting op in the graph, use token input as token
    // output.
    if (output_node_token_inputs.empty()) {
      output_node_token_inputs.insert(kXlaTokenArgNodeName);
    }
  } else if (options.is_entry_computation) {
    // Original token is manually created.
    if (HasSideEffectingNodes(*graph)) {
      TF_RETURN_IF_ERROR(
          SetNodeToken(kXlaTokenArgNodeName, xla::CreateToken(&builder)));
    }
  }

  TF_RETURN_IF_ERROR(ExecuteGraph(context, std::move(graph), device_,
                                  flib_runtime_, NextStepId()));
  if (token_input_index != -1) {
    // Add extra token output.
    std::vector<xla::XlaOp> token_inputs;
    for (const auto& node_name : output_node_token_inputs) {
      auto token_or = GetNodeToken(node_name);
      TF_RETURN_IF_ERROR(token_or.status());
      token_inputs.push_back(token_or.ValueOrDie());
    }
    token_output.reset(new xla::XlaOp(xla::AfterAll(&builder, token_inputs)));
  }
  TF_RETURN_IF_ERROR(PopNodeTokenMapping());

  int num_nonconst_outputs;
  int num_computation_outputs;
  result->computation = std::make_shared<xla::XlaComputation>();
  result->outputs.resize(context->retvals().size());
  std::vector<XlaExpression> retvals = context->retvals();
  if (options.resolve_compile_time_constants) {
    TF_RETURN_IF_ERROR(ResolveConstantExpressionsToConstants(
        client(), absl::Span<XlaExpression>(retvals)));
  } else {
    ConvertConstantsToExpressions(&builder, absl::Span<XlaExpression>(retvals));
  }
  TF_RETURN_IF_ERROR(BuildComputation(
      real_args, retvals, arg_cores, retval_cores, context->resources(),
      std::move(token_output),
      options.is_entry_computation ? options_.shape_representation_fn
                                   : ShapeRepresentationFn{},
      options.return_updated_values_for_all_resources,
      options.always_return_tuple, &builder, result->computation.get(),
      &num_computation_outputs, &num_nonconst_outputs, &result->outputs,
      &result->resource_updates, &result->xla_output_shape));

  VLOG(2) << "Outputs: total: " << context->retvals().size()
          << " nonconstant: " << num_nonconst_outputs;
  VLOG(2) << "XLA output shape: "
          << xla::ShapeUtil::HumanStringWithLayout(result->xla_output_shape);
  return Status::OK();
}

Status XlaCompiler::GetChannelHandle(const string& key,
                                     xla::ChannelHandle* channel) {
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, client()->CreateChannelHandle());
  }
  *channel = result.first->second;
  VLOG(1) << "Channel: " << key << " " << channel->DebugString();
  return Status::OK();
}

Status XlaCompiler::GetHostToDeviceChannelHandle(const string& key,
                                                 xla::ChannelHandle* channel) {
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second,
                        client()->CreateHostToDeviceChannelHandle());
  }
  *channel = result.first->second;
  VLOG(1) << "Host to device channel: " << key << " " << channel->DebugString();
  return Status::OK();
}

Status XlaCompiler::GetDeviceToHostChannelHandle(const string& key,
                                                 xla::ChannelHandle* channel) {
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second,
                        client()->CreateDeviceToHostChannelHandle());
  }
  *channel = result.first->second;
  VLOG(1) << "Device to host channel: " << key << " " << channel->DebugString();
  return Status::OK();
}

namespace {

void SetTransfer(const string& key, absl::Span<const DataType> types,
                 absl::Span<const TensorShape> shapes,
                 tf2xla::HostTransferMetadata* transfer) {
  transfer->set_key(key);
  CHECK(types.size() == shapes.size());
  for (int i = 0; i < types.size(); ++i) {
    tf2xla::TensorMetadata* metadata = transfer->add_metadata();
    metadata->set_type(types[i]);
    shapes[i].AsProto(metadata->mutable_shape());
  }
}

}  // namespace

Status XlaCompiler::SetDeviceToHostMetadata(
    const string& key, absl::Span<const DataType> types,
    absl::Span<const TensorShape> shapes) {
  if (host_compute_sends_.find(key) != host_compute_sends_.end()) {
    return errors::InvalidArgument(
        "Duplicate calls to SetDeviceToHostMetadata with key ", key);
  }
  tf2xla::HostTransferMetadata& transfer = host_compute_sends_[key];
  SetTransfer(key, types, shapes, &transfer);
  return Status::OK();
}

Status XlaCompiler::GetDeviceToHostShapes(
    const string& key, std::vector<TensorShape>* shapes) const {
  const auto iter = host_compute_sends_.find(key);
  if (iter == host_compute_sends_.end()) {
    return errors::InvalidArgument(
        "No host compute send shapes registered for key ", key);
  }
  shapes->clear();
  for (int i = 0; i < iter->second.metadata_size(); ++i) {
    TensorShape shape(iter->second.metadata(i).shape());
    shapes->push_back(shape);
  }
  return Status::OK();
}

Status XlaCompiler::SetHostToDeviceMetadata(
    const string& key, absl::Span<const DataType> types,
    absl::Span<const TensorShape> shapes) {
  if (host_compute_recvs_.find(key) != host_compute_sends_.end()) {
    return errors::InvalidArgument(
        "Duplicate calls to SetHostToDeviceMetadata with key ", key);
  }
  tf2xla::HostTransferMetadata& transfer = host_compute_recvs_[key];
  SetTransfer(key, types, shapes, &transfer);
  return Status::OK();
}

Status XlaCompiler::GetHostComputeControlDependency(
    const string& host_compute_name, xla::XlaOp* handle) {
  const auto iter = host_compute_control_output_.find(host_compute_name);
  if (iter == host_compute_control_output_.end()) {
    return errors::InvalidArgument(
        "No registered control handle for host compute Op '", host_compute_name,
        "'");
  } else {
    *handle = iter->second;
  }
  return Status::OK();
}

Status XlaCompiler::SetHostComputeControlDependency(
    const string& host_compute_name, const xla::XlaOp& handle) {
  if (host_compute_control_output_.find(host_compute_name) !=
      host_compute_control_output_.end()) {
    return errors::InvalidArgument(
        "Duplicate control handles registered for for host compute Op ",
        host_compute_name);
  }
  host_compute_control_output_[host_compute_name] = handle;
  return Status::OK();
}

void XlaCompiler::PushNodeTokenMapping() {
  node_token_mapping_stack_.emplace(std::map<string, xla::XlaOp>{});
}

Status XlaCompiler::PopNodeTokenMapping() {
  if (node_token_mapping_stack_.empty()) {
    return errors::FailedPrecondition(
        "Calling PopNodeTokenMapping() when node_token_mapping_stack_ is "
        "empty.");
  }
  node_token_mapping_stack_.pop();
  return Status::OK();
}

Status XlaCompiler::SetNodeToken(const string& node_name,
                                 const xla::XlaOp& op) {
  if (node_token_mapping_stack_.empty()) {
    return errors::FailedPrecondition(
        "Calling SetNodeToken() when node_token_mapping_stack_ is "
        "empty.");
  }
  auto insert_result = node_token_mapping_stack_.top().insert({node_name, op});
  if (!insert_result.second) {
    return errors::FailedPrecondition("Token mapping already exists for node ",
                                      node_name);
  }
  return Status::OK();
}

xla::StatusOr<xla::XlaOp> XlaCompiler::GetNodeToken(const string& node_name) {
  if (node_token_mapping_stack_.empty()) {
    return errors::FailedPrecondition(
        "Calling GetNodeToken() when node_token_mapping_stack_ is "
        "empty.");
  }
  auto iter = node_token_mapping_stack_.top().find(node_name);
  if (iter == node_token_mapping_stack_.top().end()) {
    return errors::FailedPrecondition("Cannot find token mapping for node ",
                                      node_name);
  }
  return iter->second;
}

}  // namespace tensorflow
