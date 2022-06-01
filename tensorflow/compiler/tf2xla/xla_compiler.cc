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

#include "tensorflow/compiler/mlir/mlir_bridge_rollout_policy.h"
#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/types/variant.h"
#include "tensorflow/compiler/jit/defs.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/compiler/tf2xla/graph_compiler.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/rearrange_function_argument.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/side_effect_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compilation_device.h"
#include "tensorflow/compiler/tf2xla/xla_context.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/graph_debug_info.pb.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

// Checks that arguments `args` match types `types`.
Status CheckSignature(const DataTypeVector& types,
                      absl::Span<const XlaCompiler::Argument> args) {
  if (args.size() != types.size()) {
    return errors::Internal("Compilation arguments have ", args.size(),
                            " elements while function has ", types.size());
  }
  for (int i = 0, end = types.size(); i < end; ++i) {
    // Don't perform type checks on resource variables and tensor
    // lists (DT_VARIANT) as we have to trick the type system in order to
    // plumb them through. DT_VARIANTS are wrapped in a DT_UINT8 tensor.
    if (types[i] != args[i].type && types[i] != DT_RESOURCE &&
        types[i] != DT_VARIANT) {
      return errors::Internal(
          "Argument ", i, " has declared type ", DataTypeString(args[i].type),
          " but function parameter has type ", DataTypeString(types[i]));
    }
  }
  return OkStatus();
}

// Uses the _Arg and _Retval nodes in the graph to determine an OpSharding for
// each argument and return value.
StatusOr<
    std::pair<std::map<int, xla::OpSharding>, std::map<int, xla::OpSharding>>>
ComputeArgAndRetvalShardings(const Graph& graph) {
  auto get_sharding_for_node =
      [](const Node* n) -> StatusOr<absl::optional<xla::OpSharding>> {
    TF_ASSIGN_OR_RETURN(
        auto sharding,
        ParseShardingFromDevice(*n, std::numeric_limits<int32>::max(),
                                /*add_metadata=*/false));
    return sharding;
  };
  std::map<int, xla::OpSharding> arg_shardings;
  std::map<int, xla::OpSharding> retval_shardings;
  for (const Node* n : graph.nodes()) {
    if (n->IsArg()) {
      TF_ASSIGN_OR_RETURN(auto sharding, get_sharding_for_node(n));
      if (!sharding.has_value()) continue;
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0) << "Negative _Arg index";
      arg_shardings[index] = std::move(*sharding);
    } else if (n->IsRetval()) {
      TF_ASSIGN_OR_RETURN(auto sharding, get_sharding_for_node(n));
      if (!sharding.has_value()) continue;
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(n->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0) << "Negative _Retval index";
      retval_shardings[index] = std::move(*sharding);
    }
  }
  return std::make_pair(std::move(arg_shardings), std::move(retval_shardings));
}

Status ExecuteGraph(XlaContext* xla_context, std::unique_ptr<Graph> graph,
                    XlaCompilationDevice* device, FunctionLibraryRuntime* flib,
                    int64_t step_id) {
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
  TF_RETURN_IF_ERROR(step_container->Create(device->resource_manager(),
                                            XlaContext::kXlaContextResourceName,
                                            xla_context));

  GraphCompiler graph_compiler(device, graph.get(), flib, step_container.get());
  TF_RETURN_IF_ERROR(graph_compiler.Compile());
  // Explicitly clean up the step container, to capture the cleanup status.
  step_container.reset();
  return status;
}

// Builds the XLA computation.
// - `args` is the list of input arguments
// - `retvals` is the list of retvals produced by _Retval operators, in index
//   order.
// - `arg_shardings` and `retval_shardings` are mapping from arg/return indices
//   to sharding.
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
    const std::map<int, xla::OpSharding>& arg_shardings,
    const std::map<int, xla::OpSharding>& retval_shardings,
    const std::vector<std::unique_ptr<XlaResource>>& resources,
    std::unique_ptr<xla::XlaOp> token_output,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns,
    bool is_entry_computation, bool return_updated_values_for_all_resources,
    bool always_return_tuple, bool use_tuple_arg, bool alias_resource_update,
    xla::XlaBuilder* builder, xla::XlaComputation* computation,
    int* num_computation_outputs, int* num_nonconst_outputs,
    std::vector<XlaCompiler::OutputDescription>* outputs,
    std::vector<XlaCompiler::ResourceUpdate>* resource_updates,
    xla::Shape* output_shape, absl::Span<int const> input_mapping) {
  // Attach a common operator name as metadata. This has no semantic effect — it
  // merely makes the HLO graph more readable when visualized via TensorBoard,
  // since TensorBoard forms groups out of operators with similar names.
  xla::OpMetadata retval_metadata;
  retval_metadata.set_op_name("XLA_Retvals");
  builder->SetOpMetadata(retval_metadata);
  VLOG(1) << "Building new computation";
  auto cleanup = gtl::MakeCleanup([builder]() { builder->ClearOpMetadata(); });

  // Builds a no-op XLA computation. We need to set the sharding of outputs, but
  // cannot change the sharding of the existing output op. To do this, we build
  // a new identity op to which shardings can be applied.
  auto identity_op = [builder](
                         xla::XlaOp op,
                         const absl::optional<xla::OpSharding>& sharding) {
    xla::XlaScopedShardingAssignment assign_sharding(builder, sharding);
    return xla::Copy(op);
  };

  std::vector<xla::XlaOp> elems;
  elems.reserve(retvals.size());

  // Keeps track of sharding of each retval. If a retval is not in this list,
  // replicate sharding is used. The first element is the output index, second
  // element is the sharding.
  std::unordered_map<int, xla::OpSharding> retval_index_and_sharding;
  for (int i = 0, end = retvals.size(); i < end; ++i) {
    XlaCompiler::OutputDescription& output = (*outputs)[i];
    const XlaExpression& retval = retvals[i];
    output.type = retval.dtype();
    switch (retval.kind()) {
      case XlaExpression::Kind::kConstant:
        output.is_constant = true;
        output.constant_value = *retval.constant_value();
        output.shape = output.constant_value.shape();
        break;

      case XlaExpression::Kind::kTensorList: {
        output.is_tensor_list = true;
        xla::XlaOp value = retval.handle();
        elems.push_back(value);
        break;
      }

      case XlaExpression::Kind::kXlaOp: {
        output.is_constant = false;
        TF_ASSIGN_OR_RETURN(output.shape, retval.GetShape());
        xla::XlaOp value = retval.handle();
        auto it = retval_shardings.find(i);
        absl::optional<xla::OpSharding> sharding =
            it == retval_shardings.end() ? absl::optional<xla::OpSharding>()
                                         : it->second;
        if (it != retval_shardings.end()) {
          retval_index_and_sharding[elems.size()] = it->second;
        }
        if (shape_determination_fns.shape_representation_fn) {
          TF_ASSIGN_OR_RETURN(auto original_shape, builder->GetShape(value));
          TF_ASSIGN_OR_RETURN(value,
                              ReshapeWithCorrectRepresentationAndSharding(
                                  builder, value, original_shape,
                                  shape_determination_fns, sharding,
                                  /*fast_mem=*/false));
        }
        if (it != retval_shardings.end()) {
          // Apply the sharding to the output, if there is a core assignment.
          value = identity_op(value, sharding);
        }

        elems.push_back(value);
        break;
      }

      case XlaExpression::Kind::kResource:
        // Resources will be pushed into elems later when processing resource
        // arguments below.
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

  absl::flat_hash_map<int, int> argument_to_xla_arg;
  for (int xla_arg = 0; xla_arg < input_mapping.size(); xla_arg++) {
    argument_to_xla_arg[input_mapping[xla_arg]] = xla_arg;
  }

  std::vector<xla::XlaBuilder::InputOutputAlias> aliases;
  for (const XlaResource* resource : arg_resources) {
    DCHECK_LT(resource->arg_num(), args.size());
    const XlaCompiler::Argument& arg = args[resource->arg_num()];
    auto it = arg_shardings.find(resource->arg_num());
    bool modified = !resource->value().IsIdenticalTo(resource->initial_value());
    // TensorArray gradients were modified if their values changed or there are
    // any newly created gradients.
    for (const auto& grad : resource->tensor_array_gradients()) {
      modified =
          modified ||
          !grad.second->value().IsIdenticalTo(grad.second->initial_value()) ||
          arg.tensor_array_gradients.count(grad.first) == 0;
    }

    if (return_updated_values_for_all_resources || modified ||
        arg.requires_broadcast) {
      resource_updates->emplace_back();
      XlaCompiler::ResourceUpdate& update = resource_updates->back();
      update.input_index = resource->arg_num();
      update.type = resource->type();
      update.shape = resource->shape();
      update.modified = modified;
      int param_num = use_tuple_arg ? 0 : update.input_index;
      if (is_entry_computation &&
          arg.resource_kind != XlaResource::kTensorArray &&
          alias_resource_update && argument_to_xla_arg.count(param_num)) {
        // Assuming tuple arg and results are used.
        xla::ShapeIndex param_index =
            use_tuple_arg ? xla::ShapeIndex({update.input_index})
                          : xla::ShapeIndex{};
        int xla_param_num = argument_to_xla_arg[param_num];
        int64_t output_index_num = elems.size();
        xla::ShapeIndex output_index = xla::ShapeIndex({output_index_num});
        VLOG(3) << "Storing alias: " << output_index.ToString() << ": ("
                << xla_param_num << ", " << param_index.ToString() << ")";
        aliases.push_back({output_index, xla_param_num, param_index});
      }
      for (const auto& grad : resource->tensor_array_gradients()) {
        update.tensor_array_gradients_accessed.insert(grad.first);
      }

      xla::XlaOp handle;
      TF_RETURN_IF_ERROR(resource->Pack(&handle, builder));
      auto sharding = it == arg_shardings.end()
                          ? absl::optional<xla::OpSharding>()
                          : it->second;
      // Set layout of the retval to device representation layout.
      if (shape_determination_fns.layout_preference_fn &&
          shape_determination_fns.shape_representation_fn) {
        TF_ASSIGN_OR_RETURN(auto original_shape, builder->GetShape(handle));
        TF_ASSIGN_OR_RETURN(
            handle, ReshapeWithCorrectRepresentationAndSharding(
                        builder, handle, original_shape,
                        shape_determination_fns, sharding, arg.fast_mem));
      }

      // Request that the value be returned on a specific core.
      if (it != arg_shardings.end()) {
        retval_index_and_sharding[elems.size()] = it->second;
      }
      // Ensures the correct sharding is applied to the output.
      handle = identity_op(handle, sharding);
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
  xla::XlaOp tuple;
  if (retval_index_and_sharding.empty() || !is_entry_computation) {
    tuple = xla::Tuple(builder, elems);
  } else {
    std::vector<xla::Shape> elem_shapes;
    for (const auto& elem : elems) {
      TF_ASSIGN_OR_RETURN(xla::Shape elem_shape,
                          elem.builder()->GetShape(elem));
      elem_shapes.push_back(elem_shape);
    }
    xla::Shape shape = xla::ShapeUtil::MakeTupleShape(elem_shapes);
    // Copy specified sharding from retval_index_and_sharding.
    std::vector<xla::HloSharding> sharding_elems;
    for (int i = 0, end = elems.size(); i < end; i++) {
      const auto& iter = retval_index_and_sharding.find(i);
      TF_RET_CHECK(iter != retval_index_and_sharding.end());
      const xla::OpSharding& sub_op_sharding = iter->second;
      TF_ASSIGN_OR_RETURN(xla::HloSharding sub_sharding,
                          xla::HloSharding::FromProto(sub_op_sharding));
      if (elem_shapes[i].IsTuple()) {
        const std::vector<xla::HloSharding> sub_sharding_elems =
            sub_sharding.tuple_elements();
        const int64_t sub_sharding_elems_size = sub_sharding_elems.size();
        TF_RET_CHECK(sub_sharding_elems_size ==
                     xla::ShapeUtil::GetLeafCount(elem_shapes[i]));
        for (const auto& sub_sharding_elem : sub_sharding_elems) {
          sharding_elems.push_back(sub_sharding_elem);
        }
      } else {
        sharding_elems.push_back(sub_sharding);
      }
    }
    xla::HloSharding modified_sharding =
        xla::HloSharding::Tuple(shape, sharding_elems);
    xla::OpSharding op_sharding = modified_sharding.ToProto();
    // Assign proper sharding to the tuple instruction.
    xla::XlaScopedShardingAssignment assign_sharding(builder, op_sharding);
    tuple = xla::Tuple(builder, elems);
  }
  bool returns_tuple = always_return_tuple || elems.size() != 1;
  VLOG(3) << "Computation returns a tuple=" << returns_tuple;
  if (!returns_tuple) {
    xla::GetTupleElement(tuple, 0);

    for (xla::XlaBuilder::InputOutputAlias& alias : aliases) {
      if (alias.output_index == xla::ShapeIndex({0})) {
        VLOG(3) << "For aliased parameter " << alias.param_number << ": "
                << alias.param_index.ToString()
                << " normalizing output_index from {0} to {}, as a scalar is "
                   "returned from the cluster";
        alias.output_index = xla::ShapeIndex({});
      }
    }
  }

  for (xla::XlaBuilder::InputOutputAlias& alias : aliases) {
    builder->SetUpAlias(alias.output_index, alias.param_number,
                        alias.param_index);
  }

  StatusOr<xla::XlaComputation> computation_status = builder->Build();
  if (!computation_status.ok()) {
    return computation_status.status();
  }
  *computation = computation_status.ConsumeValueOrDie();

  TF_ASSIGN_OR_RETURN(auto program_shape, computation->GetProgramShape());
  *output_shape = program_shape.result();
  return OkStatus();
}

}  // namespace


string XlaCompiler::Argument::HumanString() const {
  string common;
  if (!name.empty()) {
    common = absl::StrCat(" name=", name);
  }
  absl::StrAppend(&common, " type=", DataTypeString(type),
                  " shape=", ShapeHumanString());
  absl::StrAppend(
      &common, " is_same_data_across_replicas=", is_same_data_across_replicas);
  switch (kind) {
    case kInvalid:
      return "invalid";
    case kConstant:
      return absl::StrCat("kind=constant", common,
                          " value=", constant_value.DebugString());
    case kConstantResource:
      return absl::StrCat("kind=constant-resource", common,
                          " value=", constant_value.DebugString());
    case kResource: {
      string output = absl::StrCat(
          "kind=resource", common,
          " resource_kind=", XlaResource::KindToString(resource_kind),
          " initialized=", initialized, " is_fast_mem=", fast_mem);
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
    case kTensorList:
      return absl::StrCat("kind=tensorlist", common);
    case kToken:
      return absl::StrCat("token", common);
  }
}

std::vector<int64_t> XlaCompiler::Argument::DimensionSizes() const {
  if (absl::holds_alternative<TensorShape>(shape)) {
    return xla::InlinedVectorToVector(
        absl::get<TensorShape>(shape).dim_sizes());
  } else {
    return xla::SpanToVector(absl::get<xla::Shape>(shape).dimensions());
  }
}

absl::InlinedVector<int64_t, 4>
XlaCompiler::Argument::DimensionSizesAsInlinedVector() const {
  if (absl::holds_alternative<TensorShape>(shape)) {
    return absl::get<TensorShape>(shape).dim_sizes();
  } else {
    auto v = absl::get<xla::Shape>(shape).dimensions();
    return absl::InlinedVector<int64_t, 4>(v.begin(), v.end());
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
      initialization_status_(OkStatus()),
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
      &device_mgr_, Env::Default(), /*config=*/nullptr,
      options.graph_def_version, local_flib_def_.get(), OptimizerOptions()));
  pflr_.reset(new ProcessFunctionLibraryRuntime(
      &device_mgr_, Env::Default(), /*config=*/nullptr,
      options.graph_def_version, options.flib_def, OptimizerOptions()));

  local_flib_runtime_ = local_pflr_->GetFLR(device_->name());
  flib_runtime_ = pflr_->GetFLR(device_->name());

  // The default layout preference is no preference and the default shape
  // representation function is the identity.
  XlaShapeLayoutHelpers::ShapeDeterminationFns& shape_determination_fns =
      options_.shape_determination_fns;
  if (!shape_determination_fns.shape_representation_fn) {
    shape_determination_fns.shape_representation_fn =
        IdentityShapeRepresentationFn();
  }
  if (!shape_determination_fns.layout_preference_fn) {
    shape_determination_fns.layout_preference_fn = UseNoPreferenceLayoutFn();
  }
}

XlaCompiler::~XlaCompiler() = default;

int64_t XlaCompiler::NextStepId() { return next_step_id_++; }

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
  return OkStatus();
}

Status XlaCompiler::FindFunctionBody(const NameAttrList& function,
                                     const FunctionBody** fbody,
                                     const ConfigProto** config_proto) {
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
    if (config_proto) {
      *config_proto = flib_runtime_->config_proto();
    }
    VLOG(4) << "Function " << function.name() << " in flib_runtime_";
  } else {
    if (config_proto) {
      *config_proto = local_flib_runtime_->config_proto();
    }
    VLOG(4) << "Function " << function.name() << " in local_flib_runtime_";
  }
  return OkStatus();
}

std::unique_ptr<Graph> XlaCompiler::GetGraph(const FunctionBody* fbody) {
  std::unique_ptr<Graph> graph(new Graph(options_.flib_def));
  CopyGraph(*fbody->graph, graph.get());

  bool is_inside_mustcompile = false;
  TryGetNodeAttr(AttrSlice(&fbody->fdef.attr()), kXlaMustCompileAttr,
                 &is_inside_mustcompile);

  // Performs a first function inlining pass before shape inference, since
  // otherwise shape inference can't see inside functions and a comprehensive
  // shape_map, including function ops, is needed to constant-propagate Shape
  // Ops below.
  auto flags = GetBuildXlaOpsPassFlags();
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_common_subexpression_elimination(false);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(!flags->tf_xla_disable_constant_folding);
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
  //
  // Also do not consider constant folding Shape ops. When there is a dynamic
  // dimension in a tensor, TF2XLA currently represent them as the static
  // upperbound shape, which can be constant folded and then lose the info
  // that this Shape is dynamic.
  auto cf_consider_fn = [](const Node* n) {
    for (const auto& output_arg : n->op_def().output_arg()) {
      if (output_arg.type() == DT_VARIANT) {
        return false;
      }
    }
    const auto& ts = n->type_string();
    // XLA has special logic to handle dynamic shapes, don't constant fold
    // them.
    if (ts == "Shape" || ts == "ShapeN" || ts == "Size") {
      return false;
    }
    return true;
  };
  GraphOptimizer::Options graph_optimizer_options;
  graph_optimizer_options.cf_consider_fn = cf_consider_fn;
  graph_optimizer_options.inline_multi_device_functions = true;
  graph_optimizer_options.inline_impl_selection_group_functions = true;
  graph_optimizer_options.inline_with_single_device_body_placer = true;
  graph_optimizer_options.ignore_noinline = is_inside_mustcompile;

  {
    GraphShapeInfo shape_info;
    InferShapes(graph.get(), /*arg_shapes=*/{},
                flib_runtime_->GetFunctionLibraryDefinition(), &shape_info)
        .IgnoreError();
    auto node_name_index = graph->BuildNodeNameIndex();
    std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
    for (const auto& node_shape_info : shape_info) {
      const string& node_name = node_shape_info.first;
      const std::vector<InferredShape>& output_shapes = node_shape_info.second;
      const auto& node_iter = node_name_index.find(node_name);
      if (node_iter != node_name_index.end()) {
        auto& partial_shapes = shape_map[node_name];
        for (const auto& inferred_shape : output_shapes) {
          partial_shapes.push_back(inferred_shape.shape);
        }
      }
    }
    graph_optimizer_options.shape_map = &shape_map;
    optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                       /*device=*/nullptr, &graph, graph_optimizer_options);
  }

  // Run shape inference on the graph and optimize the graph again.
  GraphShapeInfo shape_info;
  InferShapes(graph.get(), /*arg_shapes=*/{},
              flib_runtime_->GetFunctionLibraryDefinition(), &shape_info)
      .IgnoreError();
  auto node_name_index = graph->BuildNodeNameIndex();
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  for (const auto& node_shape_info : shape_info) {
    const string& node_name = node_shape_info.first;
    const std::vector<InferredShape>& output_shapes = node_shape_info.second;
    const auto& node_iter = node_name_index.find(node_name);
    if (node_iter != node_name_index.end()) {
      auto& partial_shapes = shape_map[node_name];
      for (const auto& inferred_shape : output_shapes) {
        partial_shapes.push_back(inferred_shape.shape);
      }
    }
  }
  graph_optimizer_options.shape_map = &shape_map;
  optimizer.Optimize(flib_runtime_, flib_runtime_->env(),
                     /*device=*/nullptr, &graph, graph_optimizer_options);

  return graph;
}

// Collects all control rets from `orig_control_ret_nodes` that are still valid,
// keeping the same order.
std::vector<std::string> GetValidControlRets(
    absl::Span<Node* const> orig_control_ret_nodes, const Graph& graph) {
  // Build map from control ret node name to index.
  // We use Node name instead of Node* here to index into the map as we populate
  // the map with nodes in FunctionDef control_ret_nodes and later query it
  // using the nodes in `graph`. The Node pointers would be different but the
  // Node name is expected to remain the same between the two.
  absl::flat_hash_map<const string, int> control_ret_nodes_map;
  for (int i = 0; i < orig_control_ret_nodes.size(); ++i) {
    const Node* n = orig_control_ret_nodes[i];
    control_ret_nodes_map[n->name()] = i;
  }
  // Check which control rets are still valid.
  std::vector<bool> is_valid_control_ret(orig_control_ret_nodes.size(), false);
  int num_valid_control_rets = 0;
  for (const Node* n : graph.nodes()) {
    auto iter = control_ret_nodes_map.find(n->name());
    if (iter != control_ret_nodes_map.end()) {
      ++num_valid_control_rets;
      is_valid_control_ret[iter->second] = true;
    }
  }
  // Return valid control rets in same order as they appear in
  // `orig_control_ret_nodes`.
  std::vector<std::string> valid_control_rets;
  valid_control_rets.reserve(num_valid_control_rets);
  for (int i = 0; i < orig_control_ret_nodes.size(); ++i) {
    if (is_valid_control_ret[i]) {
      valid_control_rets.push_back(orig_control_ret_nodes[i]->name());
    }
  }
  return valid_control_rets;
}

Status XlaCompiler::CompileFunction(
    const XlaCompiler::CompileOptions& options,
    const NameAttrList& fn_name_attrs,
    absl::Span<const XlaCompiler::Argument> args,
    XlaCompiler::CompilationResult* result) {
  const string function_id =
      Canonicalize(fn_name_attrs.name(), AttrSlice(&fn_name_attrs.attr()));
  VLOG(1) << "XlaCompiler::CompileFunction " << function_id;

  const std::vector<XlaCompiler::Argument> arg_vector(args.begin(), args.end());
  auto it = cache_.find({function_id, arg_vector});
  if (it != cache_.end()) {
    *result = it->second;
    return OkStatus();
  }

  const FunctionBody* fbody;
  const ConfigProto* config = nullptr;
  TF_RETURN_IF_ERROR(FindFunctionBody(fn_name_attrs, &fbody, &config));

  absl::optional<ConfigProto> config_proto;
  if (config) {
    config_proto = *config;
  }

  TF_RETURN_WITH_CONTEXT_IF_ERROR(
      CheckSignature(fbody->arg_types, args),
      "Signature check failure while compiling: ", fn_name_attrs.name());

  // Set shapes for _Arg nodes. They are useful for constant folding (e.g. an
  // Xla op requires a compile-time constant input, and that input is shape of
  // an _Arg node.
  for (int i = 0, end = args.size(); i < end; i++) {
    // Skip resource variables and tensor lists.
    DataType dtype;
    TF_RETURN_IF_ERROR(GetNodeAttr(fbody->arg_nodes[i]->def(), "T", &dtype));
    if (dtype == DT_RESOURCE || dtype == DT_VARIANT) {
      continue;
    }

    if (absl::holds_alternative<xla::Shape>(args[i].shape)) {
      xla::Shape xla_shape = absl::get<xla::Shape>(args[i].shape);
      TensorShape tensor_shape;
      // If xla_shape is dynamic, prevent constant folding by not setting
      // output_shapes.
      if (XLAShapeToTensorShape(xla_shape, &tensor_shape).ok() &&
          xla_shape.is_static()) {
        fbody->arg_nodes[i]->ClearAttr("_output_shapes");
        fbody->arg_nodes[i]->AddAttr("_output_shapes",
                                     std::vector<TensorShape>{tensor_shape});
      }
    } else {
      TensorShape tensor_shape = absl::get<TensorShape>(args[i].shape);
      fbody->arg_nodes[i]->ClearAttr("_output_shapes");
      fbody->arg_nodes[i]->AddAttr("_output_shapes",
                                   std::vector<TensorShape>{tensor_shape});
    }
  }

  std::unique_ptr<Graph> graph = GetGraph(fbody);

  // _Arg and _Retval nodes don't exist in the stored subgraph for the function;
  // they are added by the function body looked up.  Therefore, they don't have
  // core assignments here.
  // Attempt to assign a core to each _Retval and _Arg. Chooses the
  // lowest-numbered core that consumes the argument. We choose the
  // lowest-numbered core so the assignment is deterministic.
  for (Node* n : graph->nodes()) {
    if (n->IsArg()) {
      TF_RETURN_IF_ERROR(SetNodeShardingFromNeighbors(n, /*out_edges=*/true));
    }
  }
  // Do _Retval as a second loop, in case the retval's input is an _Arg (which
  // may have gotten a device assignment from the first loop).
  for (Node* n : graph->nodes()) {
    if (n->IsRetval()) {
      TF_RETURN_IF_ERROR(SetNodeShardingFromNeighbors(n, /*out_edges=*/false));
    }
  }

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileFunction: "
            << DumpGraphToFile(
                   absl::StrCat("xla_compile_function_", function_id), *graph);
  }

  VLOG(1) << "====================================================";
  MlirBridgeRolloutPolicy policy = MlirBridgeRolloutPolicy::kDisabledByUser;
  if (options.is_entry_computation) {
    policy = GetMlirBridgeRolloutPolicy(
        *graph, /*function_library=*/nullptr, config_proto,
        /*uses_uninitialized_resource_args=*/AnyUninitializedResourceArg(args));
  }
  if (policy == MlirBridgeRolloutPolicy::kEnabledByUser) {
    VLOG(1) << "Using MLIR bridge to compile the function";
    GraphDebugInfo debug_info;

    std::vector<std::string> valid_control_rets =
        GetValidControlRets(fbody->control_ret_nodes, *graph);
    TF_RETURN_IF_ERROR(CompileGraphToXlaHlo(
        std::move(*graph), mlir::SpanToArrayRef<XlaCompiler::Argument>(args),
        valid_control_rets, options_.device_type.type_string(),
        options.use_tuple_arg, /*analyse_graph=*/false, *options_.flib_def,
        debug_info, options_.shape_determination_fns, result));
  } else {
    VLOG(1) << "Using the old bridge to compile the function";
    TF_RETURN_IF_ERROR(
        CompileGraph(options, function_id, std::move(graph), args, result));
  }
  VLOG(1) << "====================================================";

  cache_[{function_id, arg_vector}] = *result;
  return OkStatus();
}

// Computes the XLA shape for argument 'arg'.
Status XlaCompiler::XLAShapeForArgument(
    const XlaCompiler::Argument& arg, bool is_entry_computation,
    const absl::optional<xla::HloSharding>& arg_sharding,
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
        auto layout_preference =
            options_.shape_determination_fns.layout_preference_fn(
                shape, arg.type, arg.kind);
        TF_ASSIGN_OR_RETURN(
            *xla_shape,
            options_.shape_determination_fns.shape_representation_fn(
                shape, arg.type,
                /*use_fast_memory=*/false, layout_preference));
        TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
            arg_sharding, /*use_fast_memory=*/false,
            options_.shape_determination_fns, xla_shape));
      } else {
        if (absl::holds_alternative<xla::Shape>(arg.shape)) {
          *xla_shape = absl::get<xla::Shape>(arg.shape);
        } else {
          TF_RETURN_IF_ERROR(TensorShapeToXLAShape(
              arg.type, absl::get<TensorShape>(arg.shape), xla_shape));
        }
      }
      return OkStatus();
    }
    case XlaCompiler::Argument::kTensorList: {
      TF_RET_CHECK(absl::holds_alternative<xla::Shape>(arg.shape));
      *xla_shape = absl::get<xla::Shape>(arg.shape);
      return OkStatus();
    }
    case XlaCompiler::Argument::kConstantResource:
    case XlaCompiler::Argument::kResource: {
      TF_RET_CHECK(arg.initialized);

      switch (arg.resource_kind) {
        case XlaResource::kVariable: {
          TF_RET_CHECK(absl::holds_alternative<TensorShape>(arg.shape));
          auto layout_preference =
              options_.shape_determination_fns.layout_preference_fn(
                  absl::get<TensorShape>(arg.shape), arg.type, arg.kind);
          TF_ASSIGN_OR_RETURN(
              *xla_shape,
              options_.shape_determination_fns.shape_representation_fn(
                  absl::get<TensorShape>(arg.shape), arg.type,
                  /*use_fast_memory=*/arg.fast_mem, layout_preference));
          TF_RETURN_IF_ERROR(RewriteLayoutWithShardedShape(
              arg_sharding, arg.fast_mem, options_.shape_determination_fns,
              xla_shape));
          return OkStatus();
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
          return OkStatus();
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
          return OkStatus();
        }

        case XlaResource::kInvalid:
          return errors::Internal(
              "Invalid resource type in XLAShapeForArgument()");
      }
    }
    case XlaCompiler::Argument::kToken: {
      *xla_shape = xla::ShapeUtil::MakeTokenShape();
      return OkStatus();
    }
    case XlaCompiler::Argument::kInvalid:
      return errors::Internal("Invalid argument type in XLAShapeForArgument()");
  }
}

/* static */
void XlaCompiler::PopulateArgumentFromResource(const XlaResource& resource,
                                               Argument* arg) {
  arg->initialized = resource.initialized();
  arg->kind = XlaCompiler::Argument::kResource;
  arg->resource_kind = resource.kind();

  arg->type = resource.type();
  arg->shape = resource.shape();
  arg->max_array_size = resource.max_array_size();
  for (const auto& gradient : resource.tensor_array_gradients()) {
    arg->tensor_array_gradients.insert(gradient.first);
  }
  arg->name = resource.name();
}

// Builds XLA computations for each of the arguments to the computation.
// `args` are the arguments to the computation.
Status XlaCompiler::BuildArguments(
    const Graph& graph, const std::vector<XlaCompiler::Argument>& args,
    bool use_tuple_arg, xla::XlaBuilder* builder, XlaContext* context,
    const std::map<int, xla::OpSharding>& arg_shardings,
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
      case XlaCompiler::Argument::kConstantResource:
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
                /*tensor_array_multiple_writes_aggregate=*/true,
                arg.definition_stack_trace));
        arg_expression =
            arg.kind == XlaCompiler::Argument::kResource
                ? XlaExpression::Resource(resource)
                : XlaExpression::ConstantResource(arg.constant_value, resource);
        if (arg.initialized) {
          input_to_args->push_back(i);
        }
        break;
      }
      case XlaCompiler::Argument::kParameter:
      case XlaCompiler::Argument::kTensorList:
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

  if (input_to_args->empty() && !use_tuple_arg) {
    return OkStatus();
  }

  // `arg_to_inputs[c] = d` means that the c'th original arg index corresponds
  // to the d'th XLA input. Note that the value -1 corresponds to constants, or
  // other args that don't correspond to an input.
  std::vector<int> arg_to_inputs(args.size(), -1);
  for (int i = 0, end = input_to_args->size(); i < end; i++) {
    arg_to_inputs[input_to_args->at(i)] = i;
  }

  std::vector<xla::Shape> arg_shapes(input_to_args->size());
  for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
    // Computes the shapes of non-constant arguments.
    auto arg_sharding = arg_shardings.find((*input_to_args)[i]);
    absl::optional<xla::HloSharding> sharding;
    if (arg_sharding != arg_shardings.end()) {
      TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                          xla::HloSharding::FromProto(arg_sharding->second));
      sharding = hlo_sharding;
    }
    TF_RETURN_IF_ERROR(XLAShapeForArgument(args[(*input_to_args)[i]],
                                           is_entry_computation, sharding,
                                           &arg_shapes[i]));
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
      tuple_sharding.set_type(xla::OpSharding::TUPLE);
      for (int64_t parameter : *input_to_args) {
        auto it = arg_shardings.find(parameter);
        *tuple_sharding.add_tuple_shardings() =
            it == arg_shardings.end() ? xla::sharding_builder::AssignDevice(0)
                                      : it->second;
      }
      std::vector<bool> is_same_across_replicas;
      for (int i = 0, end = input_to_args->size(); i < end; ++i) {
        // Add an entry to is_same_across_replicas for every leaf buffer.
        is_same_across_replicas.insert(
            is_same_across_replicas.end(),
            xla::ShapeUtil::GetLeafCount(arg_shapes[i]),
            args[input_to_args->at(i)].is_same_data_across_replicas);
      }
      xla::XlaScopedShardingAssignment assign_tuple_sharding(
          builder, input_to_args->empty() ? absl::optional<xla::OpSharding>()
                                          : tuple_sharding);
      tuple = xla::Parameter(builder, 0, (*input_shapes)[0], "arg_tuple",
                             is_same_across_replicas);
    } else {
      tuple = xla::Parameter(builder, 0, (*input_shapes)[0], "arg_tuple");
    }

    for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
      auto it = arg_shardings.find(i);
      xla::XlaScopedShardingAssignment assign_sharding(
          builder, it == arg_shardings.end() ? absl::optional<xla::OpSharding>()
                                             : it->second);
      auto& arg = args[input_to_args->at(i)];

      xla::OpMetadata arg_metadata;
      arg_metadata.set_op_name(arg.node_name);
      builder->SetOneShotOpMetadata(arg_metadata);
      arg_handles[i] = xla::GetTupleElement(tuple, i);
    }
  } else {
    for (std::vector<int>::size_type i = 0; i < input_to_args->size(); ++i) {
      auto it = arg_shardings.find(i);
      xla::XlaScopedShardingAssignment assign_sharding(
          builder, it == arg_shardings.end() ? absl::optional<xla::OpSharding>()
                                             : it->second);
      if (is_entry_computation) {
        // Add an entry to is_same_across_replicas for every leaf buffer.
        std::vector<bool> is_same_across_replicas(
            xla::ShapeUtil::GetLeafCount((*input_shapes)[i]),
            args[input_to_args->at(i)].is_same_data_across_replicas);
        arg_handles[i] =
            xla::Parameter(builder, i, (*input_shapes)[i],
                           absl::StrCat("arg", i), is_same_across_replicas);
      } else {
        arg_handles[i] = xla::Parameter(builder, i, (*input_shapes)[i],
                                        absl::StrCat("arg", i));
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
            << " name: " << arg.name << " TF arg " << input_to_args->at(i)
            << " node name: " << arg.node_name
            << (arg_shardings.find(i) == arg_shardings.end()
                    ? ""
                    : absl::StrCat(" sharding: ",
                                   arg_shardings.at(i).DebugString()));
    XlaExpression& arg_expression = (*arg_expressions)[input_to_args->at(i)];
    switch (arg.kind) {
      case XlaCompiler::Argument::kConstantResource:
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
          if (arg.value_bound) {
            TF_RET_CHECK(arg.value_dynamism);
            // Propagate upper bound and value dynamism to arg_expression.
            arg_expression.set_value_bound(arg.value_bound.value());
            arg_expression.set_value_dynamism(arg.value_dynamism.value());
          }
        }
        break;
      case XlaCompiler::Argument::kTensorList: {
        arg_expression = XlaExpression::TensorList(arg_handles[i]);
        break;
      }
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

  return OkStatus();
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
  return OkStatus();
}

// If node is PartitionedCall or StatefulPartitionedCall, returns the
// name from the "f" attr, else returns node.def().op().
// Returned pointer points to the internal string either in node's attributes
// or in its NodeDef. This pointer is valid as long as the node has not been
// modified.
Status GetPotentialFunctionName(const Node& node, const string** name) {
  if (node.IsPartitionedCall()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(
        node.attrs().Find(FunctionLibraryDefinition::kFuncAttr, &attr_value));
    if (!attr_value->has_func()) {
      return errors::InvalidArgument(
          "The attribute value for attribute 'f' in node ", node.DebugString(),
          " does not have 'func' field set");
    }
    *name = &attr_value->func().name();
    return OkStatus();
  }
  *name = &node.type_string();
  return OkStatus();
}

// Check that the graph doesn't have any invalid nodes (e.g. incompatible with
// given device_type, invalid data type, missing attributes...)
Status ValidateGraph(const Graph* graph,
                     const FunctionLibraryDefinition& flib_def,
                     const DeviceType& device_type, const string& name) {
  // Make sure the XLA compilation kernels are registered.  This operation is
  // idempotent so it is fine if someone called it already.
  XlaOpRegistry::RegisterCompilationKernels();

  auto maybe_error = [&](const Node* node, const Status& s) -> Status {
    if (!s.ok()) {
      std::string errmsg = absl::StrCat(
          "Detected unsupported operations when trying to compile graph ", name,
          " on ", device_type.type_string(), ": ", node->def().op(), " (",
          s.error_message(), ")", FormatNodeForError(*node));
      if (absl::StrContains(device_type.type_string(), "TPU")) {
        absl::StrAppend(&errmsg,
                        "\nOne approach is to outside compile the unsupported "
                        "ops to run on CPUs by enabling soft placement "
                        "`tf.config.set_soft_device_placement(True)`."
                        " This has a potential performance penalty.\n");
      }
      if (std::shared_ptr<AbstractStackTrace> stack_trace =
              node->GetStackTrace()) {
        absl::StrAppend(
            &errmsg, "\nThe op is created at: \n",
            stack_trace->ToString({/*show_line_contents =*/true,
                                   /*filter_common_prefix =*/true,
                                   /*drop_internal_frames =*/true}));
      }

      return errors::InvalidArgument(errmsg);
    }
    return OkStatus();
  };

  for (const Node* node : graph->nodes()) {
    if (node->type_string() == FunctionLibraryDefinition::kGradientOp) {
      continue;
    }
    const string* function_name;
    TF_RETURN_IF_ERROR(GetPotentialFunctionName(*node, &function_name));
    const FunctionDef* fdef = flib_def.Find(*function_name);
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
  return OkStatus();
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

Status XlaCompiler::CompileGraph(
    const XlaCompiler::CompileOptions& options, string const& name,
    std::unique_ptr<Graph> graph, absl::Span<const XlaCompiler::Argument> args,
    CompilationResult* result) {
  VLOG(1) << "Executing graph symbolically to populate XlaBuilder.: " << name;

  TF_RETURN_IF_ERROR(PropagateConstIntoFunctionalNodes(
      graph.get(), options_.flib_def, local_flib_def_.get()));
  TF_RETURN_IF_ERROR(RearrangeFunctionArguments(
      [this](const NameAttrList& function, const FunctionBody** fbody) {
        return FindFunctionBody(function, fbody);
      },
      graph.get(), local_flib_def_.get(),
      pflr_->GetFunctionLibraryDefinition()));

  if (VLOG_IS_ON(2)) {
    VLOG(2) << "XlaCompiler::CompileGraph: "
            << DumpGraphToFile(absl::StrCat("xla_compile_graph_", name), *graph,
                               flib_runtime_->GetFunctionLibraryDefinition());
  }

  // Report the error here if initialization failed.
  TF_RETURN_IF_ERROR(initialization_status_);

  // Detect invalid nodes.
  // FunctionalizeControlFlow may remove some nodes from the graph.
  TF_RETURN_IF_ERROR(ValidateGraph(graph.get(), *options_.flib_def,
                                   options_.device_type, name));
  xla::XlaBuilder builder(name);
  XlaContext* context = new XlaContext(this, &builder, graph.get());
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

  std::map<int, xla::OpSharding> arg_shardings;
  std::map<int, xla::OpSharding> retval_shardings;
  TF_ASSIGN_OR_RETURN(std::tie(arg_shardings, retval_shardings),
                      ComputeArgAndRetvalShardings(*graph));

  std::vector<XlaExpression> arg_expressions;
  TF_RETURN_IF_ERROR(BuildArguments(
      *graph, real_args, options.use_tuple_arg, &builder, context,
      arg_shardings, &arg_expressions, &result->input_mapping,
      &result->xla_input_shapes, options.is_entry_computation));
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

  Status execute_status = ExecuteGraph(context, std::move(graph), device_,
                                       flib_runtime_, NextStepId());
  if (!execute_status.ok()) {
    VLOG(1) << "Failed executing graph " << name;
    return execute_status;
  }
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
  ConvertConstantsToExpressions(&builder, absl::Span<XlaExpression>(retvals));
  XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns{
      UseNoPreferenceLayoutFn(), IdentityShapeRepresentationFn()};
  TF_RETURN_IF_ERROR(BuildComputation(
      real_args, retvals, arg_shardings, retval_shardings, context->resources(),
      std::move(token_output),
      options.is_entry_computation ? options_.shape_determination_fns
                                   : shape_determination_fns,
      options.is_entry_computation,
      options.return_updated_values_for_all_resources,
      options.always_return_tuple, options.use_tuple_arg,
      options.alias_resource_update, &builder, result->computation.get(),
      &num_computation_outputs, &num_nonconst_outputs, &result->outputs,
      &result->resource_updates, &result->xla_output_shape,
      result->input_mapping));

  VLOG(2) << "Outputs: total: " << context->retvals().size()
          << " nonconstant: " << num_nonconst_outputs;
  VLOG(2) << "XLA output shape: "
          << xla::ShapeUtil::HumanStringWithLayout(result->xla_output_shape);
  result->collective_info = context->GetCollectiveInfo();
  return OkStatus();
}

Status XlaCompiler::GetChannelHandle(const string& key,
                                     xla::ChannelHandle* channel) {
  auto result = channels_.emplace(key, xla::ChannelHandle());
  if (result.second) {
    TF_ASSIGN_OR_RETURN(result.first->second, client()->CreateChannelHandle());
  }
  *channel = result.first->second;
  VLOG(1) << "Channel: " << key << " " << channel->DebugString();
  return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

namespace {

void SetTransfer(const string& key, absl::Span<const DataType> types,
                 absl::Span<const TensorShape> shapes,
                 tf2xla::HostTransferMetadata* transfer) {
  transfer->set_key(key);
  CHECK(types.size() == shapes.size());
  for (int i = 0, end = types.size(); i < end; ++i) {
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
    tf2xla::HostTransferMetadata& existing_transfer = host_compute_sends_[key];
    tf2xla::HostTransferMetadata new_transfer;
    SetTransfer(key, types, shapes, &new_transfer);
    if (xla::protobuf_util::ProtobufEquals(existing_transfer, new_transfer)) {
      return OkStatus();
    } else {
      return errors::InvalidArgument(
          "Duplicate calls to SetDeviceToHostMetadata with key ", key);
    }
  }
  tf2xla::HostTransferMetadata& transfer = host_compute_sends_[key];
  SetTransfer(key, types, shapes, &transfer);
  return OkStatus();
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
  return OkStatus();
}

Status XlaCompiler::SetHostToDeviceMetadata(
    const string& key, absl::Span<const DataType> types,
    absl::Span<const TensorShape> shapes) {
  if (host_compute_recvs_.find(key) != host_compute_recvs_.end()) {
    tf2xla::HostTransferMetadata& existing_transfer = host_compute_recvs_[key];
    tf2xla::HostTransferMetadata new_transfer;
    SetTransfer(key, types, shapes, &new_transfer);
    if (xla::protobuf_util::ProtobufEquals(existing_transfer, new_transfer)) {
      return OkStatus();
    } else {
      return errors::InvalidArgument(
          "Duplicate calls to SetHostToDeviceMetadata with key ", key);
    }
  }
  tf2xla::HostTransferMetadata& transfer = host_compute_recvs_[key];
  SetTransfer(key, types, shapes, &transfer);
  return OkStatus();
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
  return OkStatus();
}

Status XlaCompiler::SetHostComputeControlDependency(
    const string& host_compute_name, const xla::XlaOp& handle) {
  if (host_compute_control_output_.find(host_compute_name) !=
      host_compute_control_output_.end()) {
    return errors::InvalidArgument(
        "Duplicate control handles registered for host compute Op ",
        host_compute_name);
  }
  host_compute_control_output_[host_compute_name] = handle;
  return OkStatus();
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
  return OkStatus();
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
  return OkStatus();
}

StatusOr<xla::XlaOp> XlaCompiler::GetNodeToken(const string& node_name) {
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
