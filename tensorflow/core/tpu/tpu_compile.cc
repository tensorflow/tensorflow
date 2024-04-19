/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tpu/tpu_compile.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/compiler/tf2xla/layout_util.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "xla/client/compile_only_client.h"
#include "xla/literal_util.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/common_runtime/function_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_support.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace tpu {
namespace {

// For stateless RNGs ops, they are pure but device-dependent. Those ops are not
// constant-foldable.
// TODO(b/305092010) Use the operations' TF_NoConstantFold attribute instead.
static absl::flat_hash_set<std::string>* kBlockList =
    new absl::flat_hash_set<std::string>({
        "StatelessRandomUniform",
        "StatelessRandomNormal",
        "StatelessTruncatedNormal",
    });

std::string CoreDevice(int core) {
  return strings::StrCat("/device:", DEVICE_TPU_REPLICATED_CORE, ":", core);
}

static constexpr char kArgOp[] = "_Arg";
static constexpr char kRetvalOp[] = "_Retval";

// Sets arg shape, arg core mapping, and per core arg shapes for a given
// argument, depending on its sharding.
Status SetPerCoreArgShapes(
    const tpu::TPUCompileMetadataProto::Arg& proto_arg, const int arg_index,
    xla::Shape* xla_arg_shape,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  if (proto_arg.unrestricted_layout()) {
    xla_arg_shape->clear_layout();
  }

  (*arg_core_mapping)[arg_index].sharding = proto_arg.sharding();
  if (proto_arg.sharding().type() == xla::OpSharding::MAXIMAL) {
    const int core = proto_arg.sharding().tile_assignment_devices(0);
    TF_RET_CHECK(0 <= core && core < per_core_arg_shapes->size());
    (*arg_core_mapping)[arg_index].indices.push_back(
        (*per_core_arg_shapes)[core].size());
    (*per_core_arg_shapes)[core].push_back(*xla_arg_shape);
  } else if (proto_arg.sharding().type() == xla::OpSharding::OTHER) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(proto_arg.sharding()));
    for (int core : proto_arg.sharding().tile_assignment_devices()) {
      (*arg_core_mapping)[arg_index].indices.push_back(
          (*per_core_arg_shapes)[core].size());
      xla::Shape per_core_shape =
          GetPerDeviceShape(*xla_arg_shape, hlo_sharding, core);
      if (proto_arg.unrestricted_layout()) {
        per_core_shape.clear_layout();
      }
      (*per_core_arg_shapes)[core].push_back(per_core_shape);
    }
  } else {
    TF_RET_CHECK(proto_arg.sharding().type() == xla::OpSharding::REPLICATED)
        << "Unsupported argument sharding: "
        << " proto_arg=" << proto_arg.DebugString();
    for (int core = 0; core < per_core_arg_shapes->size(); ++core) {
      (*arg_core_mapping)[arg_index].indices.push_back(
          (*per_core_arg_shapes)[core].size());
      (*per_core_arg_shapes)[core].push_back(*xla_arg_shape);
    }
  }

  return absl::OkStatus();
}

// Adds TPU_REPLICATED_CORE device assignments to the _Arg and _Retval
// nodes in `graph', using the sharding/index assignments in
// `arg_core_mapping` and `retval_core_mapping`. The mappings are maps from
// original argument/return index to (sharding, per-core argument/return
// index) pairs. Node attributes, such as device assignments, are not
// preserved on function argument and return values nodes, so we must recreate
// them the compilation metadata.
// Function arguments and return values lose their device assignments, so we
// must recreate them.
Status AssignDevicesToArgsAndRetvals(
    absl::Span<const tpu::ShardingAndIndex> arg_core_mapping,
    absl::Span<const tpu::ShardingAndIndex> retval_core_mapping, Graph* graph) {
  auto assign = [&](Node* node, const xla::OpSharding& sharding) -> Status {
    if (sharding.type() == xla::OpSharding::MAXIMAL) {
      const string device = CoreDevice(sharding.tile_assignment_devices(0));
      node->set_assigned_device_name(device);
      node->set_requested_device(device);
    } else {
      TF_RET_CHECK(sharding.type() == xla::OpSharding::REPLICATED ||
                   sharding.type() == xla::OpSharding::OTHER)
          << "Unsupported sharding on parameter/retval: "
          << sharding.DebugString();
    }
    node->AddAttr("_XlaSharding", sharding.SerializeAsString());
    return absl::OkStatus();
  };
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(
          tensorflow::GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < arg_core_mapping.size());
      TF_RETURN_IF_ERROR(assign(node, arg_core_mapping[index].sharding));
    } else if (node->type_string() == kRetvalOp) {
      int index;
      TF_RETURN_IF_ERROR(
          tensorflow::GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < retval_core_mapping.size());
      TF_RETURN_IF_ERROR(assign(node, retval_core_mapping[index].sharding));
    }
  }
  return absl::OkStatus();
}

void ConvertGraphShapeInfoToShapeMap(
    const Graph& graph, const GraphShapeInfo& graph_shape_info,
    std::unordered_map<string, std::vector<PartialTensorShape>>* shape_map) {
  // Builds a map from node name to Node* for `graph`.
  std::unordered_map<string, Node*> index;
  for (Node* node : graph.nodes()) {
    index[node->name()] = node;
  }
  // Discards the resource handle shape info while converting to the correct map
  // form.
  for (const auto& node_shape_info : graph_shape_info) {
    const string& node_name = node_shape_info.first;
    const std::vector<InferredShape>& output_shapes = node_shape_info.second;
    // Gets the vector of partial shapes, first converting node name to Node*
    // using index. graph is the subgraph of the original graph assigned to a
    // particular core, and we only add entries to shape_map for nodes in
    // graph_shape_info that are in the subgraph.
    const auto& node_iter = index.find(node_name);
    if (node_iter != index.end()) {
      auto& partial_shapes = (*shape_map)[node_name];
      for (const auto& inferred_shape : output_shapes) {
        partial_shapes.push_back(inferred_shape.shape);
      }
    }
  }
}

bool DoNotConsiderOpsInBlockList(const Node* n) {
  if (kBlockList->contains(n->type_string())) {
    VLOG(2) << "Skip node [" << n->DebugString()
            << "] for constant folding, it is in constant folding block list";
    return false;
  }
  return true;
}

// Optimizes `graph`, given the argument descriptions in `metadata` and
// `arg_shapes`.
Status OptimizeGraph(const tpu::TPUCompileMetadataProto& metadata,
                     const std::vector<PartialTensorShape>& arg_shapes,
                     std::unique_ptr<Graph>* graph, FunctionLibraryRuntime* flr,
                     FunctionLibraryDefinition* fld) {
  // Sets up options for the optimization passes that need to be done. Notice
  // that CSE is not needed as XLA has its own CSE passes later in the
  // compilation stage.
  auto flags = GetBuildXlaOpsPassFlags();
  OptimizerOptions opts;
  opts.set_opt_level(OptimizerOptions::L0);
  opts.set_do_common_subexpression_elimination(false);
  opts.set_do_function_inlining(true);
  opts.set_do_constant_folding(!flags->tf_xla_disable_constant_folding);
  GraphOptimizer optimizer(opts);
  {
    // Performs a first function inlining pass before shape inference, since
    // otherwise shape inference can't see inside functions and a comprehensive
    // shape_map, including function ops, is needed to constant-propagate Shape
    // Ops below.
    GraphOptimizer::Options optimizer_opts;
    optimizer_opts.inline_multi_device_functions = true;
    optimizer_opts.inline_impl_selection_group_functions = true;
    optimizer_opts.inline_with_single_device_body_placer = true;
    optimizer_opts.cf_consider_fn = DoNotConsiderOpsInBlockList;
    // Infer shapes for each node in the computation. Shape inference can help
    // skip constant folding of large shapes.
    GraphShapeInfo shape_info;
    TF_RETURN_IF_ERROR(internal::RunShapeInferenceOnComputation(
        metadata, arg_shapes, graph->get(), flr, &shape_info));
    // Converts the GraphShapeInfo into the form needed by the constant-folding
    // pass of the optimizer.
    std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
    ConvertGraphShapeInfoToShapeMap(**graph, shape_info, &shape_map);
    optimizer_opts.shape_map = &shape_map;
    optimizer.Optimize(flr, flr->env(), flr->device(), graph, optimizer_opts);
  }

  {
    // Infer shapes for each node in the computation.
    GraphShapeInfo shape_info;
    TF_RETURN_IF_ERROR(internal::RunShapeInferenceOnComputation(
        metadata, arg_shapes, graph->get(), flr, &shape_info));
    std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
    ConvertGraphShapeInfoToShapeMap(**graph, shape_info, &shape_map);
    GraphOptimizer::Options optimizer_opts;
    optimizer_opts.shape_map = &shape_map;
    optimizer_opts.cf_consider_fn = DoNotConsiderOpsInBlockList;
    optimizer.Optimize(flr, flr->env(), flr->device(), graph, optimizer_opts);
  }

  TF_RETURN_IF_ERROR(RewriteTensorListWithConstElement(graph->get(), fld));

  return absl::OkStatus();
}

// Populates the mapping from return value to ShardingAndIndex.
Status AssignReturnValueToCore(
    const tpu::TPUCompileMetadataProto& metadata,
    std::vector<tpu::ShardingAndIndex>* retval_core_mapping) {
  std::vector<int> per_core_retval_counts(metadata.num_cores_per_replica(), 0);
  for (int i = 0; i < metadata.retvals_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Retval& proto_retval =
        metadata.retvals(i);
    (*retval_core_mapping)[i].sharding = proto_retval.sharding();
    if (proto_retval.sharding().type() == xla::OpSharding::MAXIMAL) {
      int core = proto_retval.sharding().tile_assignment_devices(0);
      TF_RET_CHECK(0 <= core && core < per_core_retval_counts.size());
      (*retval_core_mapping)[i].indices.push_back(
          per_core_retval_counts[core]++);
    } else if (proto_retval.sharding().type() == xla::OpSharding::OTHER) {
      for (int64_t core : proto_retval.sharding().tile_assignment_devices()) {
        (*retval_core_mapping)[i].indices.push_back(
            per_core_retval_counts[core]++);
      }
    } else {
      TF_RET_CHECK(proto_retval.sharding().type() ==
                   xla::OpSharding::REPLICATED)
          << "Unsupported return value sharding: "
          << proto_retval.sharding().DebugString();
      for (int core = 0; core < per_core_retval_counts.size(); ++core) {
        (*retval_core_mapping)[i].indices.push_back(
            per_core_retval_counts[core]++);
      }
    }
  }
  return absl::OkStatus();
}

// If the metadata specifies any bounded dynamic shapes in the arg then create
// the matching Tensor values for the Argument.
Status MaybeBuildBoundedDynamicArgValues(
    const tpu::TPUCompileMetadataProto::Arg& proto_arg,
    const TensorShape& shape, XlaCompiler::Argument& arg) {
  // If any entry in the is_bounded_dynamic_dim list is true then we update the
  // value_bound and value_dynamism fields to indicate that there is dynamism,
  // the bounds, and which dimensions are dynamic.
  auto is_dynamic_dim = absl::MakeConstSpan(proto_arg.is_bounded_dynamic_dim());
  if (std::any_of(is_dynamic_dim.begin(), is_dynamic_dim.end(),
                  [](bool v) { return v; })) {
    // Assume that the values in the shape are the maximums.
    arg.value_bound = Tensor(arg.type, shape);
    // Build a literal tensor of Bools to hold which Dims are dynamic.
    auto literal = xla::LiteralUtil::CreateR1(is_dynamic_dim);
    Tensor dynamism_tensor(DT_BOOL);
    TF_RETURN_IF_ERROR(LiteralToHostTensor(literal, DT_BOOL, &dynamism_tensor));
    arg.value_dynamism = dynamism_tensor;
  }
  return absl::OkStatus();
}

// Populates the arguments, core mapping and per core argument shape for the
// computation.
Status BuildComputationArgumentDescriptions(
    const std::vector<TensorShape>& arg_shapes,
    const GuaranteedConsts& guaranteed_constants, const XlaCompiler& compiler,
    const tpu::TPUCompileMetadataProto& metadata,
    std::vector<XlaCompiler::Argument>* args,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  arg_core_mapping->clear();
  arg_core_mapping->resize(metadata.args_size());

  per_core_arg_shapes->clear();
  per_core_arg_shapes->resize(metadata.num_cores_per_replica());

  // Builds a description of the computation's arguments.
  int constant_count = 0;
  size_t guaranteed_constants_size = 0;
  for (int i = 0; i < metadata.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& proto_arg = metadata.args(i);
    args->push_back(XlaCompiler::Argument());
    XlaCompiler::Argument& arg = args->back();
    arg.type = proto_arg.dtype();
    arg.shape = arg_shapes[i];
    arg.node_name = proto_arg.name();
    switch (proto_arg.kind()) {
      case tpu::TPUCompileMetadataProto::Arg::PARAMETER:
        arg.kind = XlaCompiler::Argument::kParameter;
        // TODO(b/308845592) Maybe do this with the XlaCompileOnDemand version
        // of this method and maybe move whole method to a shared location.
        TF_RETURN_IF_ERROR(
            MaybeBuildBoundedDynamicArgValues(proto_arg, arg_shapes[i], arg));
        break;
      case tpu::TPUCompileMetadataProto::Arg::VARIABLE:
        arg.kind = XlaCompiler::Argument::kResource;
        arg.resource_kind = XlaResource::kVariable;
        arg.initialized = true;
        arg.fast_mem = proto_arg.fast_mem();
        break;
      case tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT:
        arg.kind = XlaCompiler::Argument::kConstant;
        guaranteed_constants_size =
            guaranteed_constants.index() == 0
                ? std::get<0>(guaranteed_constants).size()
                : std::get<1>(guaranteed_constants)->size();
        TF_RET_CHECK(constant_count < guaranteed_constants_size)
            << "More constant args in TPUCompileMetadataProto than constant "
               "tensors.";
        if (guaranteed_constants.index() == 0) {
          // `guaranteed_constants` is of type `absl::Span<const TensorProto*
          // const>`.
          Tensor tensor;
          CHECK(tensor.FromProto(
              *std::get<0>(guaranteed_constants)[constant_count++]))
              << "Failed to deserialize invalid `TensorProto` into `Tensor`.";
          arg.constant_value = tensor;
        } else {
          // `guaranteed_constants` is of type `const OpInputList* const`.
          arg.constant_value =
              (*std::get<1>(guaranteed_constants))[constant_count++];
        }
        break;
      case tpu::TPUCompileMetadataProto::Arg::INVALID:
      default:
        break;
    }
    arg.is_same_data_across_replicas = proto_arg.is_same_data_across_replicas();
    arg.requires_broadcast = proto_arg.requires_xla_broadcast();
    if (arg.kind == XlaCompiler::Argument::kInvalid) {
      return errors::InvalidArgument("Invalid argument kind");
    }
    if (arg.kind == XlaCompiler::Argument::kConstant) {
      continue;
    }

    // Assign each argument a sharding.
    xla::Shape xla_arg_shape;
    TF_ASSIGN_OR_RETURN(auto arg_sharding,
                        xla::HloSharding::FromProto(proto_arg.sharding()));
    TF_RETURN_IF_ERROR(compiler.XLAShapeForArgument(
        arg, /*is_entry_computation=*/true, arg_sharding, &xla_arg_shape));
    TF_RETURN_IF_ERROR(SetPerCoreArgShapes(
        proto_arg, i, &xla_arg_shape, arg_core_mapping, per_core_arg_shapes));
  }
  TF_RET_CHECK(constant_count == guaranteed_constants_size)
      << "Not all of the constant tensors were consumed.";

  return absl::OkStatus();
}
}  // namespace

namespace internal {
Status RunShapeInferenceOnComputation(
    const tpu::TPUCompileMetadataProto& metadata,
    const std::vector<PartialTensorShape>& arg_shapes, Graph* graph,
    FunctionLibraryRuntime* flr, GraphShapeInfo* shape_info) {
  int num_args = arg_shapes.size();
  CHECK_EQ(num_args, metadata.args_size());

  std::map<int, InferredShape> arg_shapes_for_inference;
  for (int i = 0; i < num_args; ++i) {
    const auto& arg = metadata.args(i);
    InferredShape& shape_for_inference = arg_shapes_for_inference[i];
    if (arg.kind() == tpu::TPUCompileMetadataProto::Arg::VARIABLE) {
      // For resource variables, arg_shapes[] contains the shape of the
      // variable's value.
      shape_for_inference.handle_type = arg.dtype();
      shape_for_inference.handle_shape = arg_shapes[i];
      // The shape of the variable itself is always a scalar.
      shape_for_inference.shape = TensorShape();
    } else {
      if (arg.kind() ==
          tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT) {
        VLOG(1) << "PromisedConstant shape: " << arg_shapes[i].DebugString();
      }
      shape_for_inference.shape = arg_shapes[i];
    }
  }
  return InferShapes(
      graph, arg_shapes_for_inference,
      flr != nullptr ? flr->GetFunctionLibraryDefinition() : nullptr,
      shape_info);
}
}  // namespace internal

Status CompileTFFunctionToHlo(
    const FunctionLibraryDefinition& flib_def, int graph_def_version,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    const std::vector<TensorShape>& arg_shapes,
    const GuaranteedConsts& guaranteed_constants, const NameAttrList& function,
    const tpu::TPUCompileMetadataProto& metadata,
    xla::CompileOnlyClient* client,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    bool use_tuple_args, XlaCompiler::CompilationResult* compilation_result) {
  XlaCompiler::Options compiler_options;
  FunctionLibraryDefinition flib_definition(flib_def);
  compiler_options.device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  compiler_options.client = client;
  compiler_options.flib_def = &flib_definition;
  compiler_options.allow_cpu_custom_calls = false;
  compiler_options.graph_def_version = graph_def_version;
  compiler_options.shape_determination_fns = shape_determination_fns;

  auto compiler = std::make_unique<XlaCompiler>(compiler_options);

  std::vector<XlaCompiler::Argument> args;
  TF_RETURN_IF_ERROR(BuildComputationArgumentDescriptions(
      arg_shapes, guaranteed_constants, *compiler, metadata, &args,
      arg_core_mapping, per_core_arg_shapes));

  // Assign each return value to a core.
  std::vector<tpu::ShardingAndIndex> retval_core_mapping(
      metadata.retvals_size());
  TF_RETURN_IF_ERROR(AssignReturnValueToCore(metadata, &retval_core_mapping));

  LOG(INFO) << "Instantiating function:" << function.name();
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(compiler->flib_runtime()->Instantiate(
      function.name(), AttrSlice(&function.attr()), &handle));
  const FunctionBody* fbody = compiler->flib_runtime()->GetFunctionBody(handle);
  const string function_id =
      Canonicalize(function.name(), AttrSlice(&function.attr()));

  std::unique_ptr<Graph> graph(new Graph(&flib_definition));
  CopyGraph(*fbody->graph, graph.get());

  VLOG(2) << "metadata: " << metadata.DebugString();
  TF_RET_CHECK(fbody->arg_nodes.size() == args.size());
  for (size_t i = 0; i < fbody->arg_nodes.size(); i++) {
    args[i].node_name = fbody->arg_nodes[i]->name();
  }

  std::vector<gtl::InlinedVector<int64_t, 4>> arg_shape_dims;
  arg_shape_dims.reserve(arg_shapes.size());
  std::vector<PartialTensorShape> partial_arg_shapes(arg_shapes.size());
  for (const TensorShape& shape : arg_shapes) {
    arg_shape_dims.push_back(shape.dim_sizes());
  }

  for (int64_t i = 0; i < arg_shape_dims.size(); ++i) {
    auto& dims = arg_shape_dims[i];
    TF_RETURN_IF_ERROR(PartialTensorShape::MakePartialShape(
        dims.data(), dims.size(), &partial_arg_shapes[i]));
  }

  // Adds device assignments to _Arg and _Retval nodes.
  TF_RETURN_IF_ERROR(AssignDevicesToArgsAndRetvals(
      absl::MakeSpan(*arg_core_mapping), absl::MakeSpan(retval_core_mapping),
      graph.get()));

  VLOG(1) << "Optimizing TensorFlow graph";
  TF_RETURN_IF_ERROR(OptimizeGraph(metadata, partial_arg_shapes, &graph,
                                   compiler->flib_runtime(), &flib_definition));

  VLOG(1) << "Compiling TensorFlow graph to HLO";
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = false;
  compile_options.use_tuple_arg = use_tuple_args;
  compile_options.is_entry_computation = true;
  compile_options.alias_resource_update = true;
  return compiler->CompileGraph(compile_options, function_id, std::move(graph),
                                args, compilation_result);
}

Status GetShardingInfo(
    const tpu::TPUCompileMetadataProto& metadata,
    absl::Span<const TensorShape> arg_shapes,
    const XlaShapeLayoutHelpers::ShapeDeterminationFns shape_determination_fns,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  arg_core_mapping->clear();
  arg_core_mapping->resize(metadata.args_size());

  per_core_arg_shapes->clear();
  per_core_arg_shapes->resize(metadata.num_cores_per_replica());

  int num_inputs = metadata.args_size();
  for (int i = 0; i < num_inputs; ++i) {
    const auto& proto_arg = metadata.args(i);
    TF_ASSIGN_OR_RETURN(auto arg_sharding,
                        xla::HloSharding::FromProto(proto_arg.sharding()));
    auto layout_preference = shape_determination_fns.layout_preference_fn(
        arg_shapes[i], proto_arg.dtype(), std::nullopt);
    TF_ASSIGN_OR_RETURN(auto xla_arg_shape,
                        shape_determination_fns.shape_representation_fn(
                            arg_shapes[i], proto_arg.dtype(),
                            /*use_fast_memory=*/false, layout_preference));
    TF_RETURN_IF_ERROR(
        RewriteLayoutWithShardedShape(arg_sharding, /*use_fast_memory=*/false,
                                      shape_determination_fns, &xla_arg_shape));
    TF_RETURN_IF_ERROR(SetPerCoreArgShapes(
        proto_arg, i, &xla_arg_shape, arg_core_mapping, per_core_arg_shapes));
  }
  return absl::OkStatus();
}

}  // namespace tpu
}  // namespace tensorflow
