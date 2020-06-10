/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tpu/kernels/tpu_compile_op_common.h"

#include <string>

#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace tpu {

namespace {

static constexpr char kArgOp[] = "_Arg";
static constexpr char kRetvalOp[] = "_Retval";

std::string CoreDevice(int core) {
  return strings::StrCat("/device:", DEVICE_TPU_REPLICATED_CORE, ":", core);
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
        << proto_arg.sharding().DebugString();
    for (int core = 0; core < per_core_arg_shapes->size(); ++core) {
      (*arg_core_mapping)[arg_index].indices.push_back(
          (*per_core_arg_shapes)[core].size());
      (*per_core_arg_shapes)[core].push_back(*xla_arg_shape);
    }
  }

  return Status::OK();
}

}  // namespace

Status TPUCompileOpKernelCommon::AssignReturnValueToCore(
    std::vector<tpu::ShardingAndIndex>* retval_core_mapping) {
  std::vector<int> per_core_retval_counts(metadata_.num_cores_per_replica(), 0);
  for (int i = 0; i < metadata_.retvals_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Retval& proto_retval =
        metadata_.retvals(i);
    (*retval_core_mapping)[i].sharding = proto_retval.sharding();
    if (proto_retval.sharding().type() == xla::OpSharding::MAXIMAL) {
      int core = proto_retval.sharding().tile_assignment_devices(0);
      TF_RET_CHECK(0 <= core && core < per_core_retval_counts.size());
      (*retval_core_mapping)[i].indices.push_back(
          per_core_retval_counts[core]++);
    } else if (proto_retval.sharding().type() == xla::OpSharding::OTHER) {
      for (int64 core : proto_retval.sharding().tile_assignment_devices()) {
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
  return Status::OK();
}

Status TPUCompileOpKernelCommon::BuildComputationArgumentDescriptions(
    const std::vector<TensorShape>& arg_shapes,
    const OpInputList& guaranteed_constants, const XlaCompiler& compiler,
    std::vector<XlaCompiler::Argument>* args,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  // Builds a description of the computation's arguments.
  int constant_count = 0;
  for (int i = 0; i < metadata_.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& proto_arg = metadata_.args(i);
    args->push_back(XlaCompiler::Argument());
    XlaCompiler::Argument& arg = args->back();
    arg.type = proto_arg.dtype();
    arg.shape = arg_shapes[i];
    switch (proto_arg.kind()) {
      case tpu::TPUCompileMetadataProto::Arg::PARAMETER:
        arg.kind = XlaCompiler::Argument::kParameter;
        break;
      case tpu::TPUCompileMetadataProto::Arg::VARIABLE:
        arg.kind = XlaCompiler::Argument::kResource;
        arg.resource_kind = XlaResource::kVariable;
        arg.initialized = true;
        arg.fast_mem = proto_arg.fast_mem();
        break;
      case tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT:
        arg.kind = XlaCompiler::Argument::kConstant;
        TF_RET_CHECK(constant_count < guaranteed_constants.size())
            << "More constant args in TPUCompileMetadataProto than constant "
               "tensors.";
        arg.constant_value = guaranteed_constants[constant_count++];
        break;
      case tpu::TPUCompileMetadataProto::Arg::INVALID:
      default:
        break;
    }
    arg.is_same_data_across_replicas = proto_arg.is_same_data_across_replicas();
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
  TF_RET_CHECK(constant_count == guaranteed_constants.size())
      << "Not all of the constant tensors were consumed.";

  return Status::OK();
}

Status TPUCompileOpKernelCommon::GetShardingInfo(
    absl::Span<const TensorShape> arg_shapes,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  int num_inputs = metadata_.args_size();
  for (int i = 0; i < num_inputs; ++i) {
    const auto& proto_arg = metadata_.args(i);
    TF_ASSIGN_OR_RETURN(auto arg_sharding,
                        xla::HloSharding::FromProto(proto_arg.sharding()));
    TF_ASSIGN_OR_RETURN(
        auto xla_arg_shape,
        shape_representation_fn(arg_shapes[i], proto_arg.dtype(),
                                /*use_fast_memory=*/false));
    TF_RETURN_IF_ERROR(
        RewriteLayoutWithShardedShape(arg_sharding, /*use_fast_memory=*/false,
                                      shape_representation_fn, &xla_arg_shape));
    TF_RETURN_IF_ERROR(SetPerCoreArgShapes(
        proto_arg, i, &xla_arg_shape, arg_core_mapping, per_core_arg_shapes));
  }
  return Status::OK();
}

Status TPUCompileOpKernelCommon::CompileTFFunctionToHlo(
    const FunctionLibraryDefinition& flib_def, int graph_def_version,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    const std::vector<TensorShape>& arg_shapes,
    const OpInputList& guaranteed_constants, const NameAttrList& function,
    std::function<Status(ResourceMgr*)> populate_resource_manager_fn,
    xla::CompileOnlyClient* client,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes,
    XlaCompiler::CompilationResult* compilation_result) {
  XlaCompiler::Options compiler_options;
  compiler_options.device_type = DeviceType(DEVICE_TPU_XLA_JIT);
  compiler_options.client = client;
  compiler_options.flib_def = &flib_def;
  compiler_options.allow_cpu_custom_calls = false;
  compiler_options.populate_resource_manager = &populate_resource_manager_fn;
  compiler_options.graph_def_version = graph_def_version;
  compiler_options.shape_representation_fn = shape_representation_fn;

  auto compiler = absl::make_unique<XlaCompiler>(compiler_options);

  std::vector<XlaCompiler::Argument> args;
  TF_RETURN_IF_ERROR(BuildComputationArgumentDescriptions(
      arg_shapes, guaranteed_constants, *compiler, &args, arg_core_mapping,
      per_core_arg_shapes));

  // Assign each return value to a core.
  std::vector<tpu::ShardingAndIndex> retval_core_mapping(
      metadata_.retvals_size());
  TF_RETURN_IF_ERROR(
      TPUCompileOpKernelCommon::AssignReturnValueToCore(&retval_core_mapping));

  LOG(INFO) << "Instantiating function:" << function.name();
  FunctionLibraryRuntime::Handle handle;
  TF_RETURN_IF_ERROR(compiler->flib_runtime()->Instantiate(
      function.name(), AttrSlice(&function.attr()), &handle));
  const FunctionBody* fbody = compiler->flib_runtime()->GetFunctionBody(handle);
  const string function_id =
      Canonicalize(function.name(), AttrSlice(&function.attr()));

  std::unique_ptr<Graph> graph(new Graph(&flib_def));
  CopyGraph(*fbody->graph, graph.get());

  VLOG(2) << "metadata: " << metadata_.DebugString();
  std::vector<int> parameter_arg_mapping;
  for (int i = 0; i < args.size(); i++) {
    XlaCompiler::Argument& arg = args[i];
    if (arg.kind != XlaCompiler::Argument::kParameter) {
      continue;
    }
    parameter_arg_mapping.push_back(i);
  }
  TF_RET_CHECK(fbody->arg_nodes.size() == args.size());
  for (size_t i = 0; i < fbody->arg_nodes.size(); i++) {
    args[i].node_name = fbody->arg_nodes[i]->name();
  }

  std::vector<gtl::InlinedVector<int64, 4>> arg_shape_dims;
  arg_shape_dims.reserve(arg_shapes.size());
  std::vector<PartialTensorShape> partial_arg_shapes(arg_shapes.size());
  for (const TensorShape& shape : arg_shapes) {
    arg_shape_dims.push_back(shape.dim_sizes());
  }

  for (const auto& padding_mapping : metadata_.padding_maps()) {
    if (padding_mapping.padding_arg_index() >= parameter_arg_mapping.size()) {
      return errors::Internal(absl::StrCat(
          "TPUCompileMetadataProto `padding_maps` has `padding_arg_index` ",
          padding_mapping.padding_arg_index(),
          " which exceeds`parameter_arg_mapping` array bounds ",
          parameter_arg_mapping.size(),
          ". this usually indicates there are dynamic shape inputs fed into "
          "TPUs from outside compilation head extraction, which is not "
          "supported"));
    }
    int padding_arg_index =
        parameter_arg_mapping.at(padding_mapping.padding_arg_index());
    args[parameter_arg_mapping.at(padding_mapping.arg_index())]
        .dynamic_dim_to_arg_num_map[padding_mapping.shape_index()] =
        padding_arg_index;
    arg_shape_dims[parameter_arg_mapping.at(padding_mapping.arg_index())]
                  [padding_mapping.shape_index()] = -1;
    args[padding_arg_index].is_pad_arg = true;
  }

  for (int64 i = 0; i < arg_shape_dims.size(); ++i) {
    auto& dims = arg_shape_dims[i];
    TF_RETURN_IF_ERROR(PartialTensorShape::MakePartialShape(
        dims.data(), dims.size(), &partial_arg_shapes[i]));
  }

  // Adds device assignments to _Arg and _Retval nodes.
  TF_RETURN_IF_ERROR(AssignDevicesToArgsAndRetvals(
      absl::MakeSpan(*arg_core_mapping), absl::MakeSpan(retval_core_mapping),
      graph.get()));

  VLOG(1) << "Optimizing TensorFlow graph";
  FunctionLibraryDefinition flib_definition(flib_def);
  TF_RETURN_IF_ERROR(OptimizeGraph(metadata_, partial_arg_shapes, &graph,
                                   compiler->flib_runtime(), &flib_definition));

  VLOG(1) << "Compiling TensorFlow graph to HLO";
  XlaCompiler::CompileOptions compile_options;
  compile_options.return_updated_values_for_all_resources = false;
  compile_options.use_tuple_arg = true;
  compile_options.is_entry_computation = true;
  compile_options.alias_resource_update = true;
  return compiler->CompileGraph(compile_options, function_id, std::move(graph),
                                args, compilation_result);
}

/* static */ void TPUCompileOpKernelCommon::ExitCountdown(
    OpKernelContext* ctx, std::shared_ptr<std::atomic<bool>> done) {
  const int kSleepSeconds = 300;
  LOG(INFO) << "TpuCompileOp was cancelled. Sleeping for " << kSleepSeconds
            << " seconds to give time for TPUCompileOp to finished.";
  ctx->env()->SleepForMicroseconds(kSleepSeconds * 1000000);
  if (done->load()) {
    // If the TPUCompileOp has finished, then terminate peacefully.
    return;
  }

#ifdef PLATFORM_GOOGLE
  // TODO(henrytan): copybara this out.
  LOG(ERROR) << "Aborting process due to cancelled TPUCompileOp. This "
                "termination is to ensure a consistent state. If your job "
                "does not restart, modify the retries allowed. See "
                "b/71383512.";
#else
  LOG(ERROR) << "Restarting TPU host due to cancelled compilation.";
#endif

  // Exit code 42 tells the Borglet to restart the task and not report it as a
  // failure: http://g3doc/borg/slave/g3doc/users/task_features
  std::quick_exit(42);
}

/* static */ Status TPUCompileOpKernelCommon::GetDynamicShapes(
    OpKernelContext* ctx, std::vector<TensorShape>* shapes) {
  OpInputList dynamic_shapes;
  TF_RETURN_IF_ERROR(ctx->input_list("dynamic_shapes", &dynamic_shapes));

  shapes->resize(dynamic_shapes.size());
  for (int i = 0; i < dynamic_shapes.size(); ++i) {
    TF_RETURN_IF_ERROR(
        tpu::ShapeTensorToTensorShape(dynamic_shapes[i], &(*shapes)[i]));
  }
  return Status::OK();
}

/* static */ Status TPUCompileOpKernelCommon::ComputeArgumentShapes(
    const tpu::TPUCompileMetadataProto& metadata,
    const std::vector<TensorShape>& dynamic_shapes,
    std::vector<TensorShape>* arg_shapes) {
  arg_shapes->resize(metadata.args_size());
  int dynamic_shape_pos = 0;
  for (int i = 0; i < metadata.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& arg = metadata.args(i);
    // The XLA compiler determines the shape of each constant by inspecting the
    // value of its corresponding host-memory tensor. As a result, we don't need
    // to give the compiler graph-inferred shapes for constant arguments.
    if (arg.kind() == tpu::TPUCompileMetadataProto::Arg::GUARANTEED_CONSTANT) {
      continue;
    }
    TF_RETURN_IF_ERROR(PartialTensorShape::IsValidShape(arg.shape()));
    PartialTensorShape static_shape(arg.shape());

    TensorShape& shape = (*arg_shapes)[i];
    if (static_shape.IsFullyDefined()) {
      TF_RET_CHECK(static_shape.AsTensorShape(&shape));
    } else {
      TF_RET_CHECK(dynamic_shape_pos < dynamic_shapes.size())
          << "Too few dynamic shapes";
      shape = dynamic_shapes[dynamic_shape_pos++];
      if (!static_shape.IsCompatibleWith(shape)) {
        return errors::InvalidArgument(
            "Mismatch between static and dynamic shape for argument. Static "
            "shape: ",
            static_shape.DebugString(),
            "; dynamic shape: ", shape.DebugString());
      }
    }
  }
  // Checks we consumed all of the dynamic shapes.
  TF_RET_CHECK(dynamic_shape_pos == dynamic_shapes.size())
      << "Too many dynamic shapes";
  return Status::OK();
}

// Function arguments and return values lose their device assignments, so we
// must recreate them.
/* static */ Status TPUCompileOpKernelCommon::AssignDevicesToArgsAndRetvals(
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
    return Status::OK();
  };
  for (Node* node : graph->op_nodes()) {
    if (node->type_string() == kArgOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < arg_core_mapping.size());
      TF_RETURN_IF_ERROR(assign(node, arg_core_mapping[index].sharding));
    } else if (node->type_string() == kRetvalOp) {
      int index;
      TF_RETURN_IF_ERROR(GetNodeAttr(node->attrs(), "index", &index));
      TF_RET_CHECK(index >= 0 && index < retval_core_mapping.size());
      TF_RETURN_IF_ERROR(assign(node, retval_core_mapping[index].sharding));
    }
  }
  return Status::OK();
}

// Performs shape inference on the body of `graph`. Shapes for arguments
// are taken from `metadata` and `arg_shapes`.
/* static */ Status TPUCompileOpKernelCommon::RunShapeInferenceOnComputation(
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

Status TPUCompileOpKernelCommon::OptimizeGraph(
    const tpu::TPUCompileMetadataProto& metadata,
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
  // Performs a first function inlining pass before shape inference, since
  // otherwise shape inference can't see inside functions and a comprehensive
  // shape_map, including function ops, is needed to constant-propagate Shape
  // Ops below.
  GraphOptimizer::Options optimizer_opts;
  optimizer_opts.inline_multi_device_functions = true;
  optimizer_opts.inline_impl_selection_group_functions = true;
  optimizer_opts.inline_with_single_device_body_placer = true;
  optimizer.Optimize(flr, flr->env(), flr->device(), graph, optimizer_opts);

  // Infer shapes for each node in the computation.
  GraphShapeInfo shape_info;
  TF_RETURN_IF_ERROR(RunShapeInferenceOnComputation(
      metadata, arg_shapes, graph->get(), flr, &shape_info));

  // Converts the GraphShapeInfo into the form needed by the constant-folding
  // pass of the optimizer.
  std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
  ConvertGraphShapeInfoToShapeMap(**graph, shape_info, &shape_map);
  optimizer.Optimize(flr, flr->env(), flr->device(), graph, &shape_map);

  TF_RETURN_IF_ERROR(RewriteTensorListWithConstElement(graph->get(), fld));

  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
