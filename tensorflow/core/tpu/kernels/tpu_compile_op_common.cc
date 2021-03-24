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

#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/tpu/compilation_result.pb.h"
#include "tensorflow/core/protobuf/tpu/compile_metadata.pb.h"
#include "tensorflow/core/protobuf/tpu/dynamic_padding.pb.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_entry_unloader.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_cache_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_compilation_metrics.h"
#include "tensorflow/core/tpu/kernels/tpu_compile_op_options.h"
#include "tensorflow/core/tpu/kernels/tpu_op_consts.h"
#include "tensorflow/core/tpu/kernels/tpu_op_util.h"
#include "tensorflow/core/tpu/kernels/tpu_program_group_interface.h"
#include "tensorflow/core/tpu/kernels/tpu_util.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_compile_interface.h"
#include "tensorflow/core/tpu/tpu_configuration.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/core/tpu/tpu_ops_c_api.h"

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
        << " proto_arg=" << proto_arg.DebugString();
    for (int core = 0; core < per_core_arg_shapes->size(); ++core) {
      (*arg_core_mapping)[arg_index].indices.push_back(
          (*per_core_arg_shapes)[core].size());
      (*per_core_arg_shapes)[core].push_back(*xla_arg_shape);
    }
  }

  return Status::OK();
}

}  // namespace

CompileOpImplFactory* CompileOpImplFactory::factory_ = nullptr;

/* static */
CompileOpImplFactory* CompileOpImplFactory::Get() { return factory_; }

/* static */
void CompileOpImplFactory::Register(CompileOpImplFactory* factory) {
  CHECK_EQ(factory_, nullptr)
      << "CompileOpImplFactory can only be registered "
         "once and there can only be one factory active and used.";
  factory_ = factory;
}

Status TpuCompileOpKernelCommon::AssignReturnValueToCore(
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

Status TpuCompileOpKernelCommon::BuildComputationArgumentDescriptions(
    const std::vector<TensorShape>& arg_shapes,
    const GuaranteedConsts& guaranteed_constants, const XlaCompiler& compiler,
    std::vector<XlaCompiler::Argument>* args,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  // Builds a description of the computation's arguments.
  int constant_count = 0;
  size_t guaranteed_constants_size = 0;
  for (int i = 0; i < metadata_.args_size(); ++i) {
    const tpu::TPUCompileMetadataProto::Arg& proto_arg = metadata_.args(i);
    args->push_back(XlaCompiler::Argument());
    XlaCompiler::Argument& arg = args->back();
    arg.type = proto_arg.dtype();
    arg.shape = arg_shapes[i];
    arg.node_name = proto_arg.name();
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
        guaranteed_constants_size =
            guaranteed_constants.index() == 0
                ? absl::get<0>(guaranteed_constants).size()
                : absl::get<1>(guaranteed_constants)->size();
        TF_RET_CHECK(constant_count < guaranteed_constants_size)
            << "More constant args in TPUCompileMetadataProto than constant "
               "tensors.";
        if (guaranteed_constants.index() == 0) {
          // `guaranteed_constants` is of type `absl::Span<const TensorProto*
          // const>`.
          Tensor tensor;
          CHECK(tensor.FromProto(
              *absl::get<0>(guaranteed_constants)[constant_count++]))
              << "Failed to deserialize invalid `TensorProto` into `Tensor`.";
          arg.constant_value = tensor;
        } else {
          // `guaranteed_constants` is of type `const OpInputList* const`.
          arg.constant_value =
              (*absl::get<1>(guaranteed_constants))[constant_count++];
        }
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
  TF_RET_CHECK(constant_count == guaranteed_constants_size)
      << "Not all of the constant tensors were consumed.";

  return Status::OK();
}

Status TpuCompileOpKernelCommon::GetShardingInfo(
    absl::Span<const TensorShape> arg_shapes,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    std::vector<tpu::ShardingAndIndex>* arg_core_mapping,
    std::vector<std::vector<xla::Shape>>* per_core_arg_shapes) {
  arg_core_mapping->clear();
  arg_core_mapping->resize(metadata_.args_size());

  per_core_arg_shapes->clear();
  per_core_arg_shapes->resize(metadata_.num_cores_per_replica());

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

Status TpuCompileOpKernelCommon::CompileTFFunctionToHlo(
    const FunctionLibraryDefinition& flib_def, int graph_def_version,
    const XlaCompiler::ShapeRepresentationFn shape_representation_fn,
    const std::vector<TensorShape>& arg_shapes,
    const GuaranteedConsts& guaranteed_constants, const NameAttrList& function,
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
      TpuCompileOpKernelCommon::AssignReturnValueToCore(&retval_core_mapping));

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

/* static */ void TpuCompileOpKernelCommon::ExitCountdown(
    Env* env, std::shared_ptr<std::atomic<bool>> done) {
  const int kSleepSeconds = 300;
  LOG(INFO) << "TpuCompileOp was cancelled. Sleeping for " << kSleepSeconds
            << " seconds to give time for TPUCompileOp to finished.";
  env->SleepForMicroseconds(kSleepSeconds * 1000000);
  if (done->load()) {
    // If the TpuCompileOp has finished, then terminate peacefully.
    return;
  }

  LOG(ERROR) << "Aborting process due to cancelled TpuCompileOp. This "
             << "termination is to ensure a consistent state.";
  std::exit(42);
}

/* static */ Status TpuCompileOpKernelCommon::GetDynamicShapes(
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

// Function arguments and return values lose their device assignments, so we
// must recreate them.
/* static */ Status TpuCompileOpKernelCommon::AssignDevicesToArgsAndRetvals(
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
/* static */ Status TpuCompileOpKernelCommon::RunShapeInferenceOnComputation(
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

Status TpuCompileOpKernelCommon::OptimizeGraph(
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
  {
    // Performs a first function inlining pass before shape inference, since
    // otherwise shape inference can't see inside functions and a comprehensive
    // shape_map, including function ops, is needed to constant-propagate Shape
    // Ops below.
    GraphOptimizer::Options optimizer_opts;
    optimizer_opts.inline_multi_device_functions = true;
    optimizer_opts.inline_impl_selection_group_functions = true;
    optimizer_opts.inline_with_single_device_body_placer = true;
    // Infer shapes for each node in the computation. Shape inference can help
    // skip constant folding of large shapes.
    GraphShapeInfo shape_info;
    TF_RETURN_IF_ERROR(RunShapeInferenceOnComputation(
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
    TF_RETURN_IF_ERROR(RunShapeInferenceOnComputation(
        metadata, arg_shapes, graph->get(), flr, &shape_info));
    std::unordered_map<string, std::vector<PartialTensorShape>> shape_map;
    ConvertGraphShapeInfoToShapeMap(**graph, shape_info, &shape_map);
    optimizer.Optimize(flr, flr->env(), flr->device(), graph, &shape_map);
  }

  TF_RETURN_IF_ERROR(RewriteTensorListWithConstElement(graph->get(), fld));

  return Status::OK();
}

void TpuCompileOpKernelCommon::Compute(OpKernelContext* ctx) {
  VLOG(1) << "Cloud TPU: TpuCompileOpKernelCommon::Compute";

  std::shared_ptr<std::atomic<bool>> done(new std::atomic<bool>(false));

  CancellationToken token =
      ctx->cancellation_manager()->get_cancellation_token();
  const bool already_cancelled =
      !ctx->cancellation_manager()->RegisterCallback(token, [ctx, done]() {
        if (OpsApiFn()->TpuCompile_ShouldTpuCompileOpIgnoreCancellationFn()) {
          return;
        }

        // Sleep and exit in another thread so the cancellation manager can
        // continue running callbacks.
        Env* env = ctx->env();
        env->SchedClosure([env, done]() { ExitCountdown(env, done); });
      });

  // If the RPC was cancelled before we registered the cancellation callback,
  // don't compile the TPU program.
  OP_REQUIRES(ctx, !already_cancelled,
              errors::Cancelled("RPC cancelled, not compiling TPU program"));

  // We only want to abort the process if a cancellation actually occurs during
  // compilation; we must deregister the callback in the success case. It
  // doesn't hurt to also deregister the callback in the failure case; the
  // CancellationManager ensures that already-registered callbacks will be run
  // once cancellation has started.
  auto cancellation_cleanup = xla::MakeCleanup([ctx, token, done] {
    ctx->cancellation_manager()->DeregisterCallback(token);
    done->store(true);
  });

  Status compile_status = ComputeInternal(ctx);
  string status_payload;
  // Construct payload if compile_status is not ok and there's no payload for
  // compilation yet.
  if (!compile_status.ok() &&
      compile_status.GetPayload(TpuCompileInterface::kTpuCompileErrorPayloadKey)
          .empty()) {
    tpu::CompilationResultProto proto;
    proto.set_status_code(compile_status.code());
    proto.set_status_error_message(compile_status.error_message());
    status_payload = proto.SerializeAsString();
  }
  OP_REQUIRES_OK_OR_SET_PAYLOAD(ctx,
                                TpuCompileInterface::kTpuCompileErrorPayloadKey,
                                status_payload, compile_status);
}

Status TpuCompileOpKernelCommon::CompileLocallyAndFillHostCache(
    FunctionLibraryRuntime* flib_runtime,
    const SessionMetadata* session_metadata,
    const TpuMeshStateInterface* mesh_state,
    const std::vector<TensorShape>& dynamic_shapes,
    const OpInputList& guaranteed_constants, const TpuCompilationCacheKey& key,
    TpuProgramGroupInterface* tpu_program_group) {
  absl::Time start_time = absl::Now();
  std::vector<TensorShape> arg_shapes;
  TF_RETURN_IF_ERROR(
      ComputeArgumentShapes(metadata_, dynamic_shapes, &arg_shapes));
  Status compile_status;
  if (use_mlir_) {
    const ConfigProto* config = flib_runtime->config_proto();
    ConfigProto::Experimental::MlirBridgeRollout rollout_state =
        GetMlirBridgeRolloutState(config ? absl::make_optional(*config)
                                         : absl::nullopt);
    compile_status = Compile(MlirToHloArgs{mlir_module_, rollout_state},
                             mesh_state->data(), arg_shapes, tpu_program_group);
  } else {
    compile_status =
        Compile(FunctionToHloArgs{&function_,
                                  flib_runtime->GetFunctionLibraryDefinition(),
                                  flib_runtime->graph_def_version(),
                                  {&guaranteed_constants}},
                mesh_state->data(), arg_shapes, tpu_program_group);
  }

  absl::Time end_time = absl::Now();
  auto duration = end_time - start_time;

  const std::string session_name = SessionNameFromMetadata(session_metadata);
  LOG(INFO) << "Compilation of " << key.prefix << " with session name "
            << session_name << " took " << duration << " and "
            << (compile_status.ok() ? "succeeded" : "failed");
  tpu_program_group->LogProgramMemorySummary();
  metrics::UpdateXlaCompilationTime(absl::ToInt64Microseconds(duration));
  TpuCompilationMetrics::IncrementCompilationCount(session_name);

  TF_RETURN_IF_ERROR(tpu_program_group->LogCompilationStats(key, duration));

  return compile_status;
}

Status TpuCompileOpKernelCommon::ComputeInternal(OpKernelContext* ctx) {
  VLOG(1) << "Retrieving mesh state";
  // Retrieve the topology from the resource manager
  ResourceMgr* rm = GetTPUConfigResourceMgr();

  TpuMeshStateInterface* mesh_state;
  TF_RETURN_IF_ERROR(rm->Lookup(rm->default_container(),
                                kTpuMeshStateInterfaceResourceName,
                                &mesh_state));
  core::ScopedUnref mesh_state_unref(mesh_state);

  std::vector<TensorShape> dynamic_shapes;
  TF_RETURN_IF_ERROR(GetDynamicShapes(ctx, &dynamic_shapes));

  OpInputList guaranteed_constants;
  // TODO(ycao): Decide whether/how to support guaranteed constants in
  // MLIR-based TF-Compiler Bridge.
  if (!use_mlir_) {
    TF_RETURN_IF_ERROR(
        ctx->input_list("guaranteed_constants", &guaranteed_constants));
  }

  const TpuCompilationCacheKey key = CreateCompilationCacheKey(
      function_.name(), metadata_.function_library_fingerprint(),
      mlir_module_fingerprint_, guaranteed_constants, dynamic_shapes, metadata_,
      *mesh_state);

  // Process-wide cache of TPU executables.
  TpuCompilationCacheInterface* cache;
  TF_RETURN_IF_ERROR(rm->Lookup<TpuCompilationCacheInterface>(
      rm->default_container(), kCompilationCacheResourceName, &cache));
  core::ScopedUnref cache_unref(cache);

  // Per-step object that ensures that compilation cache entries aren't
  // evicted until the step completes. This mechanism ensures that the
  // downstream TPUExecute Ops in this step will be able to look up the
  // compiled executable even if it is marked for eviction before the step
  // ends.
  //
  // We can't use GetTPUConfigResourceMgr here because it may return the
  // global ResourceMgr, which is not associated with any device, and
  // GraphMgr's ScopedStepContainer only searches ResourceMgrs associated
  // with devices when deleting resources at step boundaries.
  CompilationRefHolder* ref_holder;
  if (ctx->step_container() == nullptr) {
    return errors::FailedPrecondition(
        "TPUCompileOp requires a step container.");
  }
  TF_RETURN_IF_ERROR(
      ctx->step_container()->LookupOrCreate<CompilationRefHolder>(
          ctx->resource_manager(), "ref_holder", &ref_holder,
          [cache](CompilationRefHolder** h) {
            *h = cache->MakePerStepRefHolder();
            return Status::OK();
          }));
  core::ScopedUnref ref_holder_unref(ref_holder);

  int64 uid;
  std::vector<std::string> proto_key;
  std::vector<std::string> sharding_key;
  std::vector<bool> may_modify_variables;
  absl::Span<const xla::HloProto* const> hlo_metadatas;
  Status status = cache->CompileIfKeyAbsent(
      key, ctx->session_metadata(), ref_holder, &uid, &proto_key, &sharding_key,
      &may_modify_variables, &hlo_metadatas,
      [&](TpuProgramGroupInterface* tpu_program_group) {
        VLOG(1) << "Cloud TPU: Compiling TPU program";
        // When this compile function is invoked, we know that host-memory
        // cache TpuCompilationCache saw a cache miss. There are two codepaths:
        // 1. If persistent cache is disabled, compile locally and populate
        //    host-memory cache.
        // 2. If persistent cache is enabled, we do an additional lookup on
        //    the persistent cache.
        //    - If persistent cache also sees a cache miss, trigger
        //      compilation. Then, populate both persistent cache and
        //      host-memory cache.
        //    - If persistent cache sees a cache hit, retrieve cache entry from
        //      persistent cache to populate host-memory cache without
        //      recompilation. If retrieval failed, compile locally as a
        //      fallback and use the local compilation result to populate
        //      host-memory cache.
        if (persistent_cache_ == nullptr) {
          VLOG(1) << "Persistent compilation cache not enabled. Compiling "
                     "TPU executable locally and populating host-memory cache.";
          return CompileLocallyAndFillHostCache(
              ctx->function_library(), ctx->session_metadata(), mesh_state,
              dynamic_shapes, guaranteed_constants, key, tpu_program_group);
        }
        return LookupPersistentCompilationCacheAndFillCaches(
            ctx->function_library(), ctx->session_metadata(), mesh_state,
            dynamic_shapes, guaranteed_constants, persistent_cache_.get(), key,
            tpu_program_group);
      });

  // `ref_holder` is provided to CompileIfKeyAbsent to ensure that cache
  // entry does not get evicted before TpuExecuteOp runs it and discards
  // `ref_holder`. When TpuCompilationCacheEntryUnloader get destroyed in the
  // event that user closes the session while there are in-flight program
  // executions, it will discard the cache's reference to the cache entry
  // and but not removed the entry until `ref_holder` discards the last
  // reference to the entry. This ensures that the guarantees of
  // `ref_holder` is not violated when this flag is true.
  if (unload_cache_entry_on_session_close_) {
    // Place `unloader` in TPU_SYSTEM device resource manager. Note that
    // - TPUConfigResourceMgr returned by GetTPUConfigResourceMgr() is a special
    //   process-global ResourceMgr. There is only one TPUConfigResourceMgr, and
    //   it is never destroyed.
    // - TPU_SYSTEM device resource manager is a normal device ResourceMgr for
    //   TPU_SYSTEM device. If DirectSession or isolate_session_state are used,
    //   there's one TPU_SYSTEM ResourceMgr for each session, and the
    //   ResourceMgrs will be destroyed when their corresponding session is
    //   closed. Otherwise there's one TPU_SYSTEM ResourceMgr that's only
    //   destroyed when the master-session is destroyed, not when the worker
    //   sessions are destroyed
    TpuCompilationCacheEntryUnloader* unloader;
    TF_RETURN_IF_ERROR(
        ctx->resource_manager()
            ->LookupOrCreate<TpuCompilationCacheEntryUnloader>(
                ctx->resource_manager()->default_container(),
                kCompilationCacheUnloaderResourceName, &unloader,
                [cache](TpuCompilationCacheEntryUnloader** new_unloader) {
                  *new_unloader = new TpuCompilationCacheEntryUnloader(cache);
                  return Status::OK();
                }));
    // Note that LookupOrCreate puts two refcounts on unloader.
    core::ScopedUnref unloader_unref(unloader);
    unloader->AddCacheEntryUid(uid);
  }

  int64 num_cores_with_compiled_programs = proto_key.size();
  if (proto_key.size() == 1) {
    // SPMD produces 1 program for all cores.
    num_cores_with_compiled_programs = metadata_.num_cores_per_replica();
  }
  if (status.ok() &&
      num_cores_with_compiled_programs +
              (may_modify_variables.size() * static_cast<int>(!use_mlir_)) !=
          ctx->num_outputs() - 1) {
    status = errors::Internal(
        "Number of cores with compiled programs (",
        num_cores_with_compiled_programs, ") + variable states (",
        may_modify_variables.size() * static_cast<int>(!use_mlir_),
        ") + compilation status output != number of compile op outputs (",
        ctx->num_outputs(), ")");
  }

  // TODO(jpienaar): status is not just due to the compilation. At this
  // point we should be failing the execution of the op in some cases and
  // returning a compilation error in others. For now, uniformly return an
  // error and fail in _TPUExecute if status failed here.

  // TODO(misard) the frame id will be wrong if this is ever called from
  // within a function. Consider whether to use the same hack as is
  // present in the rendezvous manager where the function call frame is
  // cast to a uint64, or do something better all around.
  std::string rendezvous_key_base = strings::StrCat(
      "host_compute_rendezvous:", ctx->op_kernel().name(), ":",
      ctx->frame_iter().frame_id, ":", ctx->frame_iter().iter_id, ":");

  // Return compilation status.
  {
    Tensor output(DT_STRING, TensorShape({}));
    tpu::CompilationResultProto proto;
    proto.set_status_code(status.code());
    if (!status.ok()) {
      proto.set_status_error_message(
          absl::StrCat("Compilation failure: ", status.error_message()));
    }
    if (return_hlo_protos_) {
      // Return the HloProtos as part of compilation status.
      for (const xla::HloProto* hlo_metadata : hlo_metadatas) {
        xla::HloProto* hlo_proto = proto.add_hlo_protos();
        *hlo_proto = *hlo_metadata;
      }
    }
    SerializeToTString(proto, &output.scalar<tstring>()());
    ctx->set_output(0, output);
    status.SetPayload(TpuCompileInterface::kTpuCompileErrorPayloadKey,
                      output.scalar<tstring>()());
  }

  if (status.ok()) {
    for (int i = 0; i < num_cores_with_compiled_programs; ++i) {
      Tensor output(DT_STRING, TensorShape({3}));
      if (proto_key.size() == 1) {
        output.vec<tstring>()(0) = proto_key[0];
      } else {
        output.vec<tstring>()(0) = proto_key[i];
      }
      output.vec<tstring>()(1) = rendezvous_key_base;
      if (sharding_key.empty()) {
        output.vec<tstring>()(2) = "";
      } else if (sharding_key.size() == 1) {
        output.vec<tstring>()(2) = sharding_key[0];
      } else {
        TF_RET_CHECK(sharding_key.size() == num_cores_with_compiled_programs);
        output.vec<tstring>()(2) = sharding_key[i];
      }
      ctx->set_output(i + 1, output);
    }
    if (!use_mlir_) {
      // If any of the programs may modify a variable, then return that all
      // do as the only current state being tracked here is if a model is
      // read-only or not.
      bool may_modify = false;
      for (bool m : may_modify_variables) {
        may_modify = may_modify || m;
      }
      for (int i = 0; i < may_modify_variables.size(); ++i) {
        Tensor output(DT_BOOL, TensorShape({}));
        output.scalar<bool>()() = may_modify;
        ctx->set_output(i + num_cores_with_compiled_programs + 1, output);
      }
    }
    VLOG(1) << "Cloud TPU: Compilation succeeded";
  } else {
    // Return error in the invalid case.
    for (int i = 0; i < num_computations_; ++i) {
      Tensor output(DT_STRING, TensorShape({3}));
      output.vec<tstring>()(0) = "<<NO PROGRAM AS COMPILATION FAILED>>";
      output.vec<tstring>()(1) = "<<NO RENDEZVOUS KEY AS COMPILATION FAILED>>";
      output.vec<tstring>()(2) = "<<NO SHARDing KEY AS COMPILATION FAILED>>";
      ctx->set_output(i + 1, output);
    }
    if (!use_mlir_) {
      // The TPUCompileMLIR op does not have MayModifyVariable output
      for (int i = 0; i < num_computations_; ++i) {
        Tensor output(false);
        ctx->set_output(i + num_computations_ + 1, output);
      }
    }
  }
  return status;
}
}  // namespace tpu
}  // namespace tensorflow
