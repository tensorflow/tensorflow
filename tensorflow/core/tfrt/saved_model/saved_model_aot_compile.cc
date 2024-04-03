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

#include "tensorflow/core/tfrt/saved_model/saved_model_aot_compile.h"

#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/compiler/jit/device_compilation_cluster_signature.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/tf_graph_to_hlo_compiler.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/tfrt_pipeline_options.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/export_mlir.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"
#include "tensorflow/core/tfrt/saved_model/utils/serialize_utils.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system_helper.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tensorflow::tfrt_stub {
namespace {
void UpdateCompileOptions(AotOptions& options) {
  // Disable DecomposeResourceOpsPass for now, as DecomposeResourceGather does
  // not work well with GPU (b/232819415).
  if (options.graph_execution_options->enable_tfrt_gpu) {
    options.graph_execution_options->compile_options.decompose_resource_ops =
        false;
  }

  options.graph_execution_options->compile_options
      .fuse_get_resource_ops_in_hoisting =
      !options.graph_execution_options->enable_mlrt;
}

Status CompileTfGraphToHlo(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    XlaCompiler::Options* options,
    XlaCompiler::CompilationResult** compilation_result) {
  // Construct a GPU device.
  DeviceAttributes device_proto;
  device_proto.set_name("/job:localhost/replica:0/task:0/device:GPU:0");
  device_proto.set_device_type(DEVICE_GPU);
  auto device =
      std::make_unique<VirtualDevice>(tensorflow::Env::Default(), device_proto);

  XlaPlatformInfo platform_info(DEVICE_GPU, se::cuda::kCudaPlatformId, nullptr,
                                nullptr, nullptr);
  *options = GenerateCompilerOptionsForPjRt(
      flib_def, graph_def_version, device.get(), platform_info, nullptr);
  // Set device type correctly so that compilation can find kernels.
  options->device_type = DeviceType("XLA_GPU_JIT");

  XlaCompiler::CompileOptions compile_options =
      GenerateCompileOptions(has_ref_vars, may_alias_resource_update);
  TfGraphToHloCompiler compiler(*options);
  auto compilation_status =
      compiler.Compile(compile_options, function, args, *compilation_result);
  if ((*compilation_result)->computation == nullptr) {
    LOG(ERROR) << compilation_status;
    return compilation_status;
  }
  return absl::OkStatus();
}

// Signature node name is "${node_name}:0". This function extracts node_name.
std::string GetNodeName(const std::string& signature_node_name) {
  int node_name_len = signature_node_name.size();
  return signature_node_name.substr(0, node_name_len - 2);
}

Status UpdateGraphDefWithInputShapes(
    MetaGraphDef& meta_graph_def,
    const absl::flat_hash_map<std::string, tensorflow::TensorShapeProto>&
        input_shapes,
    const std::string& signature_name) {
  if (!meta_graph_def.signature_def().contains(signature_name)) {
    return absl::NotFoundError(
        absl::StrCat("Signature not found: ", signature_name));
  }
  SignatureDef& signature_def =
      (*meta_graph_def.mutable_signature_def())[signature_name];

  // Maps from graph node name to its tensor shape.
  absl::flat_hash_map<std::string, tensorflow::TensorShapeProto>
      graph_input_shapes;
  for (const auto& input : input_shapes) {
    *((*signature_def.mutable_inputs())[input.first].mutable_tensor_shape()) =
        input.second;
    const std::string node_name = signature_def.inputs().at(input.first).name();
    graph_input_shapes[GetNodeName(node_name)] = input.second;
  }
  // Update GraphDef node shapes.
  for (NodeDef& node : *meta_graph_def.mutable_graph_def()->mutable_node()) {
    if (graph_input_shapes.find(node.name()) != graph_input_shapes.end()) {
      if (node.attr().contains("_output_shapes")) {
        (*(*node.mutable_attr())["_output_shapes"]
              .mutable_list()
              ->mutable_shape())[0] = graph_input_shapes[node.name()];
      }
      if (node.attr().contains("shape")) {
        *((*node.mutable_attr())["shape"].mutable_shape()) =
            graph_input_shapes[node.name()];
      }
    }
  }
  return OkStatus();
}

// Constructs function and args in place using `xla_func_def`.
void ConstructFunctionAndArgs(const std::string& name,
                              const FunctionDef& xla_func_def,
                              NameAttrList& function,
                              std::vector<XlaCompiler::Argument>& args) {
  function.set_name(name);
  *function.mutable_attr() = xla_func_def.attr();
  args.resize(xla_func_def.signature().input_arg_size());
  for (const auto& attr : xla_func_def.arg_attr()) {
    XlaCompiler::Argument arg;
    const int index = attr.first;
    arg.name = index;
    TensorShapeProto shape_proto =
        attr.second.attr().at("_output_shapes").list().shape(0);
    arg.shape = shape_proto;
    arg.kind = XlaCompiler::Argument::kParameter;
    arg.type = xla_func_def.signature().input_arg(index).type();
    arg.initialized = true;
    args[index] = arg;
  }
}
}  // namespace

AotOptions::AotOptions() : graph_execution_options(nullptr) {}

StatusOr<AotResult> AotCompileSavedModel(absl::string_view input_model_dir,
                                         AotOptions aot_options) {
  TF_ASSIGN_OR_RETURN(tensorflow::MetaGraphDef meta_graph_def,
                      ReadSavedModel(input_model_dir, aot_options.tags));

  UpdateTpuTargetByBridgeCompatibility(*aot_options.graph_execution_options,
                                       meta_graph_def.graph_def());
  UpdateCompileOptions(aot_options);
  mlir::DialectRegistry registry;
  RegisterMlirDialect(
      registry,
      aot_options.graph_execution_options->compile_options.backend_compiler);
  mlir::MLIRContext context(registry);

  tensorflow::SessionOptions session_options =
      CreateDefaultSessionOptions(*aot_options.graph_execution_options);
  session_options.config.mutable_experimental()->set_optimize_for_static_graph(
      true);
  LOG_FIRST_N(INFO, 10) << "SessionOptions: "
                        << session_options.config.DebugString();
  LOG_FIRST_N(INFO, 10) << "GraphExecutionOptions: "
                        << *aot_options.graph_execution_options;

  const ::tensorflow::FunctionDefLibrary& fdef_lib =
      meta_graph_def.graph_def().library();
  ASSIGN_OR_RETURN_IN_IMPORT(
      std::unique_ptr<tensorflow::tfrt_stub::FallbackState> fallback_state,
      FallbackState::CreateWithMockGpuDevice(session_options, fdef_lib));

  ASSIGN_OR_RETURN_IN_IMPORT(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      ImportSavedModel(&context, meta_graph_def, *fallback_state,
                       std::string(input_model_dir),
                       /*import_user_signatures=*/true,
                       aot_options.graph_execution_options
                           ->run_placer_grappler_on_functions));

  auto kernel_registry = std::make_unique<mlrt::KernelRegistry>();
  auto resource_context = std::make_unique<tfrt::ResourceContext>();
  ModelRuntimeContext model_context(&*aot_options.graph_execution_options,
                                    std::string(input_model_dir),
                                    resource_context.get());

  {
    model_context.set_meta_graph_def(&meta_graph_def);
    TF_RETURN_IF_ERROR(
        aot_options.graph_execution_options->runtime->CreateRuntimeResources(
            model_context));

    model_context.set_meta_graph_def(nullptr);
  }

  tfrt::BefBuffer bef;
  std::vector<std::string> xla_function_names;

  mlrt::bc::Buffer bytecode_buffer;
  if (aot_options.graph_execution_options->enable_mlrt) {
    mlir::OwningOpRef<mlir::ModuleOp> module_with_op_keys;

    ASSIGN_OR_RETURN_IN_COMPILE(
        bytecode_buffer,
        tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
            aot_options.graph_execution_options->compile_options,
            *fallback_state, mlir_module.get(), model_context,
            &module_with_op_keys, &xla_function_names));

    if (bytecode_buffer.empty()) {
      LOG(ERROR) << "MLRT byte buffer is empty.";
      return absl::InternalError("bytecode_buffer is empty.");
    }
  } else {
    RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
        aot_options.graph_execution_options->compile_options, mlir_module.get(),
        &bef, model_context, fallback_state.get(), &xla_function_names));
    if (bef.empty()) {
      LOG(ERROR) << "BEF byte buffer is empty.";
      return absl::InternalError("BefBuffer is empty.");
    }
  }

  const FunctionLibraryDefinition& flib_def = fallback_state->func_lib_def();
  std::vector<FunctionDef> xla_functions;
  xla_functions.reserve(xla_function_names.size());
  for (const std::string& name : xla_function_names) {
    const FunctionDef* xla_func_def = flib_def.Find(name);
    if (xla_func_def == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("XLA function ", name, " not found in library."));
    }
    xla_functions.push_back(*xla_func_def);
  }
  if (aot_options.graph_execution_options->enable_mlrt) {
    return AotResult{std::move(bytecode_buffer), std::move(xla_functions)};
  }
  return AotResult{std::move(bef), std::move(xla_functions)};
}

StatusOr<std::unique_ptr<xla::PjRtExecutable>> AotCompileToGpuPjRtExecutable(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    const stream_executor::GpuTargetConfigProto& gpu_target_config,
    XlaCompiler::CompilationResult** compilation_result) {
  XlaCompiler::Options options;
  TF_RETURN_IF_ERROR(CompileTfGraphToHlo(
      flib_def, function, graph_def_version, args, has_ref_vars,
      may_alias_resource_update, &options, compilation_result));

  xla::Compiler::TargetConfig gpu_config(gpu_target_config);
  xla::StreamExecutorGpuCompiler pjrt_gpu_compiler;
  // Create a trivial topology, which won't be used.
  xla::StreamExecutorGpuTopologyDescription topology(
      xla::CudaId(), xla::CudaName(), "fake_device", {0});
  xla::CompileOptions pjrt_options =
      GetPjRtCompileOptions(options, **compilation_result);
  pjrt_options.target_config = gpu_config;
  return pjrt_gpu_compiler.Compile(
      pjrt_options, *((*compilation_result)->computation), topology, nullptr);
}

StatusOr<std::string> AotCompileToGpuPjRtLoadedExecutableWithDevice(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    XlaCompiler::CompilationResult** compilation_result) {
  TF_ASSIGN_OR_RETURN(auto client,
                      xla::GetStreamExecutorGpuClient(xla::GpuClientOptions()));
  auto se_client = absl::WrapUnique(
      tensorflow::down_cast<xla::StreamExecutorGpuClient*>(client.release()));

  XlaCompiler::Options options;
  TF_RETURN_IF_ERROR(CompileTfGraphToHlo(
      flib_def, function, graph_def_version, args, has_ref_vars,
      may_alias_resource_update, &options, compilation_result));

  const xla::CompileOptions pjrt_options =
      GetPjRtCompileOptions(options, **compilation_result);
  TF_ASSIGN_OR_RETURN(
      auto executable,
      se_client->Compile(*((*compilation_result)->computation), pjrt_options));
  return se_client->SerializeExecutable(*executable);
}

StatusOr<AotResult::ExecutableMap> AotCompileXlaFunctionsInMetaGraphDef(
    const MetaGraphDef& meta_graph_def, const std::string& signature_name,
    const absl::flat_hash_map<std::string, tensorflow::TensorShapeProto>&
        input_shapes,
    const tensorflow::FunctionDefLibrary& fdef_lib,
    const tensorflow::SessionOptions& session_options,
    const mlir::DialectRegistry& registry, const AotOptions& aot_options,
    absl::string_view input_model_dir, ModelRuntimeContext& model_context) {
  // Make a copy since we need to modify the graph.
  MetaGraphDef input_meta_graph_def = meta_graph_def;
  TF_RETURN_IF_ERROR(UpdateGraphDefWithInputShapes(
      input_meta_graph_def, input_shapes, signature_name));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<tensorflow::tfrt_stub::FallbackState> fallback_state,
      FallbackState::CreateWithMockGpuDevice(session_options, fdef_lib));

  // Import the graph corresponding to `signature_name` into MLIR module.
  mlir::MLIRContext context(registry);
  TF_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> mlir_module,
      ImportSavedModel(
          &context, input_meta_graph_def, *fallback_state,
          std::string(input_model_dir),
          /*import_user_signatures=*/true,
          aot_options.graph_execution_options->run_placer_grappler_on_functions,
          {signature_name}));

  // Runs bridge pass.
  std::vector<std::string> xla_function_names;
  RETURN_IF_ERROR_IN_COMPILE(ConvertTfMlirToRuntimeExecutable(
      aot_options.graph_execution_options->compile_options, mlir_module.get(),
      [](mlir::PassManager& pm, mlir::ModuleOp module,
         const tensorflow::TfrtPipelineOptions& options) { return OkStatus(); },
      model_context, fallback_state.get(), &xla_function_names));

  AotResult::ExecutableMap result;
  const FunctionLibraryDefinition& flib_def = fallback_state->func_lib_def();
  // Compiles every exported XLA function.
  for (const std::string& name : xla_function_names) {
    const FunctionDef* xla_func_def = flib_def.Find(name);
    if (xla_func_def == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("XLA function ", name, " not found in library."));
    }

    NameAttrList func_attr_list = NameAttrList();
    std::vector<XlaCompiler::Argument> args;
    ConstructFunctionAndArgs(name, *xla_func_def, func_attr_list, args);

    XlaCompiler::CompilationResult out_compilation_result;
    XlaCompiler::CompilationResult* compilation_result =
        &out_compilation_result;
    TF_ASSIGN_OR_RETURN(
        std::string serialized_executable,
        AotCompileToGpuPjRtLoadedExecutableWithDevice(
            &flib_def, func_attr_list,
            input_meta_graph_def.graph_def().versions().producer(), args, false,
            false, &compilation_result));
    TF_ASSIGN_OR_RETURN(
        auto signature,
        DeviceCompilationClusterSignature::Build(func_attr_list, args));
    result.emplace(signature, serialized_executable);
  }
  return result;
}
}  // namespace tensorflow::tfrt_stub
