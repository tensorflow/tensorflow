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

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/compiler/jit/pjrt_device_compiler_client.h"
#include "tensorflow/compiler/jit/tf_graph_to_hlo_compiler.h"
#include "tensorflow/compiler/jit/xla_compiler_options_util.h"
#include "tensorflow/compiler/jit/xla_platform_info.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_compiler.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/gpu_target_config.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/export_mlir.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"
#include "tensorflow/core/tfrt/saved_model/utils/serialize_bef_utils.h"
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

AotOptions::AotOptions() : graph_execution_options(nullptr) {}

Status AotCompileSavedModelAndSaveResult(absl::string_view input_model_dir,
                                         AotOptions aot_options,
                                         absl::string_view output_model_dir) {
  // Create aot_packages directory.
  Env* env = Env::Default();
  const bool new_directory = !output_model_dir.empty();
  std::string output_dir;
  if (!new_directory) {
    output_dir = std::string(input_model_dir);
  } else {
    // TODO(chrisminge) modify to copy everything in input directory
    output_dir = std::string(output_model_dir);
    TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(output_dir, {}));
  }
  const std::string aot_directory =
      io::JoinPath(output_dir, kAoTPackagesDirectory);
  TF_RETURN_IF_ERROR(env->RecursivelyCreateDir(aot_directory));

  if (aot_options.graph_execution_options == nullptr) {
    // Since we are not going to actually run the model during AoT
    // compilation and optimization, we choose a value of 4 inter_op_threads
    // which is commonly used for testing.
    SetGlobalRuntime(tfrt_stub::Runtime::Create(/*num_inter_op_threads=*/4));

    GraphExecutionOptions graph_execution_options(GetGlobalRuntime());

    graph_execution_options.enable_tfrt_gpu = true;
    graph_execution_options.enable_grappler_function_optimizer = true;
    graph_execution_options.compile_options.enable_grappler = true;
    graph_execution_options.compile_options.device_target =
        TfrtDeviceInfraTarget::kGpu;
    graph_execution_options.compile_options.hoist_invariant_ops = true;
    graph_execution_options.compile_options
        .serialize_mlir_module_to_aot_packages = true;
    graph_execution_options.compile_options.aot_mlir_module_file =
        io::JoinPath(aot_directory, kMLIRModuleFilename);

    aot_options.graph_execution_options =
        std::make_shared<GraphExecutionOptions>(graph_execution_options);
  }

  if (aot_options.tags.empty()) {
    aot_options.tags = {"serve", "gpu"};
  }

  TF_ASSIGN_OR_RETURN(AotResult result,
                      AotCompileSavedModel(input_model_dir, aot_options));

  const std::string warmup_requests_path = io::JoinPath(
      input_model_dir, "assets.extra", "tf_serving_warmup_requests");
  TF_RETURN_IF_ERROR(env->FileExists(warmup_requests_path));

  const std::string saved_model_pb_path =
      io::JoinPath(input_model_dir, kSavedModelFilenamePb);
  const std::string saved_model_pbtxt_path =
      io::JoinPath(input_model_dir, kSavedModelFilenamePbTxt);
  bool pb_found = env->FileExists(saved_model_pb_path).ok();
  bool pbtxt_found = env->FileExists(saved_model_pbtxt_path).ok();
  if (!pb_found && !pbtxt_found) {
    return absl::NotFoundError(absl::StrCat(
        "saved_model not found in input directory: ", input_model_dir));
  }

  // Serialize BEF buffer to a file under aot_packages
  const std::string serialized_bef_path =
      io::JoinPath(aot_directory, kBefBufferFilenameMLIRBEF);
  TF_RETURN_IF_ERROR(SerializeBEF(result.bef, serialized_bef_path));

  if (pb_found) {
    const std::string output_file_directory =
        io::JoinPath(std::string(output_model_dir), kSavedModelFilenamePb);
    return env->CopyFile(saved_model_pb_path, output_file_directory);
  } else {
    const std::string output_file_directory =
        io::JoinPath(std::string(output_model_dir), kSavedModelFilenamePbTxt);
    return env->CopyFile(saved_model_pbtxt_path, output_file_directory);
  }
}

StatusOr<AotResult> AotCompileSavedModel(absl::string_view input_model_dir,
                                         AotOptions aot_options) {
  TF_ASSIGN_OR_RETURN(tensorflow::MetaGraphDef meta_graph_def,
                      ReadSavedModel(input_model_dir, aot_options.tags));

  UpdateTpuTargetByBridgeCompatibility(*aot_options.graph_execution_options,
                                       meta_graph_def.graph_def());
  UpdateCompileOptions(aot_options);
  aot_options.graph_execution_options->compile_options.saved_model_dir =
      input_model_dir;
  mlir::DialectRegistry registry;
  RegisterMlirDialect(registry);
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
  RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
      aot_options.graph_execution_options->compile_options, mlir_module.get(),
      &bef, model_context, fallback_state.get(), &xla_function_names));
  if (bef.empty()) {
    return absl::InternalError("BefBuffer is empty.");
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

  xla::gpu::GpuTargetConfig gpu_config(gpu_target_config);
  xla::StreamExecutorGpuCompiler pjrt_gpu_compiler(gpu_config);
  // Create a trivial topology, which won't be used.
  xla::StreamExecutorGpuTopologyDescription topology(
      xla::CudaId(), xla::CudaName(), "fake_device", {0});
  const xla::CompileOptions pjrt_options =
      GetPjRtCompileOptions(options, **compilation_result);
  return pjrt_gpu_compiler.Compile(
      pjrt_options, *((*compilation_result)->computation), topology, nullptr);
}

StatusOr<std::string> AotCompileToGpuPjRtLoadedExecutableWithDevice(
    const FunctionLibraryDefinition* flib_def, const NameAttrList& function,
    int graph_def_version, const std::vector<XlaCompiler::Argument>& args,
    bool has_ref_vars, bool may_alias_resource_update,
    XlaCompiler::CompilationResult** compilation_result) {
  TF_ASSIGN_OR_RETURN(auto client, xla::GetStreamExecutorGpuClient(
                                       true, /*allocator_config=*/{},
                                       /*node_id=*/0));
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
}  // namespace tensorflow::tfrt_stub
