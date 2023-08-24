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

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "tensorflow/cc/saved_model/constants.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/xla/service/compiler.h"
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
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/file_system_helper.h"
#include "tensorflow/tsl/platform/status.h"
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

AotOptions::AotOptions() : graph_execution_options(nullptr) {}

Status AotCompileSavedModel(absl::string_view input_model_dir,
                            AotOptions aot_options,
                            absl::string_view output_model_dir) {
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

    aot_options.graph_execution_options =
        std::make_shared<GraphExecutionOptions>(graph_execution_options);
  }

  if (aot_options.tags.empty()) {
    aot_options.tags = {"serve", "gpu"};
  }

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
      FallbackState::Create(session_options, fdef_lib));
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
  RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
      aot_options.graph_execution_options->compile_options, mlir_module.get(),
      &bef, model_context, fallback_state.get()));
  if (bef.empty()) {
    LOG(DFATAL) << "BefBuffer is empty.";
    return absl::InternalError("BefBuffer is empty.");
  }

  Env* env = Env::Default();
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

  // Serialize MLIR to a file under aot_packages
  const std::string mlir_module_file =
      io::JoinPath(aot_directory, kMLIRModuleFilename);
  std::string mlir_module_string = SerializeMlirModule(mlir_module.get());
  TF_RETURN_IF_ERROR(
      WriteStringToFile(env, mlir_module_file, mlir_module_string));

  // Serialize BEF buffer to a file under aot_packages
  const std::string serialized_bef_path =
      io::JoinPath(aot_directory, kBefBufferFilenameMLIRBEF);
  TF_RETURN_IF_ERROR(SerializeBEF(bef, serialized_bef_path));

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

// TODO(b/294095043): Create a function (ex Status
// SerializeAotResult(AotResult)) to avoid using temp directories.

}  // namespace tensorflow::tfrt_stub
