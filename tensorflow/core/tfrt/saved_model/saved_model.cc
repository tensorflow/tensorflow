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
#include "tensorflow/core/tfrt/saved_model/saved_model.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/metrics.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/monitoring/counter.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/fallback/op_kernel_runner.h"
#include "tensorflow/core/tfrt/graph_executor/export_mlir.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/executable.h"
#include "tensorflow/core/tfrt/mlrt/interpreter/context.h"
#include "tensorflow/core/tfrt/mlrt/kernel/batch_kernel.h"
#include "tensorflow/core/tfrt/mlrt/kernel/kernel.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"
#include "tensorflow/core/tfrt/saved_model/utils/serialize_utils.h"
#include "tensorflow/core/tfrt/stubs/model_config_stub.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/kernel_registry.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime
#include "tfrt/metrics/common_metrics.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr absl::string_view kSignatureJoiningDelimiter = "+";

auto* lazy_loading_count = monitoring::Counter<3>::New(
    "/tensorflow/tfrt/lazy_loading_count", "The total number of lazy loadings.",
    "model_name", "model_version", "use_graph_executor");

auto* saved_model_import_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/import_time",
        "Record the MLIR import time for the savedmodel.", "model_name");

auto* saved_model_compile_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/compile_time",
        "Record the compilation time for the savedmodel.", "model_name");

auto* saved_model_init_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/init_time",
        "Record the initialization time for the savedmodel.", "model_name");

// TODO(b/279197040) clean up this retention after input spec validation is
// enabled everywhere.
auto* saved_model_input_spec_validation_failure =
    tensorflow::monitoring::Gauge<bool, 1>::New(
        "/tensorflow/tfrt/saved_model/input_spec_validation_failure",
        "Record the models that failed input spec validation.", "model_name");

tensorflow::Status RunBytecodeInitializers(
    const GraphExecutionOptions& options,
    const InitializersAndSignatures& initializers_and_signatures,
    const mlrt::LoadedExecutable& loaded_executable,
    tfrt::ResourceContext* resource_context, OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array, FallbackState& fallback_state,
    const bool provide_inputs) {
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, /*run_options=*/{},
                        options.runtime->work_queue(), resource_context,
                        /*client_graph_resource_context=*/nullptr, runner_table,
                        resource_array, fallback_state,
                        fallback_state.process_function_library_runtime()));

  std::vector<tensorflow::Tensor> outputs;
  if (auto function = loaded_executable.GetFunction("_tfrt_fallback_init")) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs,
        /*sync_resource_state=*/nullptr));
  }

  for (const auto& p : initializers_and_signatures.initializers) {
    const auto& initializer_name = p.name;
    std::vector<tensorflow::Tensor> outputs;
    const std::vector<tensorflow::Tensor> empty_inputs;
    const std::vector<tensorflow::Tensor>& initializer_inputs =
        provide_inputs ? p.inputs : empty_inputs;
    TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
        options, /*run_options=*/{}, initializer_name, /*symbol_uids=*/{},
        nullptr, &loaded_executable, initializer_inputs, &outputs,
        resource_context,
        /*client_graph_resource_context=*/nullptr, runner_table, resource_array,
        *options.runtime, fallback_state,
        fallback_state.process_function_library_runtime(),
        /*req_deadline_tracker=*/nullptr,
        /*stream_callback_id=*/std::nullopt));
    DCHECK(outputs.empty());
  }

  if (auto function = loaded_executable.GetFunction("_tfrt_resource_init")) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs,
        /*sync_resource_state=*/nullptr));
  }

  return absl::OkStatus();
}

tensorflow::Status RunBefInitializers(
    const GraphExecutionOptions& options,
    const InitializersAndSignatures& initializers_and_signatures,
    tfrt::BEFFile* bef_file, tfrt::ResourceContext* resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array, FallbackState& fallback_state,
    const bool provide_inputs) {
  DCHECK(options.runtime);
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, /*run_options=*/{},
                        options.runtime->work_queue(), resource_context,
                        /*client_graph_resource_context=*/nullptr, runner_table,
                        resource_array, fallback_state,
                        fallback_state.process_function_library_runtime()));

  tfrt::ExecutionContext exec_ctx(request_info->tfrt_request_context);

  // Run "_tfrt_fallback_init" first to initialize fallback-specific states. It
  // is the special function created by compiler, which calls a sequence of
  // tfrt_fallback_async.createop to create all fallback ops used in this BEF.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_fallback_init"));

  for (const auto& p : initializers_and_signatures.initializers) {
    const auto& initializer_name = p.name;
    auto* func = bef_file->GetFunction(initializer_name);
    DCHECK(func);
    std::vector<tensorflow::Tensor> outputs;
    const std::vector<tensorflow::Tensor> empty_inputs;
    const std::vector<tensorflow::Tensor>& initializer_inputs =
        provide_inputs ? p.inputs : empty_inputs;
    TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
        options, /*run_options=*/{}, initializer_name, /*symbol_uids=*/{}, func,
        /*loaded_executable=*/nullptr, initializer_inputs, &outputs,
        resource_context,
        /*client_graph_resource_context=*/nullptr, runner_table, resource_array,
        *options.runtime, fallback_state,
        fallback_state.process_function_library_runtime(),
        /*req_deadline_tracker=*/nullptr,
        /*stream_callback_id=*/std::nullopt));
    DCHECK(outputs.empty());
  }

  // After we initialized all the resources in the original graph, we can run
  // the "_tfrt_resource_init" function to set these resources in runtime
  // states, so that later it can be efficiently retrieved without any
  // locking.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_resource_init"));

  return absl::OkStatus();
}

tensorflow::Status IsInputSpecsCorrect(
    absl::string_view name, const internal::Signature& signature,
    absl::Span<const tensorflow::Tensor> inputs) {
  TF_RET_CHECK(signature.input_specs.size() == inputs.size())
      << "signature " << name
      << " input size is wrong, expected: " << signature.input_specs.size()
      << ", actual: " << inputs.size();
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& expected_input_spec = signature.input_specs[i];
    TF_RET_CHECK(expected_input_spec.dtype == inputs[i].dtype())
        << "signature " << name
        << " input dtype is wrong, expected: " << expected_input_spec.dtype
        << ", actual: " << inputs[i].dtype();
    TF_RET_CHECK(expected_input_spec.shape.IsCompatibleWith(inputs[i].shape()))
        << "signature " << name
        << " input shape is wrong, expected : " << expected_input_spec.shape
        << ", actual: " << inputs[i].shape();
  }
  return absl::OkStatus();
}

tensorflow::Status CheckInputSpecs(
    const tensorflow::SessionMetadata& model_metadata,
    const SavedModel::RunOptions& run_options, absl::string_view signature_name,
    const internal::Signature& signature,
    absl::Span<const tensorflow::Tensor> input_tensors) {
  if (!run_options.validate_input_specs &&
      !run_options.validate_input_specs_dry_run) {
    return absl::OkStatus();
  }

  auto status = IsInputSpecsCorrect(signature_name, signature, input_tensors);
  if (!status.ok()) {
    saved_model_input_spec_validation_failure
        ->GetCell(
            absl::StrCat(model_metadata.name(), ":", model_metadata.version()))
        ->Set(true);
    const auto error_string = absl::StrCat(
        "model: ", model_metadata.name(),
        ", version: ", model_metadata.version(), ", error: ", status.message());
    if (!run_options.validate_input_specs_dry_run) {
      return tensorflow::errors::InvalidArgument(error_string);
    }
    LOG_EVERY_N_SEC(WARNING, 5)
        << "TFRT input specs validation failed, " << error_string;
  }

  return absl::OkStatus();
}

tensorflow::Status PreprocessSignature(
    const tensorflow::SessionMetadata& model_metadata,
    const SavedModel::RunOptions& run_options, absl::string_view signature_name,
    const tensorflow::SignatureDef& signature_def,
    const internal::Signature& signature,
    absl::Span<const tensorflow::Tensor> input_tensors,
    absl::flat_hash_set<std::string>* visited_feed_tensor_names,
    std::vector<std::pair<std::string, tensorflow::Tensor>>& inputs,
    std::vector<std::string>& output_tensor_names) {
  const auto& input_names = signature.input_names;

  TF_RETURN_IF_ERROR(CheckInputSpecs(model_metadata, run_options,
                                     signature_name, signature, input_tensors));

  TF_RET_CHECK(input_tensors.size() == signature_def.inputs().size())
      << "Incorrect input size for signature: " << signature_name
      << ": expected " << signature_def.inputs().size() << ", but got "
      << input_tensors.size();
  DCHECK_EQ(input_names.size(), signature_def.inputs().size());

  // Then we find out the corresponding tensor names (ie.
  // node_name:output_idx) for the inputs using the SignatureDef proto.
  //
  // TODO(tfrt-devs): Consider including tensor names in `signatures_` as
  // well, so that only `signatures_` is used here.
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& tensor_info = signature_def.inputs().at(input_names[i]);

    // TODO(b/184675681): Support other encoding cases.
    //
    // TODO(b/184679394): Add unit test for this check.
    TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
        << "Only dense tensor is supported, but got encoding case "
        << tensor_info.encoding_case();

    const auto& tensor_name = tensor_info.name();

    // Skip if we have visited the feed tensor. Otherwise, marked it as
    // visited and put it in the `flat_inputs`. Note that the following code
    // deduplicate inputs with the feed tensor names, and generates the flat
    // inputs in the same order.
    if (visited_feed_tensor_names &&
        !visited_feed_tensor_names->insert(tensor_name).second)
      continue;
    inputs.push_back(std::make_pair(tensor_name, input_tensors[i]));
  }

  for (const auto& output_key : signature.output_names) {
    const auto& tensor_info = signature_def.outputs().at(output_key);

    VLOG(1) << "Importing Signature Output: output_key = " << output_key
            << ", tensor_info = " << tensor_info.DebugString();

    TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
        << "Only dense tensor is supported, but got encoding case "
        << tensor_info.encoding_case();

    output_tensor_names.push_back(tensor_info.name());
  }

  return absl::OkStatus();
}

bool AotPackageExists(absl::string_view saved_model_dir) {
  Env* env = Env::Default();
  const std::string aot_package_path = GetAotPackagePath(saved_model_dir);
  const std::string aot_mlir_path = GetMlirFilePath(aot_package_path);
  const std::string aot_bef_path = GetBefFilePath(aot_package_path);
  return env->FileExists(aot_package_path).ok() &&
         env->FileExists(aot_mlir_path).ok() &&
         env->FileExists(aot_bef_path).ok();
}

}  // namespace

SavedModel::~SavedModel() = default;  // Out-of-line C++ key function.

tfrt::HostContext* SavedModel::GetHostContext() const {
  return runtime().core_runtime()->GetHostContext();
}

namespace {

// Gets the signatures from `signature_defs` and inserts them into `signatures`.
void GetSignaturesFromSignatureDef(
    SignatureMap& signatures,
    const google::protobuf::Map<std::string, tensorflow::SignatureDef>& signature_defs,
    const SavedModel::Options& options) {
  for (const auto& p : signature_defs) {
    const std::string& signature_name = p.first;
    const tensorflow::SignatureDef& signature_def = p.second;
    DCHECK(signatures.find(signature_name) == signatures.end());
    auto& signature = signatures[signature_name];

    signature.input_names.reserve(signature_def.inputs().size());
    signature.input_specs.reserve(signature_def.inputs().size());
    for (const auto& p : signature_def.inputs()) {
      const std::string& input_tensor_name = p.first;
      const tensorflow::TensorInfo& tensor_info = p.second;
      signature.input_names.push_back(input_tensor_name);
      signature.input_specs.push_back(
          TensorSpec(tensor_info.dtype(), tensor_info.tensor_shape()));
    }

    signature.input_devices = std::vector<std::string>(
        signature_def.inputs().size(),
        options.graph_execution_options.compile_options.default_device);

    signature.output_names.reserve(signature_def.outputs().size());
    signature.output_specs.reserve(signature_def.outputs().size());
    for (const auto& p : signature_def.outputs()) {
      const std::string& output_tensor_name = p.first;
      const tensorflow::TensorInfo& tensor_info = p.second;
      signature.output_names.push_back(output_tensor_name);
      signature.output_specs.push_back(
          TensorSpec(tensor_info.dtype(), tensor_info.tensor_shape()));
    }
  }
}

void GetDefaultInputValue(
    const google::protobuf::Map<std::string, tensorflow::SignatureDef>& signature_defs,
    ModelRuntimeContext& context, SignatureMap& signatures) {
  bool load_from_signature_def = false;
  for (const auto& [name, signature_def] : signature_defs) {
    auto itr = signatures.find(name);
    if (itr == signatures.end()) {
      continue;
    }
    LOG(INFO) << "Model signature identified for default inputs";
    if (signature_def.defaults().empty()) continue;
    LOG(INFO) << "Loading default inputs for signature: " << name
              << " from Signature def";
    load_from_signature_def = true;
    signatures[name].default_inputs = signature_def.defaults();
  }
  if (load_from_signature_def) return;
  GetDefaultInputsFromModelConfig(context, signatures);
}

void UpdateCompileOptions(SavedModel::Options& options) {
  // Disable DecomposeResourceOpsPass for now, as DecomposeResourceGather does
  // not work well with GPU (b/232819415).
  if (options.graph_execution_options.enable_tfrt_gpu) {
    options.graph_execution_options.compile_options.decompose_resource_ops =
        false;
  }

  options.graph_execution_options.compile_options
      .fuse_get_resource_ops_in_hoisting =
      !options.graph_execution_options.enable_mlrt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SavedModel>> SavedModelImpl::LoadSavedModel(
    Options options, absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags) {
  TF_ASSIGN_OR_RETURN(auto meta_graph_def,
                      ReadSavedModel(saved_model_dir, tags));
  return LoadSavedModel(std::move(options), std::move(meta_graph_def),
                        saved_model_dir);
}

absl::StatusOr<std::unique_ptr<SavedModel>> SavedModelImpl::LoadSavedModel(
    Options options, tensorflow::MetaGraphDef meta_graph_def,
    absl::string_view saved_model_dir) {
  LOG(INFO) << "TFRT loading v1 savedmodel: " << saved_model_dir;
  tfrt::metrics::AddTFRTVersionMetric();

  UpdateTpuTargetByBridgeCompatibility(options.graph_execution_options,
                                       meta_graph_def.graph_def());
  UpdateCompileOptions(options);
  const bool aot_exist = AotPackageExists(saved_model_dir);
  options.enable_lazy_loading = options.enable_lazy_loading && !aot_exist;

  if (aot_exist || options.aot_generation) {
    options.graph_execution_options.compile_options.saved_model_dir = "";
  } else {
    options.graph_execution_options.compile_options.saved_model_dir =
        saved_model_dir;
  }

  // Register TFRT dialects
  mlir::DialectRegistry registry;
  if (aot_exist) {
    LOG(INFO) << "Found AOT package. Register required dialects.";
    RegisterTfrtDialectsForAot(registry);
  }
  RegisterMlirDialect(
      registry,
      options.graph_execution_options.compile_options.backend_compiler);
  mlir::MLIRContext context(registry);

  // Step 1: Import saved model from a proto to an MLIR module.
  const auto import_start_time = absl::Now();
  auto session_options =
      CreateDefaultSessionOptions(options.graph_execution_options);
  // Set optimize_for_static_graph to true since we won't extend the graph
  // later. If optimize_for_static_graph is set to false, FallbackState will
  // keep an extra unused copy of the graph, which unnecessarily consumes
  // memory.
  session_options.config.mutable_experimental()->set_optimize_for_static_graph(
      true);
  LOG_FIRST_N(INFO, 10) << "SessionOptions: "
                        << session_options.config.DebugString();
  LOG_FIRST_N(INFO, 10) << "GraphExecutionOptions: "
                        << options.graph_execution_options;

  // Creating the fallback_state using the original function def library
  // without applying placer or grappler, it is OK for now because it's only
  // used for captured functions in certain tf.data ops
  const auto& fdef_lib = meta_graph_def.graph_def().library();

  std::unique_ptr<FallbackState> fallback_state;
  if (options.graph_execution_options.compile_options.device_target ==
      TfrtDeviceInfraTarget::kCpu) {
    ASSIGN_OR_RETURN_IN_IMPORT(
        fallback_state,
        FallbackState::CreateWithCpuDevice(session_options, fdef_lib));
  } else {
    ASSIGN_OR_RETURN_IN_IMPORT(
        fallback_state, FallbackState::Create(session_options, fdef_lib));
  }

  mlir::OwningOpRef<mlir::ModuleOp> mlir_module;
  if (aot_exist) {
    LOG(INFO) << "Found AOT package. Load and deserialize MLIR module.";

    TF_RETURN_IF_ERROR(
        DeserializeAoTMlirModule(saved_model_dir, &context, &mlir_module));
  } else {
    ASSIGN_OR_RETURN_IN_IMPORT(
        mlir_module,
        ImportSavedModel(
            &context, meta_graph_def, *fallback_state,
            std::string(saved_model_dir),
            /*import_user_signatures=*/!options.enable_lazy_loading,
            options.graph_execution_options.run_placer_grappler_on_functions));
  }
  // TODO(b/278143179): Upload module w/o control flow.
  SymbolUids symbol_uids;
  symbol_uids.tf_symbol_uid = MaybeUploadMlirToXsymbol(mlir_module.get());

  const std::string saved_model_dir_string = std::string(saved_model_dir);
  const auto import_duration = absl::Now() - import_start_time;
  saved_model_import_time_seconds->GetCell(saved_model_dir_string)
      ->Set(absl::ToInt64Seconds(import_duration));
  LOG(INFO) << "TFRT finished importing savedmodel. Took "
            << absl::ToInt64Milliseconds(import_duration) << " ms.";

  // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
  const auto compile_start_time = absl::Now();
  InitializersAndSignatures initializers_and_signatures;
  if (aot_exist || options.aot_generation) {
    ASSIGN_OR_RETURN_IN_COMPILE(
        initializers_and_signatures,
        GetInitializersAndSignatures(mlir_module.get(), saved_model_dir));
  } else {
    ASSIGN_OR_RETURN_IN_COMPILE(
        initializers_and_signatures,
        GetInitializersAndSignatures(mlir_module.get()));
  }

  // If lazy loading is enabled, the user signatures are not exported via MLIR
  // module, so we need to get them from the proto.
  // TODO(b/187228559): Unify the code paths for populating the signature map.
  if (options.enable_lazy_loading) {
    GetSignaturesFromSignatureDef(initializers_and_signatures.signature_map,
                                  meta_graph_def.signature_def(), options);
  }

  auto kernel_registry = std::make_unique<mlrt::KernelRegistry>();

  // Creates a ResourceContext and populate it with per model resource from
  // Runtime.
  auto resource_context = std::make_unique<tfrt::ResourceContext>();
  ModelRuntimeContext model_context(&options.graph_execution_options,
                                    std::string(saved_model_dir),
                                    resource_context.get());

  {
    CallableOptions callable_options =
        CombineSignatureDefs(meta_graph_def.signature_def());
    model_context.set_graph_def(&meta_graph_def.graph_def());
    model_context.set_callable_options(&callable_options);
    TF_RETURN_IF_ERROR(
        options.graph_execution_options.runtime->CreateRuntimeResources(
            model_context));
    // These are only needed for `CreateRuntimeResources`, and also safer
    // since meta_graph_def will be moved.
    model_context.set_graph_def(nullptr);
    model_context.set_callable_options(nullptr);
  }

  GetDefaultInputValue(meta_graph_def.signature_def(), model_context,
                       initializers_and_signatures.signature_map);

  mlrt::bc::Buffer bytecode;
  tfrt::BefBuffer bef;
  if (aot_exist) {
    LOG(INFO) << "Found AoT package. Load and deserialize BEF.";
    if (options.graph_execution_options.enable_mlrt) {
      LOG(INFO) << "Found AoT package. Load and deserialize MLRT Bytecode.";

      ASSIGN_OR_RETURN_IN_COMPILE(
          bytecode,
          LoadMlrtAndMlir(options.graph_execution_options.compile_options,
                          mlir_module.get(), saved_model_dir_string,
                          fallback_state.get()));

    } else {
      LOG(INFO) << "Found AoT package. Load and deserialize BEF.";

      ASSIGN_OR_RETURN_IN_COMPILE(
          bef, LoadBefAndMlir(options.graph_execution_options.compile_options,
                              mlir_module.get(), saved_model_dir_string,
                              fallback_state.get()));
      metrics::UpdateAotBefMlirLoadCount();
    }

  } else {
    tensorflow::tf_mlrt::RegisterTfMlrtKernels(*kernel_registry);
    tensorflow::tf_mlrt::RegisterTfMlrtBatchKernels(*kernel_registry);

    if (options.graph_execution_options.enable_mlrt) {
      ASSIGN_OR_RETURN_IN_COMPILE(
          bytecode, tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
                        options.graph_execution_options.compile_options,
                        *fallback_state, mlir_module.get(), model_context));
    } else {
      RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
          options.graph_execution_options.compile_options, mlir_module.get(),
          &bef, model_context, fallback_state.get()));
      if (options.graph_execution_options.compile_options
              .serialize_bef_to_aot_packages) {
        TF_RETURN_IF_ERROR(SerializeBEF(
            bef, options.graph_execution_options.compile_options.aot_bef_file));
      }
    }
  }

  ASSIGN_OR_RETURN_WITH_STAGE_INFO(
      "graph_executor creation", auto graph_executor,
      GraphExecutor::Create(options.graph_execution_options,
                            std::move(fallback_state),
                            std::move(resource_context),
                            std::move(*meta_graph_def.mutable_graph_def()),
                            std::move(kernel_registry)));

  symbol_uids.tfrt_symbol_uid = MaybeUploadMlirToXsymbol(mlir_module.get());
  const auto compile_duration = absl::Now() - compile_start_time;
  saved_model_compile_time_seconds->GetCell(saved_model_dir_string)
      ->Set(absl::ToInt64Seconds(compile_duration));
  LOG(INFO) << "TFRT finished compiling savedmodel. Took "
            << absl::ToInt64Milliseconds(compile_duration) << " ms.";

  // Step 3: Initialize runtime states using special BEF functions.
  const auto init_start_time = absl::Now();

  std::optional<mlrt::LoadedExecutable> loaded_executable;
  tfrt::RCReference<tfrt::BEFFile> bef_file;
  if (!bytecode.empty()) {
    loaded_executable.emplace(mlrt::bc::Executable(bytecode.data()),
                              graph_executor->kernel_registry());
  } else {
    DCHECK(!bef.empty());
    ASSIGN_OR_RETURN_IN_INIT(
        bef_file, tfrt::CreateBefFileFromBefBuffer(
                      *options.graph_execution_options.runtime, bef));
  }

  auto runner_table = std::make_unique<OpKernelRunnerTable>();
  auto resource_array = std::make_unique<tfd::FallbackResourceArray>();
  if (loaded_executable) {
    RETURN_IF_ERROR_IN_INIT(RunBytecodeInitializers(
        graph_executor->options(), initializers_and_signatures,
        *loaded_executable, &graph_executor->resource_context(),
        runner_table.get(), resource_array.get(),
        graph_executor->fallback_state(), aot_exist || options.aot_generation));
  } else {
    DCHECK(bef_file);
    RETURN_IF_ERROR_IN_INIT(RunBefInitializers(
        graph_executor->options(), initializers_and_signatures, bef_file.get(),
        &graph_executor->resource_context(), runner_table.get(),
        resource_array.get(), graph_executor->fallback_state(),
        aot_exist || options.aot_generation));
  }

  const auto init_duration = absl::Now() - init_start_time;
  saved_model_init_time_seconds->GetCell(saved_model_dir_string)
      ->Set(absl::ToInt64Seconds(init_duration));
  LOG(INFO) << "TFRT finished initializing savedmodel. Took "
            << absl::ToInt64Milliseconds(init_duration) << " ms.";

  if (aot_exist) {
    // Set persistent cache directory so that the binaries can be loaded from
    // the AOT directory.
    const std::string persistent_cache_directory =
        GetAotPackagePath(saved_model_dir);
    tensorflow::GetMarkForCompilationPassFlags()
        ->tf_xla_persistent_cache_directory = persistent_cache_directory;
    tensorflow::GetMarkForCompilationPassFlags()
        ->tf_xla_persistent_cache_read_only = true;
    LOG(INFO) << "Set persistent cache directory to "
              << persistent_cache_directory << ", and set it to read-only.";
  }

  // Finally, create the saved model.
  return {std::make_unique<SavedModelImpl>(
      std::move(options), std::move(symbol_uids), std::move(meta_graph_def),
      std::move(bef), std::move(bef_file), std::move(bytecode),
      std::move(loaded_executable),
      std::move(initializers_and_signatures.signature_map),
      std::move(runner_table), std::move(resource_array),
      std::move(graph_executor))};
}

SavedModelImpl::SavedModelImpl(
    Options options, SymbolUids symbol_uids,
    tensorflow::MetaGraphDef meta_graph_def, tfrt::BefBuffer bef,
    tfrt::RCReference<tfrt::BEFFile> bef_file, mlrt::bc::Buffer bytecode,
    std::optional<mlrt::LoadedExecutable> loaded_executable,
    SignatureMap signatures, std::unique_ptr<OpKernelRunnerTable> runner_table,
    std::unique_ptr<tfd::FallbackResourceArray> resource_array,
    std::unique_ptr<GraphExecutor> graph_executor)
    : SavedModel(std::move(options), std::move(graph_executor)),
      symbol_uids_(std::move(symbol_uids)),
      meta_graph_def_(std::move(meta_graph_def)),
      bef_(std::move(bef)),
      bef_file_(std::move(bef_file)),
      req_deadline_tracker_(
          options_.graph_execution_options.runtime->core_runtime()
              ->GetHostContext()),
      signatures_(std::move(signatures)),
      runner_table_(std::move(runner_table)),
      resource_array_(std::move(resource_array)) {
  if (!options_.enable_lazy_loading) {
    bytecode_ = std::move(bytecode);
    loaded_executable_ = std::move(loaded_executable);
  }
}

std::vector<std::string> SavedModelImpl::GetFunctionNames() const {
  std::vector<std::string> result;
  for (const auto& entry : signatures_) {
    result.push_back(entry.first);
  }
  return result;
}

const tensorflow::MetaGraphDef& SavedModelImpl::GetMetaGraphDef() const {
  return meta_graph_def_;
}

std::optional<FunctionMetadata> SavedModelImpl::GetFunctionMetadata(
    absl::string_view func_name) const {
  auto iter = signatures_.find(func_name);
  if (iter == signatures_.end()) return std::nullopt;
  return FunctionMetadata(&iter->second);
}

tensorflow::Status SavedModelImpl::Run(
    const RunOptions& run_options, absl::string_view name,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs) {
  TF_RET_CHECK(outputs) << "outputs must be provided";
  outputs->clear();

  auto sig_iter = signatures_.find(name);
  TF_RET_CHECK(sig_iter != signatures_.end())
      << "failed to find signature " << name << " in the graph";
  const auto& signature = sig_iter->second;
  const auto& signature_def = meta_graph_def_.signature_def().at(name);
  const tensorflow::SessionMetadata& model_metadata =
      options_.graph_execution_options.model_metadata;

  if (options_.enable_lazy_loading &&
      options_.lazy_loading_use_graph_executor) {
    lazy_loading_count
        ->GetCell(model_metadata.name(), absl::StrCat(model_metadata.version()),
                  "true")
        ->IncrementBy(1);

    std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
    input_tensors.reserve(inputs.size());

    std::vector<std::string> output_tensor_names;
    output_tensor_names.reserve(signature.output_names.size());

    TF_RETURN_IF_ERROR(PreprocessSignature(
        model_metadata, run_options, name, signature_def, signature, inputs,
        /*visited_feed_tensor_names=*/nullptr, input_tensors,
        output_tensor_names));

    auto run_opt = run_options;
    run_opt.name = name;

    return graph_executor_->Run(run_opt, input_tensors, output_tensor_names,
                                /*target_tensor_names=*/{}, outputs);
  }

  TF_RETURN_IF_ERROR(
      CheckInputSpecs(model_metadata, run_options, name, signature, inputs));

  const SymbolUids* symbol_uids = nullptr;
  const tfrt::Function* func = nullptr;
  const mlrt::LoadedExecutable* loaded_executable = nullptr;
  OpKernelRunnerTable* runner_table = nullptr;
  tfd::FallbackResourceArray* resource_array = nullptr;
  tfrt::ResourceContext* client_graph_resource_context = nullptr;
  if (options_.enable_lazy_loading) {
    // TODO(b/216379787): Remove this lazy loading path once b/279197040 is
    // unblocked.
    lazy_loading_count
        ->GetCell(model_metadata.name(), absl::StrCat(model_metadata.version()),
                  "false")
        ->IncrementBy(1);

    // If lazy loading is enabled, no signature is loaded into `bef_file_`, so
    // we need to find the BEF from the cache or create one.
    TF_ASSIGN_OR_RETURN(
        const LoadingResult& loading_result,
        GetOrCreateLoadingResult(run_options, {std::string(name)}));
    symbol_uids = &loading_result.symbol_uids;
    loaded_executable = loading_result.bytecode_executable.get();
    if (loaded_executable == nullptr) {
      func = loading_result.bef_file->GetFunction(loading_result.name);
    }
    runner_table = loading_result.runner_table.get();
    resource_array = loading_result.resource_array.get();
    client_graph_resource_context = loading_result.resource_context.get();
  } else {
    symbol_uids = &symbol_uids_;
    if (loaded_executable_) {
      loaded_executable = &(*loaded_executable_);
    } else {
      func = bef_file_->GetFunction(name);
    }
    runner_table = runner_table_.get();
    resource_array = resource_array_.get();
  }

  auto* resource_context = &graph_executor_->resource_context();
  DCHECK(runner_table);
  DCHECK(resource_array);

  return GraphExecutionRunOnFunction(
      options_.graph_execution_options, run_options, name, *symbol_uids, func,
      loaded_executable, inputs, outputs, resource_context,
      client_graph_resource_context, runner_table, resource_array, runtime(),
      fallback_state(), fallback_state().process_function_library_runtime(),
      &req_deadline_tracker_, /*stream_callback_id=*/std::nullopt);
}

struct SavedModelImpl::JoinedSignature {
  // A unique name for the joined signature.
  std::string name;
  // The feed nodes for the corresponding inputs, but they might not be in the
  // original order and if there are more than one original inputs mapped to the
  // same feed node, only one is picked here.
  tensorflow::GraphImportConfig::InputArrays input_nodes;
  // The fetch nodes for the outputs, which should be in the original order.
  std::vector<std::string> output_nodes;
  // The target nodes that should be run but not returned as outputs.
  std::vector<std::string> target_nodes;
};

tensorflow::Status SavedModelImpl::RunMultipleSignatures(
    const RunOptions& run_options, absl::Span<const std::string> names,
    absl::Span<const std::vector<tensorflow::Tensor>> multi_inputs,
    std::vector<std::vector<tensorflow::Tensor>>* multi_outputs) {
  TF_RET_CHECK(names.size() == multi_inputs.size())
      << "the sizes of names and inputs should be the same";
  TF_RET_CHECK(multi_outputs) << "outputs must be provided";
  multi_outputs->clear();

  // Due to possible overlapping of feed nodes among user-specified inputs, We
  // deduplicate against fetch tensor names and produce the desired inputs in a
  // new order. The same dedup logic is used here to generate the flattened
  // input values in the same order.
  //
  // Note that we don't need to do any deduplicating nor reordering for the
  // fetch nodes.
  std::vector<std::pair<std::string /*tensor_name*/, tensorflow::Tensor>>
      flat_inputs;
  std::vector<std::string> flat_output_names;
  absl::flat_hash_set<std::string> visited_feed_tensor_names;

  const auto& signature_defs = meta_graph_def_.signature_def();
  for (int i = 0; i < names.size(); ++i) {
    const auto& signature_name = names[i];
    const auto& input_tensors = multi_inputs[i];
    auto sig_iter = signature_defs.find(signature_name);

    // Early out if any signature can't be found.
    TF_RET_CHECK(sig_iter != signature_defs.end())
        << "failed to find signature in the graph";
    const auto& signature_def = sig_iter->second;

    // `signatures_` keeps the user-specified input names that is in the same
    // order as `input_tensors`.
    const auto& signature = signatures_.at(signature_name);

    TF_RETURN_IF_ERROR(PreprocessSignature(
        options_.graph_execution_options.model_metadata, run_options,
        signature_name, signature_def, signature, input_tensors,
        &visited_feed_tensor_names, flat_inputs, flat_output_names));
  }

  std::vector<tensorflow::Tensor> flat_outputs;

  TF_RETURN_IF_ERROR(
      graph_executor_->Run(run_options, flat_inputs, flat_output_names,
                           /*target_tensor_names=*/{}, &flat_outputs));

  // The outputs of the compiled function are in the user-specified order,
  // though they are flattened. So we just need to regroup the outputs for each
  // signature using the number of outputs of it.
  multi_outputs->resize(names.size());
  auto cur = flat_outputs.begin();
  for (size_t i = 0; i < names.size(); ++i) {
    const auto& signature_name = names[i];
    const size_t len = signature_defs.at(signature_name).outputs().size();
    std::move(cur, cur + len, std::back_inserter(multi_outputs->at(i)));
    cur += len;
    DCHECK_LE(std::distance(flat_outputs.begin(), cur), flat_outputs.size());
  }
  return absl::OkStatus();
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelImpl::ImportSubgraph(
    mlir::MLIRContext* context, absl::string_view name,
    const tensorflow::GraphImportConfig::InputArrays& input_nodes,
    const std::vector<std::string>& output_nodes,
    const std::vector<std::string>& target_nodes) {
  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.graph_func_name = name;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  graph_import_config.inputs = input_nodes;
  graph_import_config.outputs = output_nodes;
  graph_import_config.control_outputs = target_nodes;
  graph_import_config.set_original_tf_func_name = true;

  // Optimize the graph.
  TF_ASSIGN_OR_RETURN(
      auto optimization_result,
      graph_executor_->graph_execution_state().CreateOptimizedGraph(
          graph_import_config));

  // Convert the optimized graph to an MLIR module.
  return tensorflow::ConvertGraphToMlir(
      *optimization_result.graph, /*debug_info=*/{},
      optimization_result.graph->flib_def(), graph_import_config, context);
}

tensorflow::Status SavedModelImpl::RunByTensorNames(
    const RunOptions& run_options,
    absl::Span<const std::pair<std::string, tensorflow::Tensor>> inputs,
    absl::Span<const std::string> output_tensor_names,
    absl::Span<const std::string> target_node_names,
    std::vector<tensorflow::Tensor>* outputs) {
  // TODO(b/192498110): Validate input type.

  return graph_executor_->Run(run_options, inputs, output_tensor_names,
                              target_node_names, outputs);
}

namespace {

using JoinedSignature = SavedModelImpl::JoinedSignature;

// Returns a joined signature with the signatures in `names`. For inputs, as
// their corresponding nodes may overlap, we deduplicate them by the nodes so
// the order of inputs for the joined signature would be different from the
// original order. For outputs, overlapping is fine so we only flatten it in the
// original order.
absl::StatusOr<JoinedSignature> JoinSignatures(
    absl::Span<const std::string> names, const SignatureMap& signature_map,
    const tensorflow::protobuf::Map<std::string, tensorflow::SignatureDef>&
        signature_def_map) {
  // Join all the names, all the inputs, and all the outputs.
  JoinedSignature joined_signature;
  joined_signature.name = absl::StrJoin(names, kSignatureJoiningDelimiter);

  // Keep the feed tensor names visited.
  absl::flat_hash_set<std::string> visited_feed_tensor_names;

  for (const auto& name : names) {
    const auto& signature_def = signature_def_map.at(name);

    // For inputs, we deduplicate possible overlapping feed nodes and create the
    // new input array.
    for (const auto& iter : signature_def.inputs()) {
      const auto& tensor_info = iter.second;

      // Skip if this feed node is already visited.
      if (visited_feed_tensor_names.contains(tensor_info.name())) continue;

      // Otherwise, we parse its tensor info and collect it for later
      // compilation.
      visited_feed_tensor_names.insert(tensor_info.name());

      // TODO(b/184675681): Support other encoding cases.
      //
      // TODO(b/184679394): Add unit test for this check.
      TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
          << "Only dense tensor is supported, but got encoding case "
          << tensor_info.encoding_case();

      VLOG(1) << "Importing Signature Input: input_key = " << iter.first
              << ", tensor_info = " << tensor_info.DebugString();

      tensorflow::ArrayInfo array_info;
      array_info.imported_dtype = tensor_info.dtype();

      if (tensor_info.has_tensor_shape()) {
        array_info.shape = tensor_info.tensor_shape();
      } else {
        // If there is no tensor shape in the tensor info, conservatively set
        // unknown_rank to true.
        array_info.shape.set_unknown_rank(true);
      }

      joined_signature.input_nodes.insert(
          std::pair<std::string, tensorflow::ArrayInfo>(tensor_info.name(),
                                                        std::move(array_info)));
    }

    // For outputs, we simply flatten them in the original order, as it is fine
    // to have duplicated fetch nodes.
    const internal::Signature& signature = signature_map.at(name);
    for (const auto& output_key : signature.output_names) {
      const auto& tensor_info = signature_def.outputs().at(output_key);

      VLOG(1) << "Importing Signature Output: output_key = " << output_key
              << ", tensor_info = " << tensor_info.DebugString();

      TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
          << "Only dense tensor is supported, but got encoding case "
          << tensor_info.encoding_case();

      joined_signature.output_nodes.push_back(tensor_info.name());
    }
  }

  return joined_signature;
}

}  // namespace

// TODO(b/216379787): Reuse `GraphExecutor::LoadClientGraph()`.
absl::StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
SavedModelImpl::LoadJoinedSignature(const JoinedSignature& joined_signature) {
  // Step 1: Import the combined subgraph from proto to an MLIR module.
  mlir::DialectRegistry registry;
  RegisterMlirDialect(
      registry, graph_executor_->options().compile_options.backend_compiler);
  mlir::MLIRContext context(registry);

  ASSIGN_OR_RETURN_IN_IMPORT(auto module,
                             ImportSubgraph(&context, joined_signature.name,
                                            joined_signature.input_nodes,
                                            joined_signature.output_nodes,
                                            joined_signature.target_nodes));
  // TODO(b/278143179): Upload module w/o control flow.
  SymbolUids symbol_uids;
  symbol_uids.tf_symbol_uid = MaybeUploadMlirToXsymbol(module.get());

  // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
  auto loading_result = std::make_unique<LoadingResult>();
  loading_result->name = joined_signature.name;
  loading_result->runner_table = std::make_unique<OpKernelRunnerTable>();
  loading_result->resource_array =
      std::make_unique<tfd::FallbackResourceArray>();
  loading_result->resource_context = std::make_unique<tfrt::ResourceContext>();

  ModelRuntimeContext model_context(
      &options_.graph_execution_options,
      options_.graph_execution_options.compile_options.saved_model_dir,
      &graph_executor_->resource_context());

  if (options_.graph_execution_options.enable_mlrt) {
    ASSIGN_OR_RETURN_IN_COMPILE(
        loading_result->bytecode_buffer,
        tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
            options_.graph_execution_options.compile_options, fallback_state(),
            module.get(), model_context));
    mlrt::bc::Executable executable(loading_result->bytecode_buffer.data());
    loading_result->bytecode_executable =
        std::make_unique<mlrt::LoadedExecutable>(
            executable, graph_executor_->kernel_registry());
    RETURN_IF_ERROR_IN_INIT(RunBytecodeInitializers(
        graph_executor_->options(), /*initializers_and_signatures=*/{},
        *loading_result->bytecode_executable,
        &graph_executor_->resource_context(),
        loading_result->runner_table.get(),
        loading_result->resource_array.get(), fallback_state(),
        /*provide_inputs false for JIT compilation*/ false));
  } else {
    TF_RETURN_IF_ERROR(tensorflow::ConvertTfMlirToBef(
        options_.graph_execution_options.compile_options, module.get(),
        &loading_result->bef, model_context, &fallback_state()));
    ASSIGN_OR_RETURN_IN_COMPILE(
        loading_result->bef_file,
        tfrt::CreateBefFileFromBefBuffer(
            *options_.graph_execution_options.runtime, loading_result->bef));
    RETURN_IF_ERROR_IN_INIT(RunBefInitializers(
        graph_executor_->options(),
        /*initializers_and_signatures=*/{}, loading_result->bef_file.get(),
        &graph_executor_->resource_context(),
        loading_result->runner_table.get(),
        loading_result->resource_array.get(), fallback_state(),
        /*provide_inputs false for JIT compilation*/ false));
  }
  symbol_uids.tfrt_symbol_uid = MaybeUploadMlirToXsymbol(module.get());
  loading_result->symbol_uids = std::move(symbol_uids);

  // Store loading_result in cache.
  const auto* loading_result_ptr = loading_result.get();
  loading_result_cache_[joined_signature.name] = std::move(loading_result);
  return {*loading_result_ptr};
}

absl::StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
SavedModelImpl::GetOrCreateLoadingResult(const RunOptions& run_options,
                                         absl::Span<const std::string> names) {
  const auto joined_name = absl::StrJoin(names, kSignatureJoiningDelimiter);
  tensorflow::mutex_lock l(loading_result_cache_mu_);
  const auto iter = loading_result_cache_.find(joined_name);
  if (iter != loading_result_cache_.end()) return {*iter->second};

  if (run_options.disable_compilation) {
    return tensorflow::errors::InvalidArgument(
        absl::StrCat("GraphExecutor: compilation is disabled in execution but "
                     "the compiled graph is not found for ",
                     joined_name));
  }

  TF_ASSIGN_OR_RETURN(
      const auto joined_signature,
      JoinSignatures(names, signatures_, meta_graph_def_.signature_def()));

  LOG(INFO) << "TFRT loading joined signature " << joined_signature.name;
  absl::Cleanup log_finish([&joined_signature, start_time = absl::Now()]() {
    LOG(INFO) << "TFRT finished loading joined signature "
              << joined_signature.name << ". Took "
              << absl::ToInt64Milliseconds(absl::Now() - start_time) << " ms.";
  });

  return LoadJoinedSignature(joined_signature);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
