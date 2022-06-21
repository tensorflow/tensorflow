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
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/upgrade_graph.h"
#include "tensorflow/compiler/mlir/tfrt/jit/tf_jitrt_request_context.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_execute_compat.h"
#include "tensorflow/core/runtime_fallback/util/tensor_util.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_import_input.h"
#include "tensorflow/core/tfrt/tpu/tpu_resources.h"
// TODO(b/200579737): using FunctionRegistry is simpler than the OSS trick.
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/tensor_util.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/tensor_handle.h"  // from @tf_runtime
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/metrics/common_metrics.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime
#include "tfrt/support/logging.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr absl::string_view kSignatureJoiningDelimiter = "+";
constexpr absl::string_view kTensorNameJoiningDelimiter = "-";
constexpr absl::string_view kArgumentTypeJoiningDelimiter = "^";

using SignatureMap = absl::flat_hash_map<std::string, internal::Signature>;
using ::tensorflow::SessionMetadata;
using ::tensorflow::StatusOr;

struct InitializersAndSignatures {
  llvm::SmallVector<std::string, 4> initializers;
  SignatureMap signature_map;
};

auto* saved_model_read_meta_graph_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/read_meta_graph_time",
        "Record the time of reading meta_graph from disk.", "model_name");

auto* saved_model_functionalization_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/functionalization_time",
        "Record the functionalization time for the savedmodel.", "model_name");

auto* saved_model_grappler_time_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/grappler_time",
        "Record the grappler time for the savedmodel.", "model_name");

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

tensorflow::Tensor CreateScalarStringTensor(absl::string_view str) {
  return tensorflow::Tensor(tensorflow::tstring(str));
}

// Create the tensor for the bound input, which can be a variable or an asset.
//
// TODO(chky): For V2 models, the bound input can also be a resource.
StatusOr<tensorflow::Tensor> CreateTensorFromBoundInput(
    mlir::Operation* bound_input, absl::string_view saved_model_dir,
    absl::flat_hash_map<std::string, tensorflow::Tensor>* variables) {
  // Assets are files in the saved model directory. We pass their filenames to
  // functions so that they can be used.
  if (auto asset = llvm::dyn_cast<mlir::tf_saved_model::AssetOp>(bound_input)) {
    // The filename in the asset is a relative path. So we prefix it with the
    // directory path.
    return CreateScalarStringTensor(
        tensorflow::io::JoinPath(saved_model_dir, asset.filename().str()));
  }

  return tensorflow::errors::Internal(
      "Failed to create captured tensors: unknown bound input type.");
}

StatusOr<SignatureMap> GetFunctionSignaturesFromTFSavedModelMLIR(
    absl::string_view saved_model_dir, mlir::ModuleOp module) {
  absl::flat_hash_map<std::string, tensorflow::Tensor> variables;
  SignatureMap signatures;

  tensorflow::StatusGroup status_group;
  TF_RETURN_IF_ERROR(tensorflow::MapFunctionSignaturesFromTFSavedModelMLIR(
      module, [&status_group, &variables, &signatures, saved_model_dir](
                  const tensorflow::TFRTSavedModelSignatureInfo& sig_info) {
        auto& signature = signatures[std::string(sig_info.func_name)];

        auto copy = [](llvm::ArrayRef<llvm::StringRef> src,
                       std::vector<std::string>* dst) {
          transform(src, std::back_inserter(*dst),
                    [](llvm::StringRef x) { return x.str(); });
        };
        copy(sig_info.input_names, &signature.input_names);
        copy(sig_info.output_names, &signature.output_names);
        copy(sig_info.input_devices, &signature.input_devices);

        DCHECK(signature.input_specs.empty());
        signature.input_specs.reserve(sig_info.input_specs.size());
        for (auto& spec : sig_info.input_specs) {
          signature.input_specs.push_back(TensorSpec(spec.first, spec.second));
        }

        DCHECK(signature.output_specs.empty());
        signature.output_specs.reserve(sig_info.output_specs.size());
        for (auto& spec : sig_info.output_specs) {
          signature.output_specs.push_back(TensorSpec(spec.first, spec.second));
        }

        for (auto* bound_input : sig_info.bound_inputs) {
          auto statusor_capture = CreateTensorFromBoundInput(
              bound_input, saved_model_dir, &variables);
          if (!statusor_capture.ok()) {
            status_group.Update(statusor_capture.status());
            // Insert a random tensor in case of errors.
            signature.captures.push_back(tensorflow::Tensor());
          } else {
            signature.captures.push_back(
                std::move(statusor_capture).ValueOrDie());
          }
        }
      }));

  if (!status_group.ok()) return status_group.as_concatenated_status();

  return signatures;
}

tensorflow::Status RunInitializers(
    const InitializersAndSignatures& initializers_and_signatures,
    const SessionMetadata& model_metadata, tfrt::BEFFile* bef_file,
    const Runtime& runtime, tfrt::ResourceContext* resource_context,
    const FallbackState& fallback_state) {
  auto* host = runtime.core_runtime()->GetHostContext();
  TF_ASSIGN_OR_RETURN(auto request_info,
                      SetUpRequestContext(/*run_options=*/{}, model_metadata,
                                          host, runtime.work_queue(),
                                          resource_context, fallback_state));

  tfrt::ExecutionContext exec_ctx(request_info->tfrt_request_context);

  // Run "_tfrt_fallback_init" first to initialize fallback-specific states. It
  // is the special function created by compiler, which calls a sequence of
  // tfrt_fallback_async.createop to create all fallback ops used in this BEF.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_fallback_init"));

  for (const auto& init : initializers_and_signatures.initializers) {
    // TODO(b/184771263): Consider using `GraphExecutionRunOnFunction()`
    // instead.

    auto* func = bef_file->GetFunction(init);
    assert(func);

    const auto& signature = initializers_and_signatures.signature_map.at(init);

    auto ready_chain = tfrt::GetReadyChain();

    // The actual arguments are the concat of side-effect chain and assets.
    llvm::SmallVector<tfrt::AsyncValue*, 1> arguments;
    auto cleanup = tensorflow::gtl::MakeCleanup([&]() {
      for (auto* argument : arguments) argument->DropRef();
    });

    arguments.push_back(ready_chain.release());

    for (const auto& capture : signature.captures) {
      arguments.push_back(
          tfrt::MakeAvailableAsyncValueRef<FallbackTensor>(capture).release());
    }

    assert(arguments.size() == func->argument_types().size());

    llvm::SmallVector<tfrt::RCReference<tfrt::AsyncValue>, 1> results;
    results.resize(func->result_types().size());
    assert(results.size() == 1);

    func->Execute(exec_ctx, arguments, results);

    // Wait for the function execution to finish, as well as the side-effects.
    host->Await(results);

    if (auto* error = results[0]->GetErrorIfPresent()) {
      return CreateTfErrorStatus(*error);
    }
  }

  // After we initialized all the resources in the original graph, we can run
  // the "_tfrt_resource_init" function to set these resources in runtime
  // states, so that later it can be efficiently retrieved without any locking.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_resource_init"));

  return OkStatus();
}

std::vector<std::string> FindNamesForValidSignatures(
    const tensorflow::MetaGraphDef& meta_graph_def) {
  std::vector<std::string> valid_signature_names;

  auto is_dense_tensor_info = [](const auto& named_tensor_info) {
    return !named_tensor_info.second.name().empty();
  };

  auto is_ref_type_tensor_info = [](const auto& named_tensor_info) {
    return tensorflow::IsRefType(named_tensor_info.second.dtype());
  };

  for (const auto& iter : meta_graph_def.signature_def()) {
    const auto& sig_key = iter.first;
    const auto& signature = iter.second;
    if (!std::all_of(signature.inputs().begin(), signature.inputs().end(),
                     is_dense_tensor_info) ||
        !std::all_of(signature.outputs().begin(), signature.outputs().end(),
                     is_dense_tensor_info)) {
      LOG(WARNING) << "Unsupported signature with non-dense tensors as "
                      "input/output. Name: "
                   << sig_key << "; Signature: " << signature.DebugString();
      continue;
    }
    if (std::any_of(signature.inputs().begin(), signature.inputs().end(),
                    is_ref_type_tensor_info) ||
        std::any_of(signature.outputs().begin(), signature.outputs().end(),
                    is_ref_type_tensor_info)) {
      LOG(WARNING) << "Unsupported signature with ref type tensors as "
                      "input/output. Name: "
                   << sig_key << "; Signature: " << signature.DebugString();
      continue;
    }
    valid_signature_names.push_back(sig_key);
  }
  return valid_signature_names;
}

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    mlir::MLIRContext* context, const tensorflow::MetaGraphDef& meta_graph_def,
    const FallbackState& fallback_state, std::string saved_model_dir,
    bool import_user_signatures, bool run_placer_grappler_on_functions,
    bool enable_tfrt_gpu) {
  std::vector<std::string> signature_names;
  if (import_user_signatures) {
    signature_names = FindNamesForValidSignatures(meta_graph_def);
    if (signature_names.empty())
      LOG(WARNING) << "No valid signature found for model: " << saved_model_dir;
  }

  // TfrtSavedModelMLIRImportInput basically implements the graph processing
  // logic (eg. Placer and Grappler) used in DirectSession, which apply graph
  // transformations on each subgraphs (ie. signatures). It is reusing the
  // code path in DirectSession to avoid problems caused by different behavior
  // in a different code path. And it is injected to the MLIR importer so that
  // the importer can import the transformed graph instead of the original
  // graph.
  TF_ASSIGN_OR_RETURN(auto import_input,
                      TfrtSavedModelMLIRImportInput::Create(
                          fallback_state, &meta_graph_def, /*debug_info=*/{},
                          run_placer_grappler_on_functions, enable_tfrt_gpu));

  TF_ASSIGN_OR_RETURN(
      auto module,
      tensorflow::ConvertSavedModelV1ToMlirLite(
          import_input,
          /*exported_names=*/absl::MakeSpan(signature_names), context));

  LOG(INFO) << "TFRT ImportSavedModel: Functionalization took "
            << absl::ToInt64Milliseconds(
                   import_input.GetFunctionalizationDuration())
            << " ms.";
  LOG(INFO) << "TFRT ImportSavedModel: Grappler took "
            << absl::ToInt64Milliseconds(import_input.GetGrapplerDuration())
            << " ms.";

  saved_model_functionalization_time_seconds->GetCell(saved_model_dir)
      ->Set(absl::ToInt64Seconds(import_input.GetFunctionalizationDuration()));

  saved_model_grappler_time_seconds->GetCell(saved_model_dir)
      ->Set(absl::ToInt64Seconds(import_input.GetGrapplerDuration()));

  return module;
}

StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module, absl::string_view saved_model_dir) {
  InitializersAndSignatures result;
  TF_ASSIGN_OR_RETURN(
      result.signature_map,
      GetFunctionSignaturesFromTFSavedModelMLIR(saved_model_dir, module));
  for (auto session_initializer_name :
       mlir::tf_saved_model::GetSessionInitializerExportedName(module)) {
    result.initializers.push_back(session_initializer_name.str());
  }
  return result;
}

tensorflow::Status InitSavedModel(
    const InitializersAndSignatures& initializers_and_signatures,
    tfrt::BEFFile* bef_file, const SavedModel::Options& options,
    tfrt::ResourceContext* resource_context,
    const FallbackState& fallback_state) {
  TF_RETURN_IF_ERROR(
      RunInitializers(initializers_and_signatures,
                      options.graph_execution_options.model_metadata, bef_file,
                      *options.graph_execution_options.runtime,
                      resource_context, fallback_state));

  return OkStatus();
}

}  // namespace

SavedModel::~SavedModel() {}

tfrt::HostContext* SavedModel::GetHostContext() const {
  return runtime_->core_runtime()->GetHostContext();
}

std::unique_ptr<SavedModel> SavedModelImpl::LoadSavedModel(
    Options options, absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags, tensorflow::Status* status) {
  LOG(INFO) << "TFRT reading v1 savedmodel: " << saved_model_dir;
  auto read_start_time = absl::Now();

  tensorflow::MetaGraphDef meta_graph_def;
  auto read_status = tensorflow::ReadMetaGraphDefFromSavedModel(
      std::string(saved_model_dir), tags, &meta_graph_def);

  if (!read_status.ok()) {
    *status = read_status;
    return nullptr;
  }

  auto read_meta_graph_duration = absl::Now() - read_start_time;
  saved_model_read_meta_graph_time_seconds
      ->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(read_meta_graph_duration));
  LOG(INFO) << "TFRT finished reading meta graph. Took "
            << absl::ToInt64Milliseconds(read_meta_graph_duration) << " ms.";

  return LoadSavedModel(std::move(options), saved_model_dir,
                        std::move(meta_graph_def), status);
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

void UpdateCompileOptions(SavedModel::Options& options) {
  // Disable DecomposeResourceOpsPass for now, as DecomposeResourceGather does
  // not work well with GPU (b/232819415).
  if (options.graph_execution_options.enable_tfrt_gpu) {
    options.graph_execution_options.compile_options.decompose_resource_ops =
        false;
  }
}

}  // namespace

std::unique_ptr<SavedModel> SavedModelImpl::LoadSavedModel(
    Options options, absl::string_view saved_model_dir,
    tensorflow::MetaGraphDef meta_graph_def, tensorflow::Status* status) {
  LOG(INFO) << "TFRT loading v1 savedmodel: " << saved_model_dir;
  tfrt::metrics::AddTFRTVersionMetric();

  UpdateTpuTargetByBridgeCompatibility(options.graph_execution_options,
                                       meta_graph_def.graph_def());
  UpdateCompileOptions(options);

  auto statusor_saved_model =
      [&]() -> tensorflow::StatusOr<std::unique_ptr<SavedModel>> {
    mlir::MLIRContext context;

    // Step 1: Import saved model from a proto to an MLIR module.
    auto import_start_time = absl::Now();
    auto session_options =
        CreateDefaultSessionOptions(options.graph_execution_options);
    // Set optimize_for_static_graph to true since we won't extend the graph
    // later. If optimize_for_static_graph is set to false, FallbackState will
    // keep an extra unused copy of the graph, which unnecessarily consumes
    // memory.
    session_options.config.mutable_experimental()
        ->set_optimize_for_static_graph(true);

    // Creating the fallback_state using the original function def library
    // without applying placer or grappler, it is OK for now because it's only
    // used for captured functions in certain tf.data ops
    const auto& fdef_lib = meta_graph_def.graph_def().library();
    TF_ASSIGN_OR_RETURN(auto fallback_state,
                        FallbackState::Create(session_options, fdef_lib));
    TF_ASSIGN_OR_RETURN(
        auto mlir_module,
        ImportSavedModel(
            &context, meta_graph_def, *fallback_state,
            std::string(saved_model_dir),
            /*import_user_signatures=*/!options.enable_lazy_loading,
            options.graph_execution_options.run_placer_grappler_on_functions,
            options.graph_execution_options.enable_tfrt_gpu));

    auto import_duration = absl::Now() - import_start_time;
    saved_model_import_time_seconds->GetCell(std::string(saved_model_dir))
        ->Set(absl::ToInt64Seconds(import_duration));
    LOG(INFO) << "TFRT finished importing savedmodel. Took "
              << absl::ToInt64Milliseconds(import_duration) << " ms.";

    // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
    auto compile_start_time = absl::Now();
    TF_ASSIGN_OR_RETURN(
        auto initializers_and_signatures,
        GetInitializersAndSignatures(mlir_module.get(), saved_model_dir));
    // If lazy loading is enabled, the user signatures are not exported via MLIR
    // module, so we need to get them from the proto.
    // TODO(b/187228559): Unify the code paths for populating the signature map.
    if (options.enable_lazy_loading) {
      GetSignaturesFromSignatureDef(initializers_and_signatures.signature_map,
                                    meta_graph_def.signature_def(), options);
    }
    tfrt::BefBuffer bef;
    TF_RETURN_IF_ERROR(tensorflow::ConvertTfMlirToBef(
        options.graph_execution_options.compile_options, mlir_module.get(),
        &bef));

    auto compile_duration = absl::Now() - compile_start_time;
    saved_model_compile_time_seconds->GetCell(std::string(saved_model_dir))
        ->Set(absl::ToInt64Seconds(compile_duration));
    LOG(INFO) << "TFRT finished compiling savedmodel. Took "
              << absl::ToInt64Milliseconds(compile_duration) << " ms.";

    // Step 3: Initialize runtime states using special BEF functions.
    auto init_start_time = absl::Now();
    TF_ASSIGN_OR_RETURN(auto bef_file,
                        tfrt::CreateBefFileFromBefBuffer(
                            *options.graph_execution_options.runtime, bef));

    auto tpu_model_resource = std::make_unique<tfrt::tpu::TpuModelResource>();
    auto resource_context = CreateResourceContext(
        *options.graph_execution_options.runtime, tpu_model_resource.get(),
        options.graph_execution_options.compile_options.tpu_target);
    TF_RETURN_IF_ERROR(InitSavedModel(initializers_and_signatures,
                                      bef_file.get(), options,
                                      resource_context.get(), *fallback_state));

    auto init_duration = absl::Now() - init_start_time;
    saved_model_init_time_seconds->GetCell(std::string(saved_model_dir))
        ->Set(absl::ToInt64Seconds(init_duration));
    LOG(INFO) << "TFRT finished initializing savedmodel. Took "
              << absl::ToInt64Milliseconds(init_duration) << " ms.";

    TF_ASSIGN_OR_RETURN(
        auto graph_executor,
        GraphExecutor::Create(options.graph_execution_options, *fallback_state,
                              tpu_model_resource.get(),
                              std::move(*meta_graph_def.mutable_graph_def())));

    // Finally, create the saved model.
    return {std::make_unique<SavedModelImpl>(
        std::move(options), std::move(meta_graph_def), std::move(bef),
        std::move(bef_file),
        std::move(initializers_and_signatures.signature_map),
        std::move(fallback_state), std::move(tpu_model_resource),
        std::move(resource_context), std::move(graph_executor))};
  }();

  if (!statusor_saved_model.ok()) {
    *status = statusor_saved_model.status();
    return nullptr;
  }
  *status = OkStatus();
  return std::move(statusor_saved_model).ValueOrDie();
}

SavedModelImpl::SavedModelImpl(
    Options options, tensorflow::MetaGraphDef meta_graph_def,
    tfrt::BefBuffer bef, tfrt::RCReference<tfrt::BEFFile> bef_file,
    SignatureMap signatures, std::unique_ptr<FallbackState> fallback_state,
    std::unique_ptr<tfrt::tpu::TpuModelResource> tpu_model_resource,
    std::unique_ptr<tfrt::ResourceContext> resource_context,
    std::unique_ptr<GraphExecutor> graph_executor)
    : SavedModel(options.graph_execution_options.runtime),
      options_(std::move(options)),
      meta_graph_def_(std::move(meta_graph_def)),
      bef_(std::move(bef)),
      bef_file_(std::move(bef_file)),
      req_deadline_tracker_(
          options.graph_execution_options.runtime->core_runtime()
              ->GetHostContext()),
      signatures_(std::move(signatures)),
      fallback_state_(std::move(fallback_state)),
      tpu_model_resource_(std::move(tpu_model_resource)),
      resource_context_(std::move(resource_context)),
      graph_executor_(std::move(graph_executor)) {}

SavedModelImpl::~SavedModelImpl() = default;

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

absl::optional<FunctionMetadata> SavedModelImpl::GetFunctionMetadata(
    absl::string_view func_name) const {
  auto iter = signatures_.find(func_name);
  if (iter == signatures_.end()) return absl::nullopt;
  return FunctionMetadata(&iter->second);
}

namespace {
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
  return OkStatus();
}
}  // namespace

tensorflow::Status SavedModelImpl::Run(
    const RunOptions& run_options, absl::string_view name,
    absl::Span<const tensorflow::Tensor> inputs,
    std::vector<tensorflow::Tensor>* outputs) {
  TF_RET_CHECK(outputs) << "outputs must be provided";
  outputs->clear();

  auto sig_iter = signatures_.find(name);
  TF_RET_CHECK(sig_iter != signatures_.end())
      << "failed to find signature " << name << " in the graph";
  if (run_options.validate_input_specs) {
    TF_RETURN_IF_ERROR(IsInputSpecsCorrect(name, sig_iter->second, inputs));
  }
  std::vector<tensorflow::Tensor> captures;
  for (const auto& capture : sig_iter->second.captures) {
    captures.push_back(capture);
  }

  const tfrt::Function* func;
  tfrt::ResourceContext* resource_context;
  if (options_.enable_lazy_loading) {
    // If lazy loading is enabled, no signature is loaded into `bef_file_`, so
    // we need to find the BEF from the cache or create one.
    TF_ASSIGN_OR_RETURN(const LoadingResult& loading_result,
                        GetOrCreateLoadingResult({std::string(name)}));
    func = loading_result.bef_file->GetFunction(
        tensorflow::kImportModelDefaultGraphFuncName);
    resource_context = loading_result.resource_context.get();
  } else {
    func = bef_file_->GetFunction({name.data(), name.size()});
    resource_context = resource_context_.get();
  }
  DCHECK(func);

  return GraphExecutionRunOnFunction(options_.graph_execution_options,
                                     run_options, name, *func, inputs, captures,
                                     outputs, resource_context, runtime(),
                                     *fallback_state_, req_deadline_tracker_);
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

  // Due to possible overlapping of feed nodes among user-specified inputs,
  // `JoinSignatures()` will deduplicate against fetch tensor names and produce
  // the desired inputs in a new order. The same dedup logic is used here to
  // generate the flattened input values in the same order.
  //
  // Note that we don't need to do any deduplicating nor reordering for the
  // fetch nodes.
  //
  // TODO(tfrt-devs): Consider refactoring JoinSignatures so that we don't have
  // the implicit requirement that the same dedup logic must be used here and in
  // JoinSignatures().
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
    const auto& input_names = signature.input_names;
    TF_RETURN_IF_ERROR(
        IsInputSpecsCorrect(signature_name, signature, input_tensors));
    DCHECK(signature.captures.empty());

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
    for (int j = 0; j < input_tensors.size(); ++j) {
      const auto& tensor_info = signature_def.inputs().at(input_names[j]);

      // TODO(b/184675681): Support other encoding cases.
      //
      // TODO(b/184679394): Add unit test for this check.
      TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
          << "Only dense tensor is supported, but got encoding case "
          << tensor_info.encoding_case();

      const auto& tensor_name = tensor_info.name();

      // Skip if we have visited the feed tensor. Otherwise, marked it as
      // visited and put it in the `flat_inputs`. Note that the following code
      // uses the same logic as in JoinSignatures() to deduplicate inputs with
      // the feed tensor names, and generates the flat inputs in the same order.
      if (visited_feed_tensor_names.contains(tensor_name)) continue;
      visited_feed_tensor_names.insert(tensor_name);
      flat_inputs.push_back(std::make_pair(tensor_name, input_tensors[j]));
    }

    for (const auto& output_key : signature.output_names) {
      const auto& tensor_info = signature_def.outputs().at(output_key);

      VLOG(1) << "Importing Signature Output: output_key = " << output_key
              << ", tensor_info = " << tensor_info.DebugString();

      TF_RET_CHECK(tensor_info.encoding_case() == tensorflow::TensorInfo::kName)
          << "Only dense tensor is supported, but got encoding case "
          << tensor_info.encoding_case();

      flat_output_names.push_back(tensor_info.name());
    }
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
  return OkStatus();
}

tensorflow::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>>
SavedModelImpl::ImportSubgraph(
    mlir::MLIRContext* context,
    const tensorflow::GraphImportConfig::InputArrays& input_nodes,
    const std::vector<std::string>& output_nodes,
    const std::vector<std::string>& target_nodes) {
  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  graph_import_config.inputs = input_nodes;
  graph_import_config.outputs = output_nodes;
  graph_import_config.control_outputs = target_nodes;

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
StatusOr<JoinedSignature> JoinSignatures(
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
StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
SavedModelImpl::LoadJoinedSignature(const JoinedSignature& joined_signature) {
  // Step 1: Import the combined subgraph from proto to an MLIR module.
  mlir::MLIRContext context;
  TF_ASSIGN_OR_RETURN(auto module,
                      ImportSubgraph(&context, joined_signature.input_nodes,
                                     joined_signature.output_nodes,
                                     joined_signature.target_nodes));

  // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
  auto loading_result = std::make_unique<LoadingResult>();
  loading_result->name = joined_signature.name;
  loading_result->resource_context = CreateResourceContext(
      runtime(), tpu_model_resource_.get(),
      options_.graph_execution_options.compile_options.tpu_target);

  TF_RETURN_IF_ERROR(tensorflow::ConvertTfMlirToBef(
      options_.graph_execution_options.compile_options, module.get(),
      &loading_result->bef));

  // Step 3: Initialize runtime states using special BEF functions.
  TF_ASSIGN_OR_RETURN(
      loading_result->bef_file,
      tfrt::CreateBefFileFromBefBuffer(
          *options_.graph_execution_options.runtime, loading_result->bef));
  TF_RETURN_IF_ERROR(RunInitializers(
      /*initializers_and_signatures=*/{},
      options_.graph_execution_options.model_metadata,
      loading_result->bef_file.get(), *options_.graph_execution_options.runtime,
      loading_result->resource_context.get(), *fallback_state_));

  // Store loading_result in cache.
  const auto* loading_result_ptr = loading_result.get();
  loading_result_cache_[joined_signature.name] = std::move(loading_result);
  return {*loading_result_ptr};
}

StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
SavedModelImpl::GetOrCreateLoadingResult(absl::Span<const std::string> names) {
  const auto joined_name = absl::StrJoin(names, kSignatureJoiningDelimiter);
  tensorflow::mutex_lock l(loading_result_cache_mu_);
  const auto iter = loading_result_cache_.find(joined_name);
  if (iter != loading_result_cache_.end()) return {*iter->second};

  TF_ASSIGN_OR_RETURN(
      const auto joined_signature,
      JoinSignatures(names, signatures_, meta_graph_def_.signature_def()));

  return LoadJoinedSignature(joined_signature);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
