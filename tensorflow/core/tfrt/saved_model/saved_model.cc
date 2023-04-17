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
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/compiler/transforms/import_model.h"
#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel/batch_kernel.h"
#include "learning/brain/experimental/tfrt/mlrt/application/tensorflow/kernel/kernel.h"
#include "learning/brain/experimental/tfrt/native_lowering/kernels/math_kernels.h"
#include "learning/infra/mira/mlrt/bytecode/bytecode.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/tf_mlir_translate.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_compat_request_state.h"
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/graph_executor/graph_executor.h"
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_import_input.h"
#include "tensorflow/core/tfrt/utils/error_util.h"
#include "tensorflow/core/tfrt/utils/fallback_tensor.h"
#include "tensorflow/core/tfrt/utils/utils.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/request_deadline_tracker.h"  // from @tf_runtime
#include "tfrt/metrics/common_metrics.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

constexpr absl::string_view kSignatureJoiningDelimiter = "+";

using SignatureMap = absl::flat_hash_map<std::string, internal::Signature>;
using ::tensorflow::SessionMetadata;
using ::tensorflow::StatusOr;

struct Initializer {
  std::string name;
  std::vector<tensorflow::Tensor> inputs;
};

struct InitializersAndSignatures {
  // Initializers are kept in a certain order as they need to be executed in
  // that order.
  std::vector<Initializer> initializers;
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

auto* saved_model_mla_check_time_milli_seconds =
    tensorflow::monitoring::Gauge<int64_t, 1>::New(
        "/tensorflow/tfrt/saved_model/mla_check_time",
        "Record the MLA check time for the savedmodel.", "model_name");

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

// TODO(b/239749833) clean up this retention after input spec validation is
// enabled everywhere.
auto* saved_model_input_spec_validation_failure =
    tensorflow::monitoring::Gauge<bool, 1>::New(
        "/tensorflow/tfrt/saved_model/input_spec_validation_failure",
        "Record the models that failed input spec validation.", "model_name");

tensorflow::Tensor CreateScalarStringTensor(absl::string_view str) {
  return tensorflow::Tensor(tensorflow::tstring(str));
}

// Create the tensor for the bound input, which can be a variable or an asset.
//
// TODO(chky): For V2 models, the bound input can also be a resource.
StatusOr<tensorflow::Tensor> CreateTensorFromBoundInput(
    mlir::Operation* bound_input, absl::string_view saved_model_dir) {
  // Assets are files in the saved model directory. We pass their filenames to
  // functions so that they can be used.
  if (auto asset = llvm::dyn_cast<mlir::tf_saved_model::AssetOp>(bound_input)) {
    // The filename in the asset is a relative path. So we prefix it with the
    // directory path.
    return CreateScalarStringTensor(
        tensorflow::io::JoinPath(saved_model_dir, asset.getFilename().str()));
  }

  return tensorflow::errors::Internal(
      "Failed to create captured tensors: unknown bound input type.");
}

StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module, absl::string_view saved_model_dir) {
  InitializersAndSignatures result;

  // A map for initializer inputs.
  absl::flat_hash_map<std::string, std::vector<tensorflow::Tensor>>
      initializer_input_map;

  // Create placeholders for initializers.
  for (auto session_initializer_name :
       mlir::tf_saved_model::GetSessionInitializerExportedName(module)) {
    Initializer initializer;
    initializer.name = session_initializer_name.str();
    initializer_input_map[initializer.name];
    result.initializers.push_back(std::move(initializer));
  }

  auto& signatures = result.signature_map;
  tensorflow::StatusGroup status_group;
  TF_RETURN_IF_ERROR(tensorflow::MapFunctionSignaturesFromTFSavedModelMLIR(
      module,
      [&status_group, &signatures, &initializer_input_map, saved_model_dir](
          const tensorflow::TFRTSavedModelSignatureInfo& sig_info) {
        auto signature_name = std::string(sig_info.func_name);
        auto& signature = signatures[signature_name];

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

        auto init_iter = initializer_input_map.find(signature_name);
        if (init_iter == initializer_input_map.end()) return;

        auto& init_inputs = init_iter->second;

        for (auto* bound_input : sig_info.bound_inputs) {
          auto capture =
              CreateTensorFromBoundInput(bound_input, saved_model_dir);
          if (!capture.ok()) {
            status_group.Update(capture.status());
            // Insert a random tensor in case of errors.
            init_inputs.push_back(tensorflow::Tensor());
          } else {
            init_inputs.push_back(*std::move(capture));
          }
        }
      }));

  if (!status_group.ok()) return status_group.as_concatenated_status();

  for (auto& initializer : result.initializers) {
    initializer.inputs = std::move(initializer_input_map.at(initializer.name));
  }

  return result;
}

tensorflow::Status RunBytecodeInitializers(
    const GraphExecutionOptions& options,
    const InitializersAndSignatures& initializers_and_signatures,
    const mlrt::LoadedExecutable& loaded_executable,
    tfrt::ResourceContext* resource_context, OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    const FallbackState& fallback_state) {
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, /*run_options=*/{},
                        options.runtime->work_queue(), resource_context,
                        /*client_graph_resource_context=*/nullptr, runner_table,
                        resource_array, fallback_state));

  std::vector<tensorflow::Tensor> outputs;
  if (auto function = loaded_executable.GetFunction("_tfrt_fallback_init")) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs));
  }

  for (const auto& p : initializers_and_signatures.initializers) {
    const auto& initializer_name = p.name;
    const auto& initializer_inputs = p.inputs;
    std::vector<tensorflow::Tensor> outputs;
    TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
        options, /*run_options=*/{}, initializer_name, nullptr,
        &loaded_executable, initializer_inputs, &outputs, resource_context,
        /*client_graph_resource_context=*/nullptr, runner_table, resource_array,
        *options.runtime, fallback_state,
        /*req_deadline_tracker=*/nullptr));
    DCHECK(outputs.empty());
  }

  if (auto function = loaded_executable.GetFunction("_tfrt_resource_init")) {
    TF_RETURN_IF_ERROR(RunMlrtFunction(
        function, loaded_executable, request_info->tfrt_request_context,
        *request_info->request_queue, {}, &outputs));
  }

  return OkStatus();
}

tensorflow::Status RunBefInitializers(
    const GraphExecutionOptions& options,
    const InitializersAndSignatures& initializers_and_signatures,
    tfrt::BEFFile* bef_file, tfrt::ResourceContext* resource_context,
    OpKernelRunnerTable* runner_table,
    tfd::FallbackResourceArray* resource_array,
    const FallbackState& fallback_state) {
  DCHECK(options.runtime);
  TF_ASSIGN_OR_RETURN(
      auto request_info,
      CreateRequestInfo(options, /*run_options=*/{},
                        options.runtime->work_queue(), resource_context,
                        /*client_graph_resource_context=*/nullptr, runner_table,
                        resource_array, fallback_state));

  tfrt::ExecutionContext exec_ctx(request_info->tfrt_request_context);

  // Run "_tfrt_fallback_init" first to initialize fallback-specific states. It
  // is the special function created by compiler, which calls a sequence of
  // tfrt_fallback_async.createop to create all fallback ops used in this BEF.
  TF_RETURN_IF_ERROR(
      RunRuntimeInitializer(exec_ctx, bef_file, "_tfrt_fallback_init"));

  for (const auto& p : initializers_and_signatures.initializers) {
    const auto& initializer_name = p.name;
    const auto& initializer_inputs = p.inputs;
    auto* func = bef_file->GetFunction(initializer_name);
    DCHECK(func);
    std::vector<tensorflow::Tensor> outputs;
    TF_RETURN_IF_ERROR(GraphExecutionRunOnFunction(
        options, /*run_options=*/{}, initializer_name, func,
        /*loaded_executable=*/nullptr, initializer_inputs, &outputs,
        resource_context,
        /*client_graph_resource_context=*/nullptr, runner_table, resource_array,
        *options.runtime, fallback_state,
        /*req_deadline_tracker=*/nullptr));
    DCHECK(outputs.empty());
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
    bool enable_tfrt_gpu, bool use_bridge_for_gpu) {
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
                          run_placer_grappler_on_functions, enable_tfrt_gpu,
                          use_bridge_for_gpu));

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

tensorflow::Status CheckInputSpecs(
    const tensorflow::SessionMetadata& model_metadata,
    const SavedModel::RunOptions& run_options, absl::string_view signature_name,
    const internal::Signature& signature,
    absl::Span<const tensorflow::Tensor> input_tensors) {
  if (!run_options.validate_input_specs &&
      !run_options.validate_input_specs_dry_run) {
    return OkStatus();
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
    LOG_EVERY_N_SEC(ERROR, 5)
        << "TFRT input specs validation failed, " << error_string;
  }

  return OkStatus();
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

  return OkStatus();
}

}  // namespace

SavedModel::~SavedModel() = default;  // Out-of-line C++ key function.

tfrt::HostContext* SavedModel::GetHostContext() const {
  return runtime_->core_runtime()->GetHostContext();
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
    // TODO(b/260915352): Remove this flag and use GPU bridge by default, and
    // remove the obsolete TFRT GPU runtime as well.
    options.graph_execution_options.compile_options.use_bridge_for_gpu = true;
  }

  options.graph_execution_options.compile_options
      .fuse_get_resource_ops_in_hoisting =
      !options.graph_execution_options.enable_mlrt;

  if (options.graph_execution_options.enable_mlrt) {
    options.lazy_loading_use_graph_executor = options.enable_lazy_loading;
    LOG(INFO) << "lazy_loading_use_graph_executor is updated to be the same as "
                 "enable_lazy_loading: "
              << options.enable_lazy_loading;
  }
}

StatusOr<tensorflow::MetaGraphDef> ReadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags) {
  LOG(INFO) << "TFRT reading v1 savedmodel: " << saved_model_dir;
  const auto read_start_time = absl::Now();

  tensorflow::MetaGraphDef meta_graph_def;
  TF_RETURN_IF_ERROR(tensorflow::ReadMetaGraphDefFromSavedModel(
      std::string(saved_model_dir), tags, &meta_graph_def));

  const auto read_meta_graph_duration = absl::Now() - read_start_time;
  saved_model_read_meta_graph_time_seconds
      ->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(read_meta_graph_duration));
  LOG(INFO) << "TFRT finished reading meta graph. Took "
            << absl::ToInt64Milliseconds(read_meta_graph_duration) << " ms.";
  return std::move(meta_graph_def);
}

}  // namespace

tensorflow::StatusOr<std::unique_ptr<SavedModel>>
SavedModelImpl::LoadSavedModel(Options options,
                               absl::string_view saved_model_dir,
                               const std::unordered_set<std::string>& tags) {
  TF_ASSIGN_OR_RETURN(auto meta_graph_def,
                      ReadSavedModel(saved_model_dir, tags));
  return LoadSavedModel(std::move(options), std::move(meta_graph_def),
                        saved_model_dir);
}

tensorflow::StatusOr<std::unique_ptr<SavedModel>>
SavedModelImpl::LoadSavedModel(Options options,
                               tensorflow::MetaGraphDef meta_graph_def,
                               absl::string_view saved_model_dir) {
  LOG(INFO) << "TFRT loading v1 savedmodel: " << saved_model_dir;
  tfrt::metrics::AddTFRTVersionMetric();

  UpdateTpuTargetByBridgeCompatibility(options.graph_execution_options,
                                       meta_graph_def.graph_def());
  UpdateCompileOptions(options);

  mlir::MLIRContext context;

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
  ASSIGN_OR_RETURN_IN_IMPORT(auto fallback_state,
                             FallbackState::Create(session_options, fdef_lib));
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto mlir_module,
      ImportSavedModel(
          &context, meta_graph_def, *fallback_state,
          std::string(saved_model_dir),
          /*import_user_signatures=*/!options.enable_lazy_loading,
          options.graph_execution_options.run_placer_grappler_on_functions,
          options.graph_execution_options.enable_tfrt_gpu,
          options.graph_execution_options.compile_options.use_bridge_for_gpu));

  const auto import_duration = absl::Now() - import_start_time;
  saved_model_import_time_seconds->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(import_duration));
  LOG(INFO) << "TFRT finished importing savedmodel. Took "
            << absl::ToInt64Milliseconds(import_duration) << " ms.";

  // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
  const auto compile_start_time = absl::Now();
  ASSIGN_OR_RETURN_IN_COMPILE(
      auto initializers_and_signatures,
      GetInitializersAndSignatures(mlir_module.get(), saved_model_dir));
  // If lazy loading is enabled, the user signatures are not exported via MLIR
  // module, so we need to get them from the proto.
  // TODO(b/187228559): Unify the code paths for populating the signature map.
  if (options.enable_lazy_loading) {
    GetSignaturesFromSignatureDef(initializers_and_signatures.signature_map,
                                  meta_graph_def.signature_def(), options);
  }
  mlrt::bc::Buffer bytecode;
  tfrt::BefBuffer bef;
  if (options.graph_execution_options.enable_mlrt) {
    ASSIGN_OR_RETURN_IN_COMPILE(
        bytecode, tensorflow::mlrt_compiler::ConvertTfMlirToBytecode(
                      options.graph_execution_options.compile_options,
                      mlir_module.get()));
  } else {
    RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
        options.graph_execution_options.compile_options, mlir_module.get(),
        &bef, fallback_state.get()));
  }

  const auto compile_duration = absl::Now() - compile_start_time;
  saved_model_compile_time_seconds->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(compile_duration));
  LOG(INFO) << "TFRT finished compiling savedmodel. Took "
            << absl::ToInt64Milliseconds(compile_duration) << " ms.";

  // Step 3: Initialize runtime states using special BEF functions.
  const auto init_start_time = absl::Now();

  auto kernel_registry = std::make_unique<mlrt::KernelRegistry>();
  // Register infra and standard math kernels
  tensorflow::tf_mlrt::RegisterTfMlrtKernels(*kernel_registry);
  tensorflow::tf_mlrt::RegisterTfMlrtBatchKernels(*kernel_registry);
  tfrt::cpu::RegisterMlrtMathKernels(kernel_registry.get());

  std::optional<mlrt::LoadedExecutable> loaded_executable;
  tfrt::RCReference<tfrt::BEFFile> bef_file;
  if (!bytecode.empty()) {
    loaded_executable.emplace(mlrt::bc::Executable(bytecode.data()),
                              *kernel_registry);
  } else {
    DCHECK(!bef.empty());
    ASSIGN_OR_RETURN_IN_INIT(
        bef_file, tfrt::CreateBefFileFromBefBuffer(
                      *options.graph_execution_options.runtime, bef));
  }

  auto runner_table = std::make_unique<OpKernelRunnerTable>();
  auto resource_array = std::make_unique<tfd::FallbackResourceArray>();

  ASSIGN_OR_RETURN_WITH_STAGE_INFO(
      "graph_executor creation", auto graph_executor,
      GraphExecutor::Create(options.graph_execution_options, *fallback_state,
                            std::move(*meta_graph_def.mutable_graph_def()),
                            std::move(kernel_registry)));

  if (loaded_executable) {
    RETURN_IF_ERROR_IN_INIT(RunBytecodeInitializers(
        graph_executor->options(), initializers_and_signatures,
        *loaded_executable, &graph_executor->resource_context(),
        runner_table.get(), resource_array.get(), *fallback_state));
  } else {
    DCHECK(bef_file);
    RETURN_IF_ERROR_IN_INIT(RunBefInitializers(
        graph_executor->options(), initializers_and_signatures, bef_file.get(),
        &graph_executor->resource_context(), runner_table.get(),
        resource_array.get(), *fallback_state));
  }

  const auto init_duration = absl::Now() - init_start_time;
  saved_model_init_time_seconds->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(init_duration));
  LOG(INFO) << "TFRT finished initializing savedmodel. Took "
            << absl::ToInt64Milliseconds(init_duration) << " ms.";

  // Finally, create the saved model.
  return {std::make_unique<SavedModelImpl>(
      std::move(options), std::move(meta_graph_def), std::move(bef),
      std::move(bef_file), std::move(bytecode), std::move(loaded_executable),
      std::move(initializers_and_signatures.signature_map),
      std::move(fallback_state), std::move(runner_table),
      std::move(resource_array), std::move(graph_executor))};
}

SavedModelImpl::SavedModelImpl(
    Options options, tensorflow::MetaGraphDef meta_graph_def,
    tfrt::BefBuffer bef, tfrt::RCReference<tfrt::BEFFile> bef_file,
    mlrt::bc::Buffer bytecode,
    std::optional<mlrt::LoadedExecutable> loaded_executable,
    SignatureMap signatures, std::unique_ptr<FallbackState> fallback_state,
    std::unique_ptr<OpKernelRunnerTable> runner_table,
    std::unique_ptr<tfd::FallbackResourceArray> resource_array,
    std::unique_ptr<GraphExecutor> graph_executor)
    : SavedModel(options.graph_execution_options.runtime),
      options_(std::move(options)),
      meta_graph_def_(std::move(meta_graph_def)),
      bef_(std::move(bef)),
      bef_file_(std::move(bef_file)),
      bytecode_(std::move(bytecode)),
      loaded_executable_(std::move(loaded_executable)),
      req_deadline_tracker_(
          options.graph_execution_options.runtime->core_runtime()
              ->GetHostContext()),
      signatures_(std::move(signatures)),
      fallback_state_(std::move(fallback_state)),
      runner_table_(std::move(runner_table)),
      resource_array_(std::move(resource_array)),
      graph_executor_(std::move(graph_executor)) {}

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

  if (options_.enable_lazy_loading &&
      options_.lazy_loading_use_graph_executor) {
    std::vector<std::pair<std::string, tensorflow::Tensor>> input_tensors;
    input_tensors.reserve(inputs.size());

    std::vector<std::string> output_tensor_names;
    output_tensor_names.reserve(signature.output_names.size());

    TF_RETURN_IF_ERROR(
        PreprocessSignature(options_.graph_execution_options.model_metadata,
                            run_options, name, signature_def, signature, inputs,
                            /*visited_feed_tensor_names=*/nullptr,
                            input_tensors, output_tensor_names));

    return graph_executor_->Run(run_options, input_tensors, output_tensor_names,
                                /*target_tensor_names=*/{}, outputs);
  }

  TF_RETURN_IF_ERROR(
      CheckInputSpecs(options_.graph_execution_options.model_metadata,
                      run_options, name, signature, inputs));

  const tfrt::Function* func = nullptr;
  const mlrt::LoadedExecutable* loaded_executable = nullptr;
  OpKernelRunnerTable* runner_table = nullptr;
  tfd::FallbackResourceArray* resource_array = nullptr;
  if (options_.enable_lazy_loading) {
    // TODO(b/216379787): Remove this lazy loading path once b/239749833 is
    // unblocked.

    // If lazy loading is enabled, no signature is loaded into `bef_file_`, so
    // we need to find the BEF from the cache or create one.
    TF_ASSIGN_OR_RETURN(
        const LoadingResult& loading_result,
        GetOrCreateLoadingResult(run_options, {std::string(name)}));
    func = loading_result.bef_file->GetFunction(
        tensorflow::kImportModelDefaultGraphFuncName);
    runner_table = loading_result.runner_table.get();
    resource_array = loading_result.resource_array.get();
  } else {
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
      options_.graph_execution_options, run_options, name, func,
      loaded_executable, inputs, outputs, resource_context,
      /*client_graph_resource_context=*/nullptr, runner_table, resource_array,
      runtime(), *fallback_state_, &req_deadline_tracker_);
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
  ASSIGN_OR_RETURN_IN_IMPORT(
      auto module, ImportSubgraph(&context, joined_signature.input_nodes,
                                  joined_signature.output_nodes,
                                  joined_signature.target_nodes));

  // Step 2: Compile the MLIR module from TF dialect to TFRT dialect (in BEF).
  auto loading_result = std::make_unique<LoadingResult>();
  loading_result->name = joined_signature.name;

  loading_result->runner_table = std::make_unique<OpKernelRunnerTable>();
  loading_result->resource_array =
      std::make_unique<tfd::FallbackResourceArray>();

  RETURN_IF_ERROR_IN_COMPILE(tensorflow::ConvertTfMlirToBef(
      options_.graph_execution_options.compile_options, module.get(),
      &loading_result->bef, fallback_state_.get()));

  // Step 3: Initialize runtime states using special BEF functions.
  ASSIGN_OR_RETURN_IN_INIT(
      loading_result->bef_file,
      tfrt::CreateBefFileFromBefBuffer(
          *options_.graph_execution_options.runtime, loading_result->bef));
  RETURN_IF_ERROR_IN_INIT(RunBefInitializers(
      graph_executor_->options(),
      /*initializers_and_signatures=*/{}, loading_result->bef_file.get(),
      &graph_executor_->resource_context(), loading_result->runner_table.get(),
      loading_result->resource_array.get(), *fallback_state_));

  // Store loading_result in cache.
  const auto* loading_result_ptr = loading_result.get();
  loading_result_cache_[joined_signature.name] = std::move(loading_result);
  return {*loading_result_ptr};
}

StatusOr<std::reference_wrapper<const SavedModelImpl::LoadingResult>>
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

  return LoadJoinedSignature(joined_signature);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
