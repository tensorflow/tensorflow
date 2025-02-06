/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/saved_model/saved_model_util.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/serialize_mlir_module_utils.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.h"
#include "tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/gpu_passes.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/tfrt_compile_options.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tensorflow/core/tfrt/mlrt/bytecode/bytecode.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_import_input.h"
#include "tensorflow/core/tfrt/saved_model/utils/serialize_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime
#include "tfrt/init_tfrt_dialects.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {

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

tensorflow::Tensor CreateScalarStringTensor(absl::string_view str) {
  return tensorflow::Tensor(tensorflow::tstring(str));
}

// Create the tensor for the bound input, which can be a variable or an asset.
//
// TODO(chky): For V2 models, the bound input can also be a resource.
absl::StatusOr<tensorflow::Tensor> CreateTensorFromBoundInput(
    mlir::Operation* bound_input, absl::string_view saved_model_dir) {
  // Assets are files in the saved model directory. We pass their filenames to
  // functions so that they can be used.
  if (auto asset = llvm::dyn_cast<mlir::tf_saved_model::AssetOp>(bound_input)) {
    // The filename in the asset is a relative path. So we prefix it with the
    // directory path.
    return CreateScalarStringTensor(
        tsl::io::JoinPath(saved_model_dir, asset.getFilename().str()));
  }

  return absl::AbortedError(
      "Failed to create captured tensors: unknown bound input type.");
}

absl::StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module, absl::string_view saved_model_dir) {
  InitializersAndSignatures result;

  const bool should_initialize_inputs = !saved_model_dir.empty();
  // A map for initializer inputs.
  absl::flat_hash_map<std::string, std::vector<tensorflow::Tensor>>
      initializer_input_map;

  // Create placeholders for initializers.
  for (auto session_initializer_name :
       mlir::tf_saved_model::GetSessionInitializerExportedName(module)) {
    Initializer initializer;
    initializer.name = session_initializer_name.str();
    if (should_initialize_inputs) initializer_input_map[initializer.name];
    result.initializers.push_back(std::move(initializer));
  }

  auto& signatures = result.signature_map;
  tensorflow::StatusGroup status_group;
  TF_RETURN_IF_ERROR(tensorflow::MapFunctionSignaturesFromTFSavedModelMLIR(
      module, [&status_group, &signatures, &initializer_input_map,
               saved_model_dir, should_initialize_inputs](
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

        if (should_initialize_inputs) {
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
        }
      }));

  if (!status_group.ok()) return status_group.as_concatenated_status();

  if (should_initialize_inputs) {
    for (auto& initializer : result.initializers) {
      initializer.inputs =
          std::move(initializer_input_map.at(initializer.name));
    }
  }

  return result;
}

absl::StatusOr<tensorflow::MetaGraphDef> ReadSavedModel(
    absl::string_view saved_model_dir,
    const std::unordered_set<std::string>& tags) {
  LOG(INFO) << "TFRT reading v1 savedmodel: " << saved_model_dir;
  const auto read_start_time = absl::Now();

  tensorflow::MetaGraphDef meta_graph_def;
  TF_RETURN_IF_ERROR(tensorflow::ReadMetaGraphDefFromSavedModel(
      saved_model_dir, tags, &meta_graph_def));

  const auto read_meta_graph_duration = absl::Now() - read_start_time;
  saved_model_read_meta_graph_time_seconds
      ->GetCell(std::string(saved_model_dir))
      ->Set(absl::ToInt64Seconds(read_meta_graph_duration));
  LOG(INFO) << "TFRT finished reading meta graph. Took "
            << absl::ToInt64Milliseconds(read_meta_graph_duration) << " ms.";
  return std::move(meta_graph_def);
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    mlir::MLIRContext* context, const tensorflow::MetaGraphDef& meta_graph_def,
    const FallbackState& fallback_state, std::string saved_model_dir,
    bool import_user_signatures, bool run_placer_grappler_on_functions,
    const std::vector<std::string>& import_signature_names,
    tensorflow::tfrt_stub::RuntimeConfig* runtime_config) {
  std::vector<std::string> signature_names;
  if (import_user_signatures) {
    if (!import_signature_names.empty()) {
      signature_names = import_signature_names;
    } else {
      signature_names = FindNamesForValidSignatures(meta_graph_def);
    }
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
                          run_placer_grappler_on_functions, runtime_config));

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

std::string GetAotPackagePath(absl::string_view saved_model_dir) {
  return tsl::io::JoinPath(std::string(saved_model_dir), kAotPackagesDirectory);
}

std::string GetBefFilePath(std::string aot_package_directory) {
  return tsl::io::JoinPath(aot_package_directory,
                           std::string(kBefBufferFileName));
}

std::string GetMlrtByteCodeFilePath(const std::string& aot_package_directory) {
  return tsl::io::JoinPath(aot_package_directory,
                           std::string(kMlrtBufferFileName));
}

std::string GetMlirFilePath(const std::string& aot_package_directory) {
  return tsl::io::JoinPath(aot_package_directory, kMlirModuleFilename);
}

absl::StatusOr<tfrt::BefBuffer> LoadBefAndMlir(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state) {
  const std::string aot_package_directory = GetAotPackagePath(saved_model_dir);
  const std::string bef_file_path =
      tfrt_stub::GetBefFilePath(aot_package_directory);
  TF_ASSIGN_OR_RETURN(tfrt::BefBuffer bef, DeserializeBEFBuffer(bef_file_path));

  if (bef.empty()) {
    return absl::InternalError("BefBuffer is empty.");
  }

  if (options.device_target == TfrtDeviceInfraTarget::kGpu) {
    TF_RETURN_IF_ERROR(AddXlaFunctions(fallback_state, mlir_module));
  }

  return bef;
}

absl::StatusOr<mlrt::bc::Buffer> LoadMlrtAndMlir(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state) {
  const std::string aot_package_directory = GetAotPackagePath(saved_model_dir);
  const std::string mlrt_file_path =
      tfrt_stub::GetMlrtByteCodeFilePath(aot_package_directory);
  TF_ASSIGN_OR_RETURN(mlrt::bc::Buffer mlrt_bytecode,
                      DeserializeMlrtBytecodeBuffer(mlrt_file_path));

  if (mlrt_bytecode.empty()) {
    return absl::InternalError("MLRT Bytecode is empty.");
  }

  if (options.device_target == TfrtDeviceInfraTarget::kGpu) {
    TF_RETURN_IF_ERROR(AddXlaFunctions(fallback_state, mlir_module));
  }

  return mlrt_bytecode;
}

absl::Status DeserializeAoTMlirModule(
    absl::string_view saved_model_dir, mlir::MLIRContext* context,
    mlir::OwningOpRef<mlir::ModuleOp>* mlir_module) {
  const std::string aot_package_directory = GetAotPackagePath(saved_model_dir);
  const std::string mlir_file_path = GetMlirFilePath(aot_package_directory);
  std::string mlir_module_str;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(), mlir_file_path,
                                           &mlir_module_str));
  TF_RETURN_IF_ERROR(
      DeserializeMlirModule(mlir_module_str, context, mlir_module));
  return absl::OkStatus();
}

CallableOptions CombineSignatureDefs(
    const google::protobuf::Map<std::string, SignatureDef>& signature_defs) {
  CallableOptions callable_options;
  for (const auto& sig_iter : signature_defs) {
    const auto& signature_def = sig_iter.second;

    for (const auto& p : signature_def.inputs()) {
      callable_options.add_feed(p.second.name());
    }
    for (const auto& p : signature_def.outputs()) {
      callable_options.add_fetch(p.second.name());
    }
  }
  return callable_options;
}

void RegisterTfrtDialectsForAot(mlir::DialectRegistry& registry) {
  tfrt::RegisterTFRTDialects(registry);
  registry.insert<tfrt::fallback::FallbackDialect>();
  registry.insert<tfrt::fallback_async::FallbackAsyncDialect>();
  tensorflow::RegisterGpuDialects(&registry);
}

}  // namespace tfrt_stub
}  // namespace tensorflow
