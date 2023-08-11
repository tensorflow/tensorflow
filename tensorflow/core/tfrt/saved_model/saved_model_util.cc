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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "tensorflow/cc/saved_model/reader.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/import_model.h"
#include "tensorflow/compiler/mlir/tfrt/saved_model/saved_model.h"
#include "tensorflow/compiler/mlir/tfrt/translate/import_model.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/monitoring/gauge.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_import_input.h"
#include "tensorflow/core/tfrt/saved_model/utils/serialize_bef_utils.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/path.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tfrt/bef/bef_buffer.h"  // from @tf_runtime

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

StatusOr<InitializersAndSignatures> GetInitializersAndSignatures(
    mlir::ModuleOp module) {
  InitializersAndSignatures result;

  // Create placeholders for initializers.
  for (auto session_initializer_name :
       mlir::tf_saved_model::GetSessionInitializerExportedName(module)) {
    Initializer initializer;
    initializer.name = session_initializer_name.str();
    result.initializers.push_back(std::move(initializer));
  }

  auto& signatures = result.signature_map;
  TF_RETURN_IF_ERROR(tensorflow::MapFunctionSignaturesFromTFSavedModelMLIR(
      module,
      [&signatures](const tensorflow::TFRTSavedModelSignatureInfo& sig_info) {
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
      }));

  return result;
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

StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> ImportSavedModel(
    mlir::MLIRContext* context, const tensorflow::MetaGraphDef& meta_graph_def,
    const FallbackState& fallback_state, std::string saved_model_dir,
    bool import_user_signatures, bool run_placer_grappler_on_functions) {
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
                          run_placer_grappler_on_functions));

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
  return tsl::io::JoinPath(std::string(saved_model_dir), kAoTPackagesDirectory);
}

std::string GetBEFFilePath(std::string aot_package_directory) {
  return tsl::io::JoinPath(aot_package_directory,
                           std::string(kBefBufferFilenameMLIRBEF));
}

// TODO(b/295241000): Implement MLIR deserialization to skip it AoT and remove
// redundant steps
absl::StatusOr<tfrt::BefBuffer> LoadAotPackages(
    const TfrtCompileOptions& options, mlir::ModuleOp mlir_module,
    const std::string& saved_model_dir,
    tfrt_stub::FallbackState* fallback_state) {
  const std::string aot_package_directory = GetAotPackagePath(saved_model_dir);
  const std::string bef_file_path =
      tfrt_stub::GetBEFFilePath(aot_package_directory);
  TF_ASSIGN_OR_RETURN(tfrt::BefBuffer bef, DeserializeBEFBuffer(bef_file_path));

  if (bef.empty()) {
    return absl::InternalError("BefBuffer is empty.");
  }
  // TODO (b/295241000): Currently AoT for TFRT only supports GPU so we only
  // check for GPU. Remove after MLIR deserialization.
  TF_RETURN_IF_ERROR(
      RunTFXLABridgeAndAddXlaFunctions(options, fallback_state, mlir_module));
  return bef;
}

}  // namespace tfrt_stub
}  // namespace tensorflow
