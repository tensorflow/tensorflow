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
#include "tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.h"

#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/utils/utils.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {
void AttachCostPerDevice(mlir::ModuleOp module,
                         llvm::ArrayRef<std::string> device_specs) {
  std::set<std::string> processed_device_specs;
  for (const auto& device_spec : device_specs) {
    processed_device_specs.insert(
        mlir::TFL::tac::GetCanonicalHardwareName(device_spec));
  }
  processed_device_specs.insert("CPU");

  module.walk([&](mlir::Operation* op) {
    if (!mlir::TFL::tac::IsNonConstOp(op) &&
        !llvm::isa<func::ReturnOp, func::FuncOp, CallOpInterface>(op))
      return;

    // Attach cost per target.
    // Unsupported op will have negative values.
    mlir::SmallVector<mlir::NamedAttribute, 4> device_costs;
    for (const auto& device : processed_device_specs) {
      auto* target_hardware = mlir::TFL::tac::GetTargetHardware(device);
      float cost = -1;
      if (target_hardware->IsOpSupported(op)) {
        cost = target_hardware->GetOpCost(op);
      }

      mlir::StringAttr device_identifier =
          mlir::StringAttr::get(module.getContext(), device);
      auto float_type = mlir::FloatType::getF32(module.getContext());
      auto float_attr =
          mlir::FloatAttr::get(float_type, static_cast<float>(cost));
      device_costs.push_back({device_identifier, float_attr});
    }

    op->setAttr("per_device_costs",
                mlir::DictionaryAttr::get(module.getContext(), device_costs));
  });
}

}  // namespace

//////////// Importer ////////////
absl::StatusOr<OwningOpRef<mlir::ModuleOp>> TfLiteImporter::Import() {
  source_mgr_handler_ = std::make_unique<mlir::SourceMgrDiagnosticHandler>(
      source_mgr_, &context_);
  return ImportFlatbufferOrMlir(
      options_.file_name, options_.input_mlir,
      /*experimental_prune_unreachable_nodes_unconditionally=*/true,
      &source_mgr_, &context_);
}

//////////// Exporter ////////////
absl::Status TfLiteExporter::Export(mlir::ModuleOp module) {
  // return absl::OkStatus();
  if (options_.export_runtime_metadata) {
    // Run the cost model for each device/op.
    AttachCostPerDevice(module, options_.target_hardware_backends);

    // We will export the runtime metadata with the same name under the same
    // directory except with a different extention ".rtmeta".
    llvm::SmallString<128> metadata_filename(options_.output_file_name);
    const char kRuntimeMetadataName[] = "rtmeta";
    llvm::sys::path::replace_extension(metadata_filename, kRuntimeMetadataName);

    std::string error_msg;
    auto output = mlir::openOutputFile(metadata_filename, &error_msg);
    if (output == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("cannot open output file: ", error_msg));
    }
    auto result = tflite::ExportRuntimeMetadata(module);
    if (!result) {
      return absl::InvalidArgumentError("Cannot export runtime metadata.");
    }
    output->os() << result;
    output->keep();
  }

  return mlir::TFL::tac::ExportFlatbufferOrMlir(options_.output_file_name,
                                                options_.output_mlir, module,
                                                /*enable_select_tf_ops=*/false);
}

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
