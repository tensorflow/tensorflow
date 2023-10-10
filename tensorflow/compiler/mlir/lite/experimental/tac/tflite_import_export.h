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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TFLITE_IMPORT_EXPORT_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TFLITE_IMPORT_EXPORT_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "llvm/Support/SourceMgr.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_importer_exporter.h"

namespace mlir {
namespace TFL {
namespace tac {
// TAC Importer for TFLite.
// This import to MLIR from tflite file or MLIR
class TfLiteImporter : public mlir::TFL::tac::TacImporter {
 public:
  // Options for configuring the importer.
  struct Options {
    std::string file_name;
    // Whether the input file is an MLIR not tflite file.
    bool input_mlir = false;
  };

  explicit TfLiteImporter(const Options& options) : options_(options) {}

  absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> Import() override;

 private:
  Options options_;
  mlir::MLIRContext context_;
  llvm::SourceMgr source_mgr_;
  std::unique_ptr<mlir::SourceMgrDiagnosticHandler> source_mgr_handler_;
};

// TAC Exporter. It exports the provided Module to a tflite file.
class TfLiteExporter : public mlir::TFL::tac::TacExporter {
 public:
  // Exporter configuration options.
  struct Options {
    bool export_runtime_metadata = false;
    bool output_mlir = false;
    std::string output_file_name;
    std::vector<std::string> target_hardware_backends;
  };

  explicit TfLiteExporter(const Options& options) : options_(options) {}

  absl::Status Export(mlir::ModuleOp module) override;

 private:
  Options options_;
};
}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TFLITE_IMPORT_EXPORT_H_
