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

#include <memory>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Tools/mlir-translate/Translation.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/execution_metadata_exporter.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_module.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tflite_import_export.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/utils/utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/platform/init_main.h"

using llvm::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_file_name(llvm::cl::Positional,
                                 llvm::cl::desc("<input file>"),
                                 llvm::cl::init("-"));

// NOLINTNEXTLINE
opt<std::string> output_file_name("o", llvm::cl::desc("<output file>"),
                                  llvm::cl::value_desc("filename"),
                                  llvm::cl::init("-"));
// NOLINTNEXTLINE
opt<bool> input_mlir("input-mlir",
                     llvm::cl::desc("Input is MLIR rather than FlatBuffer"),
                     llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> output_mlir(
    "output-mlir",
    llvm::cl::desc(
        "Output MLIR rather than FlatBuffer for the generated TFLite model"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> inline_subgraphs(
    "inline-subgraphs",
    llvm::cl::desc("Whether or not to inline all the subgraphs"),
    llvm::cl::init(true));

// NOLINTNEXTLINE
opt<std::string> device_specs(
    "device-specs", llvm::cl::desc("comma separated list of device specs."),
    llvm::cl::init(""));

// NOLINTNEXTLINE
opt<bool> export_runtime_metadata(
    "export-runtime-metadata",
    llvm::cl::desc("Whether or not to export metadata, if yes, the metadata "
                   "file will be exported along with the output model"),
    llvm::cl::init(false));

namespace {

std::unique_ptr<mlir::TFL::tac::TacImporter> CreateTfLiteImporter() {
  mlir::TFL::tac::TfLiteImporter::Options options;
  options.file_name = input_file_name;
  options.input_mlir = input_mlir;
  return std::make_unique<mlir::TFL::tac::TfLiteImporter>(options);
}

std::unique_ptr<mlir::TFL::tac::TacExporter> CreateTfLiteExporter(
    const std::vector<std::string>& target_hardware_backends) {
  mlir::TFL::tac::TfLiteExporter::Options options;
  options.output_mlir = output_mlir;
  options.output_file_name = output_file_name;
  options.export_runtime_metadata = export_runtime_metadata;
  options.target_hardware_backends = target_hardware_backends;
  return std::make_unique<mlir::TFL::tac::TfLiteExporter>(options);
}

absl::Status TargetAwareConversionMain() {
  std::vector<std::string> device_specs_array =
      absl::StrSplit(device_specs, ',', absl::SkipEmpty());
  mlir::TFL::tac::TacModule::Options options;
  options.hardware_backends = device_specs_array;
  options.debug_mode = true;
  if (!output_mlir || inline_subgraphs) {
    options.debug_mode = false;
  }
  options.enable_inliner = true;
  options.legalize_to_tflite_ops = true;
  mlir::TFL::tac::TacModule tac_module(options);
  mlir::DialectRegistry registry;
  mlir::RegisterAllTensorFlowDialects(registry);
  tac_module.RegisterExtraDialects(registry);
  tac_module.SetImporter(CreateTfLiteImporter());
  tac_module.SetExporter(CreateTfLiteExporter(options.hardware_backends));
  return tac_module.Run();
}
}  // namespace

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);

  llvm::cl::ParseCommandLineOptions(argc, argv, "Target aware conversion\n");

  absl::Status status = TargetAwareConversionMain();
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
  return 0;
}
