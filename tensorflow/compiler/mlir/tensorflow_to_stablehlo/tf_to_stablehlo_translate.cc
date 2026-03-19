/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow_to_stablehlo/tf_to_stablehlo.h"

namespace {

using llvm::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_path(llvm::cl::Positional,
                            llvm::cl::desc("<input path>"), llvm::cl::Required);

// NOLINTNEXTLINE
opt<std::string> output_filename("o", llvm::cl::desc("<output path>"),
                                 llvm::cl::Optional, llvm::cl::init("-"));

// NOLINTNEXTLINE
opt<std::string> input_arg_shapes_str(
    "input-arg-shapes",
    llvm::cl::desc(
        "A string representation of input argument shapes for 'main' "
        "entry-point, separating tensors with ':', dimension with ',', and "
        "using '?' for unknown sizes. For example, 'input-arg-shapes=1,2::1,?' "
        "expresses argument shapes [1,2], [] and [1,?]"),
    llvm::cl::Optional, llvm::cl::init(""));

// NOLINTNEXTLINE
opt<std::string> exported_model_signatures(
    "exported-model-signatures",
    llvm::cl::desc(
        "Comma-separated list of exported model signatures to convert"),
    llvm::cl::Optional, llvm::cl::init("serving_default"));

// NOLINTNEXTLINE
opt<std::string> tag_names(
    "tags",
    llvm::cl::desc("Comma-separated list of tags for loading SavedModel. "
                   "Ignored for MLIR input"),
    llvm::cl::Optional, llvm::cl::init("serve"));

// NOLINTNEXTLINE
opt<bool> elide_large_elements_attrs(
    "e",
    llvm::cl::desc(
        "Elide large elements attrs while dumping the output StableHLO."),
    llvm::cl::Optional, llvm::cl::init(false));

}  // namespace

namespace mlir {

namespace {
// Dump the ModuleOp 'module' to the file specified using 'outputFileName'
absl::Status ExportModule(ModuleOp module) {
  std::string error_msg;
  auto output = openOutputFile(output_filename, &error_msg);
  if (output == nullptr) {
    return absl::AbortedError(
        absl::StrCat("Unable to write to output path: ", error_msg));
  }

  // Export StableHLO MLIR as output
  std::string result;
  llvm::raw_string_ostream os(result);
  OpPrintingFlags printing_flags;
  if (elide_large_elements_attrs) {
    printing_flags.elideLargeElementsAttrs();
  }
  module.print(os, printing_flags);
  os.flush();

  output->os() << result;
  output->keep();

  return absl::OkStatus();
}

}  // namespace
}  // namespace mlir

int main(int argc, char** argv) {
  tensorflow::InitMlir y(&argc, &argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TF Saved Model to Stablehlo converter\n");

  mlir::DialectRegistry registry;
  RegisterAllTensorFlowDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool is_input_mlir_module = absl::EndsWith(input_path, ".mlir");
  std::vector<std::string> exported_model_signatures_in_vector =
      absl::StrSplit(exported_model_signatures, ',');
  std::vector<std::string> tag_names_in_vector = absl::StrSplit(tag_names, ',');
  auto module = TfToStablehlo(
      input_path, &context, exported_model_signatures_in_vector,
      tag_names_in_vector, input_arg_shapes_str, is_input_mlir_module);
  if (!module.ok()) {
    llvm::errs() << module.status().ToString() << "\n";
    return module.status().raw_code();
  }

  return mlir::ExportModule(module->get()).raw_code();
}
