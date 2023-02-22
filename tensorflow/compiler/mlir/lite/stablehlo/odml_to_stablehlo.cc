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

#include <algorithm>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/InitAllDialects.h"  // from @llvm-project
#include "mlir/InitAllPasses.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/FileUtilities.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/compiler/mlir/init_mlir.h"
#include "tensorflow/compiler/mlir/lite/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/serializer/flatbuffer_export.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/check_accepted_ops_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/op_stat_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_tfl_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"
#include "tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantize_preprocess.h"
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_graph_optimization_pass.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir/framework/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/register.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

// Tool which lowers TensorFlow Graphs to StableHLO graphs.
//
// This tool is used by the ODML Programmability effort to consume input TF
// graphs, and lower them to StableHLO graphs for use by downstream tools.
//
// Input: TF Saved Model or TF MLIR files.
// Output: StableHLO MLIR or TFLite Flatbuffer.
//
// Usage:
// 1) convert the input model to StableHLO MLIR
// odml_to_stablehlo /path/to/model -o=model.mlir
// 2) convert the input model to TFLite Flatbuffer
// odml_to_stablehlo /path/to/model -o=model.tflite --export-type=tflite
//
// TODO(pulkitb): Add more options.
//  * Verbose - dump intermediate translations.
//  * Choose specific signatures/functions from a saved model.
//  * Options for full/partial conversion, Op exceptions list.
//  * Option to serialize output to TFL flatbuffer format.

using llvm::cl::opt;

// NOLINTNEXTLINE
opt<std::string> input_model(llvm::cl::Positional,
                             llvm::cl::desc("<input model path>"),
                             llvm::cl::Required);

// NOLINTNEXTLINE
opt<std::string> output_path("o", llvm::cl::desc("<output path>"),
                             llvm::cl::Required);

// NOLINTNEXTLINE
opt<std::string> export_type("export-type", llvm::cl::desc("<export type>"),
                             llvm::cl::Optional, llvm::cl::init("mlir"));

// NOLINTNEXTLINE
opt<bool> verbose(
    "v", llvm::cl::desc("Dump intermediate translations to output dir."),
    llvm::cl::Optional, llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> elide_large_elements_attrs(
    "e", llvm::cl::desc("Elide large elements attrs."), llvm::cl::Optional,
    llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> allow_tf("allow-tf", llvm::cl::desc("Allow TF dialect."),
                   llvm::cl::Optional, llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> skip_checks("skip-checks",
                      llvm::cl::desc("Skip checking for disallowed ops."),
                      llvm::cl::Optional, llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> skip_resize(
    "skip-resize",
    llvm::cl::desc(
        "Skip converting tf.ResizeBilinear and tf.ResizeNearestNeighbor ops."),
    llvm::cl::Optional, llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> smuggle_disallowed_ops(
    "smuggle-disallowed-ops",
    llvm::cl::desc("Smuggle disallowed ops via stablehlo.custom_calls."),
    llvm::cl::Optional, llvm::cl::init(false));

// NOLINTNEXTLINE
opt<bool> freeze_tf_graph(
    "freeze-tf-graph",
    llvm::cl::desc("Freeze TF graph to remove tf.ResourceVariable, etc."),
    llvm::cl::Optional, llvm::cl::init(false));

namespace mlir {
namespace odml {

tensorflow::StatusOr<OwningOpRef<mlir::ModuleOp>> ImportSavedModelOrMLIR(
    const std::string& input_path, MLIRContext* context,
    llvm::SourceMgr* source_mgr,
    std::unique_ptr<tensorflow::SavedModelBundle>* saved_model_bundle) {
  if (absl::EndsWith(input_path, ".mlir")) {
    auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_path.c_str());
    if (std::error_code error = file_or_err.getError()) {
      return tensorflow::errors::InvalidArgument(
          absl::StrCat("Could not open input file: ", input_path,
                       ", message=", error.message()));
    }

    // Load the MLIR module.
    source_mgr->AddNewSourceBuffer(std::move(*file_or_err), llvm::SMLoc());
    return mlir::parseSourceFile<mlir::ModuleOp>(*source_mgr, context);
  }

  // TODO(pulkitb): Remove hard-coded tag.
  std::unordered_set<std::string> tags({"serve"});
  auto exported_names_in_vector = std::vector<std::string>({});
  absl::Span<std::string> exported_names(exported_names_in_vector);
  std::vector<std::string> custom_opdefs;

  tensorflow::GraphImportConfig specs;
  specs.upgrade_legacy = true;

  return ImportSavedModel(input_path, /*saved_model_version=*/1, tags,
                          absl::MakeSpan(custom_opdefs), exported_names, specs,
                          /*enable_variable_lifting=*/true, context,
                          saved_model_bundle);
}

tensorflow::Status ConvertStableHLOToFlatbuffer(mlir::ModuleOp module,
                                                std::string* flatbuffer_str) {
  mlir::odml::FlatbufferExportOptions options;
  if (!mlir::odml::MlirToFlatBufferTranslateFunction(module, options,
                                                     flatbuffer_str)) {
    return tensorflow::errors::Aborted("Unable to export flatbuffer");
  }

  return ::tensorflow::OkStatus();
}

tensorflow::Status ExportModule(mlir::ModuleOp module,
                                const std::string& output_filename,
                                bool elide_large_elements_attrs) {
  std::string error_msg;
  auto output = mlir::openOutputFile(output_filename, &error_msg);
  if (output == nullptr) {
    llvm::errs() << error_msg << '\n';
    return tensorflow::errors::Aborted("Unable to write to output path.");
  }

  // Export TFLite Flatbuffer as output
  if (export_type == "tflite") {
    std::string flatbuffer_str;
    auto status =
        mlir::odml::ConvertStableHLOToFlatbuffer(module, &flatbuffer_str);
    if (!status.ok()) {
      return status;
    }

    output->os() << flatbuffer_str;
    output->keep();
    return ::tensorflow::OkStatus();
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

  return ::tensorflow::OkStatus();
}

tensorflow::Status ConvertTFToStableHLO(
    ModuleOp tf_module, const PassPipelineCLParser& pass_pipeline) {
  PassManager pm(tf_module.getContext());
  applyPassManagerCLOptions(pm);

  auto error_handler = [&](const Twine& msg) {
    emitError(UnknownLoc::get(pm.getContext())) << msg;
    return failure();
  };
  if (failed(pass_pipeline.addToPipeline(pm, error_handler))) {
    return tensorflow::errors::Aborted("Failed to add passes to pipeline.");
  }

  AddTFToStablehloPasses(pm, skip_resize, smuggle_disallowed_ops);

  if (!skip_checks) {
    std::vector<std::string> optional_accepted_dialects;
    if (allow_tf) {
      // For the normal case, this should be always FALSE.
      // This `allow_tf` logic should be removed after the backend supports tf.
      optional_accepted_dialects.push_back("tf");
    }
    pm.addPass(
        mlir::odml::createCheckAcceptedOpsPass(optional_accepted_dialects));
  }

  mlir::odml::AddStablehloOptimizationPasses(pm);

  if (verbose) {
    // Print out a detailed report of non-converted stats.
    // Because this pass aborts the pass if there are unconverted ops,
    // we need to locate createPrintOpStatsPass after all optimization.
    pm.addPass(mlir::odml::createPrintOpStatsPass());
  }

  if (failed(pm.run(tf_module))) {
    return tensorflow::errors::Aborted("Lowering to StableHLO failed.");
  }

  return ::tensorflow::OkStatus();
}

tensorflow::Status RunConverter(const PassPipelineCLParser& pass_pipeline) {
  DialectRegistry registry;
  registerAllDialects(registry);
  RegisterAllTensorFlowDialects(registry);
  mhlo::registerAllMhloDialects(registry);
  stablehlo::registerAllDialects(registry);
  registry.insert<mlir::func::FuncDialect, mlir::tf_type::TFTypeDialect,
                  mlir::quant::QuantizationDialect>();
  mlir::quant::RegisterOps();

  MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::SourceMgr source_mgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(source_mgr, &context);

  auto verbose_dir = llvm::sys::path::parent_path(output_path).str();

  const char* type_set[] = {"tflite", "mlir"};
  bool found = std::find(std::begin(type_set), std::end(type_set),
                         export_type) != std::end(type_set);
  if (!found) {
    return tensorflow::errors::Aborted("Export type is not supported.");
  }

  auto bundle = std::make_unique<tensorflow::SavedModelBundle>();
  bundle = nullptr;

  TF_ASSIGN_OR_RETURN(
      auto module,
      ImportSavedModelOrMLIR(input_model, &context, &source_mgr, &bundle));
  if (verbose) {
    TF_RETURN_IF_ERROR(ExportModule(*module,
                                    absl::StrCat(verbose_dir, "/debug_tf.mlir"),
                                    elide_large_elements_attrs));
  }

  llvm::Optional<tensorflow::Session*> session = std::nullopt;
  if (bundle) session = bundle->GetSession();  // NOMUTANTS--it should pass.

  if (freeze_tf_graph) {
    if (absl::EndsWith(input_model, ".mlir")) {
      LOG(INFO) << "file type is mlir. ignoring freezing TF graph";
    } else {
      LOG(INFO) << "--freeze-tf-graph is on. Freezing TF graph...";
      const auto status = tensorflow::quantization::PreprocessAndFreezeGraph(
          *module, module->getContext(), session);
      if (!status.ok()) {
        return tensorflow::errors::Aborted(
            "Failed to preprocess & freeze TF graph");
      }
    }
  }

  auto conversion_status = ConvertTFToStableHLO(*module, pass_pipeline);
  auto output_export_status =
      ExportModule(*module, output_path, elide_large_elements_attrs);
  if (!conversion_status.ok()) {
    LOG(ERROR) << "TF to StableHLO conversion failed: "
               << conversion_status.error_message();

    auto debug_export_status = ExportModule(
        *module, absl::StrCat(verbose_dir, "/debug_stablehlo.mlir"),
        elide_large_elements_attrs);
    if (!debug_export_status.ok()) {
      LOG(ERROR) << "Failed to export debug_stablehlo.mlir: "
                 << debug_export_status.error_message();
    }

    return conversion_status;
  }
  return output_export_status;
}

// All MLIR and TF passes are registered here, similar to mlirOptMain.
//
// This is done so users of the binary have the opportunity to experiment with
// the pipeline from the command line by adding additional passes when
// necessary.
//
// Once the pipeline stabilizes, and most models convert successfully this may
// be removed so only pipeline passes run.
void initAllPasses() {
  mlir::registerPassManagerCLOptions();
  mlir::registerAsmPrinterCLOptions();
  mlir::registerAllPasses();
  mlir::registerTensorFlowPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  // These are in compiler/mlir/xla and not part of the above MHLO passes.
  mlir::mhlo::registerTfXlaPasses();
  mlir::mhlo::registerLegalizeTFPass();
  mlir::mhlo::registerLegalizeTfTypesPassPass();
  mlir::xla_framework::registerXlaFrameworkPasses();
  tensorflow::RegisterConvertMlirToXlaHloPipelineWithDefaults();
  tensorflow::RegisterGraphOptimizationPasses();
}

}  // namespace odml
}  // namespace mlir

int main(int argc, char* argv[]) {
  tensorflow::InitMlir y(&argc, &argv);

  mlir::odml::initAllPasses();
  mlir::PassPipelineCLParser passPipeline("", "Add available compiler passes.");
  llvm::cl::ParseCommandLineOptions(argc, argv, "ODML StableHLO Bridge.\n");

  auto status = mlir::odml::RunConverter(passPipeline);

  if (!status.ok()) {
    LOG(ERROR) << status;
    return status.code();
  }
  return 0;
}
