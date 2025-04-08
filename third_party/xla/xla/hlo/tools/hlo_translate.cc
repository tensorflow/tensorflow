/* Copyright 2024 The OpenXLA Authors.

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
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "stablehlo/transforms/Passes.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_proto_util.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "tsl/platform/protobuf.h"

namespace {

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_mhlo("emit-mhlo",
                              llvm::cl::desc("Translate to MHLO instead of "
                                             "default StableHLO"),
                              llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> emit_proto("emit-proto",
                               llvm::cl::desc("Emit HLO proto instead of text"),
                               llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_layouts(
    "print-layouts", llvm::cl::desc("Print layouts in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_large_constants(
    "print-large-constants",
    llvm::cl::desc("Print large constants in the generated HLO text"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> print_sugar(
    "print-sugar",
    llvm::cl::desc(
        "Print async ops using syntactic sugar in the generated HLO text"),
    llvm::cl::init(true));

// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tsl::protobuf::io::ErrorCollector {
 public:
  void AddError(int line, int column, const std::string& message) override {}
};

bool LoadHloProto(const std::string& contents, xla::HloProto* hlo_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

constexpr char kLoadHloError[] = "Failed to parse HLO.";

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetModuleFromHLOText(
    absl::string_view content, mlir::MLIRContext* context) {
  auto hlo_text = xla::ParseAndReturnUnverifiedModule(
      content, {}, xla::HloParserOptions().set_keep_module_auto_layouts(true));
  if (!hlo_text.ok()) return absl::InvalidArgumentError(kLoadHloError);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));
  auto hlo_module = std::move(hlo_text.value());
  auto status = ConvertHloToMlirHlo(*module, hlo_module.get(),
                                    /*import_all_computations=*/true,
                                    /*flatten_computation_args_result*/ true);
  if (!status.ok()) return status;
  return module;
}

absl::StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GetModuleFromHLOProto(
    const std::string& content, mlir::MLIRContext* context) {
  xla::HloProto hlo_proto;
  if (!LoadHloProto(content, &hlo_proto))
    return absl::InvalidArgumentError(kLoadHloError);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      xla::llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));
  auto status =
      ConvertHloToMlirHlo(module.get(), hlo_proto.mutable_hlo_module(),
                          /*import_all_computations=*/true,
                          /*flatten_computation_args_result=*/true);
  if (!status.ok()) return status;
  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> GetModuleFromHloInput(
    const std::shared_ptr<llvm::SourceMgr>& source_mgr,
    mlir::MLIRContext* context) {
  const llvm::MemoryBuffer* input =
      source_mgr->getMemoryBuffer(source_mgr->getMainFileID());
  absl::string_view content =
      absl::string_view(input->getBufferStart(), input->getBufferSize());

  // Emit error using file location 0.
  auto emitError = [&]() {
    auto loc =
        mlir::FileLineColLoc::get(context, input->getBufferIdentifier(), 0, 0);
    return mlir::emitError(loc);
  };

  // Try HLO Text
  auto module_from_text = GetModuleFromHLOText(content, context);
  if (module_from_text.ok()) return std::move(module_from_text.value());
  if (module_from_text.status().message() != kLoadHloError) {
    emitError() << "Failed to convert HLO to MLIR: "
                << module_from_text.status().message();
    return nullptr;
  }

  // Try HLO Proto
  auto module_from_proto = GetModuleFromHLOProto(std::string(content), context);
  if (module_from_proto.ok()) return std::move(module_from_proto.value());
  if (module_from_proto.status().message() != kLoadHloError) {
    emitError() << "Failed to convert HLO to MLIR: "
                << module_from_proto.status().message();
    return nullptr;
  }

  // Failed to parse
  emitError() << "Failed to parse input as HLO text or proto.";
  return nullptr;
}

}  // namespace

static mlir::OwningOpRef<mlir::ModuleOp> HloToMlirTranslate(
    const std::shared_ptr<llvm::SourceMgr>& sourceMgr,
    mlir::MLIRContext* context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      GetModuleFromHloInput(sourceMgr, context);

  if (!module) return nullptr;

  if (emit_mhlo) return module;

  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (failed(pm.run(*module))) {
    module->emitError("Failed to legalize to StableHLO");
    return nullptr;
  }

  return module;
}

static mlir::LogicalResult MlirToHloTranslate(mlir::ModuleOp mlir_module,
                                              llvm::raw_ostream& output) {
  // Also support portable artifacts in tooling, no-op if module is already
  // StableHLO.
  mlir::PassManager pm(mlir_module.getContext());
  mlir::stablehlo::createStablehloDeserializePipeline(pm);
  if (failed(pm.run(mlir_module))) {
    mlir_module->emitError("Failed to deserialize StableHLO");
    return mlir::failure();
  }

  // Convert to HLO
  auto hlo_module_or_status = xla::ConvertStablehloToHlo(mlir_module);
  if (!hlo_module_or_status.ok()) {
    mlir_module->emitError(hlo_module_or_status.status().message());
    LOG(ERROR) << "Module conversion failed: " << hlo_module_or_status.status();
    return mlir::failure();
  }
  xla::HloModule* hlo_module = hlo_module_or_status.value().get();
  if (emit_proto) {
    // Print as HloProto with empty BufferAssignment for legacy compatibility.
    output << MakeHloProto(*hlo_module).DebugString();
  } else {
    // Print as HLO text.
    output << hlo_module->ToString(
        xla::HloPrintOptions()
            .set_include_layout_in_shapes(print_layouts)
            .set_syntax_sugar_async_ops(print_sugar)
            .set_print_large_constants(print_large_constants));

    // Output alias information as comments in the HLO text.
    hlo_module->input_output_alias_config().ForEachAlias(
        [&](const xla::ShapeIndex& output_index,
            const xla::HloInputOutputAliasConfig::Alias& alias) {
          output << "// OutputIndex " << output_index.ToString()
                 << " aliases with input " << alias.parameter_number << " at "
                 << alias.parameter_index.ToString() << "\n";
        });
  }
  return mlir::success();
}

static mlir::TranslateToMLIRRegistration HloToMlirTranslateRegistration(
    "hlo-to-mlir", "hlo to mlir translation", HloToMlirTranslate);

static mlir::TranslateFromMLIRRegistration MlirToHloTranslateRegistration(
    "mlir-to-hlo", "mlir to hlo translation", MlirToHloTranslate,
    xla::RegisterMlirToHloDependentDialects);

int main(int argc, char** argv) {
  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR<->HLO translation driver\n"));
}
