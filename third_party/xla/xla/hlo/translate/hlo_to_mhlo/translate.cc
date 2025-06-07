/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/translate/hlo_to_mhlo/translate.h"

#include <string>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/PassManager.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/stablehlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "tsl/platform/protobuf.h"

namespace xla {

namespace {
// Error collector that simply ignores errors reported.
class NoOpErrorCollector : public tsl::protobuf::io::ErrorCollector {
 public:
  void RecordError(int line, tsl::protobuf::io::ColumnNumber column,
                   absl::string_view message) override {}
};

bool LoadHloProto(const std::string& contents, HloProto* hlo_proto) {
  tsl::protobuf::TextFormat::Parser parser;
  NoOpErrorCollector collector;
  parser.RecordErrorsTo(&collector);
  return hlo_proto->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto) ||
         hlo_proto->mutable_hlo_module()->ParseFromString(contents) ||
         parser.ParseFromString(contents, hlo_proto->mutable_hlo_module());
}

}  // namespace

mlir::OwningOpRef<mlir::ModuleOp> HloToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations, bool flatten_computation_args_result,
    bool emit_stablehlo) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));

  HloProto hlo_proto;
  std::string content(input.data(), input.size());
  if (!LoadHloProto(content, &hlo_proto)) {
    module->emitError("Failed to load proto");
    return nullptr;
  }

  auto status = ConvertHloToMlirHlo(
      module.get(), hlo_proto.mutable_hlo_module(), import_all_computations,
      flatten_computation_args_result, emit_stablehlo);
  if (!status.ok()) {
    module->emitError("Hlo module import failed: ") << status.message();
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> HloTextToMlirHloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context,
    bool import_all_computations, bool flatten_computation_args_result,
    bool emit_stablehlo) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      llvm_ir::CreateMlirModuleOp(mlir::UnknownLoc::get(context));

  std::string content(input.data(), input.size());
  auto hlo_module_error = ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    module->emitError("HLO Module loading failed: ")
        << hlo_module_error.status().message();
    return nullptr;
  }

  auto hlo_module = std::move(hlo_module_error.value());
  auto status =
      ConvertHloToMlirHlo(*module, hlo_module.get(), import_all_computations,
                          flatten_computation_args_result, emit_stablehlo);
  if (!status.ok()) {
    module->emitError("HLO Module import failed: ") << status.message();
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> HloToStablehloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context) {
  auto module = xla::HloToMlirHloTranslateFunction(
      input, context, /*import_all_computations=*/true,
      /*flatten_computation_args_result=*/true);
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (failed(pm.run(*module))) {
    module->emitError("Failed to legalize to StableHLO");
    return nullptr;
  }

  return module;
}

mlir::OwningOpRef<mlir::ModuleOp> HloTextToStablehloTranslateFunction(
    llvm::StringRef input, mlir::MLIRContext* context) {
  std::string content(input.data(), input.size());

  auto hlo_module_error = ParseAndReturnUnverifiedModule(content);
  if (!hlo_module_error.ok()) {
    LOG(ERROR) << "HLO Module loading failed: " << hlo_module_error.status();
    return nullptr;
  }

  auto stablehlo_module =
      ConvertHloToStablehlo(*context, hlo_module_error.value().get());
  if (!stablehlo_module.ok()) {
    LOG(ERROR) << "HLO Module import failed: " << stablehlo_module.status();
    return nullptr;
  }

  return std::move(stablehlo_module).value();
}

}  // namespace xla
