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

#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/python/ifrt/ir/compilation/passes.h"  // IWYU pragma: keep
#include "xla/python/ifrt/ir/compilation/utils.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/utils.h"

namespace xla {
namespace ifrt {

#define GEN_PASS_DEF_IFRTVERIFYDEVICETYPECONSISTENCYPASS
#include "xla/python/ifrt/ir/compilation/passes.h.inc"

namespace {

// Represents module type inferred from function.
enum class ModuleType { kUnknown, kXLA };

// Infers module type from the dialects in function.
//
// The assumption behind the analysis is that mhlo/stablehlo dialects will be
// run on TPU or GPU devices.
class ModuleTypeAnalysis {
 public:
  // `op` should be a FuncOp.
  explicit ModuleTypeAnalysis(mlir::Operation* op);

  ModuleTypeAnalysis(const ModuleTypeAnalysis& other) = delete;
  ModuleTypeAnalysis& operator=(const ModuleTypeAnalysis& other) = delete;

  // Get the analyzed device type.
  ModuleType GetModuleType() const { return module_type_; }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuleTypeAnalysis);

 private:
  ModuleType module_type_;
};

ModuleTypeAnalysis::ModuleTypeAnalysis(mlir::Operation* op) {
  auto func = llvm::cast<mlir::func::FuncOp>(op);
  bool is_xla_module = false;

  func.walk([&](mlir::Operation* op) {
    if (auto* dialect = op->getDialect();
        dialect->getTypeID() == mlir::TypeID::get<mlir::mhlo::MhloDialect>() ||
        dialect->getTypeID() ==
            mlir::TypeID::get<mlir::stablehlo::StablehloDialect>()) {
      is_xla_module = true;
    }
  });

  module_type_ = is_xla_module ? ModuleType::kXLA : ModuleType::kUnknown;
}

class IfrtVerifyDeviceTypeConsistencyPass
    : public impl::IfrtVerifyDeviceTypeConsistencyPassBase<
          IfrtVerifyDeviceTypeConsistencyPass> {
 public:
  using impl::IfrtVerifyDeviceTypeConsistencyPassBase<
      IfrtVerifyDeviceTypeConsistencyPass>::
      IfrtVerifyDeviceTypeConsistencyPassBase;

  mlir::LogicalResult initialize(mlir::MLIRContext* context) override;

  void runOnOperation() override;

 private:
  std::vector<std::string> platform_names_;

  bool IsConsistentWithModuleType(ModuleType module_type,
                                  absl::string_view platform_name) const;
};

mlir::LogicalResult IfrtVerifyDeviceTypeConsistencyPass::initialize(
    mlir::MLIRContext* context) {
  auto platform_names_or = ExpandPlatformNames(platform_names);
  if (!platform_names_or.ok()) {
    return mlir::emitError(mlir::UnknownLoc(),
                           platform_names_or.status().message());
  }
  platform_names_ = std::move(*platform_names_or);
  for (const std::string& platform_name : platform_names_) {
    if (platform_name != "host" && !IsXlaPlatform(platform_name)) {
      LOG(ERROR) << "Unsupported platform: " << platform_name;
      return mlir::failure();
    }
  }
  return mlir::success();
}

bool IfrtVerifyDeviceTypeConsistencyPass::IsConsistentWithModuleType(
    ModuleType module_type, absl::string_view platform_name) const {
  switch (module_type) {
    case ModuleType::kUnknown:
      return true;
    case ModuleType::kXLA:
      return IsXlaPlatform(platform_name);
  }
  LOG(FATAL) << "Unexpected ModuleType: " << static_cast<int>(module_type);
}

void IfrtVerifyDeviceTypeConsistencyPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  mlir::SymbolTableCollection symbol_table;
  for (mlir::func::FuncOp func_op : GetIfrtFunctions(module_op)) {
    for (CallOp call_op : func_op.getOps<CallOp>()) {
      llvm::ArrayRef<int> devices = call_op.getDevices();

      ModuleType callee_module_type = getChildAnalysis<ModuleTypeAnalysis>(
                                          call_op.getCalleeOp(symbol_table))
                                          .GetModuleType();

      // Use the first device ID to find platform name.
      mlir::FailureOr<llvm::StringRef> first_platform =
          GetPlatformNameForDeviceId(call_op, devices.front(), platform_names_);
      if (mlir::failed(first_platform)) {
        signalPassFailure();
        return;
      }

      if (!IsConsistentWithModuleType(callee_module_type, *first_platform)) {
        call_op->emitOpError()
            << "has platform: " << *first_platform
            << ", which is incompatible with the module type inferred from "
               "callee.";
        signalPassFailure();
        return;
      }

      for (int device_id : devices) {
        mlir::FailureOr<llvm::StringRef> device_platform =
            GetPlatformNameForDeviceId(call_op, device_id, platform_names_);
        if (mlir::failed(device_platform)) {
          signalPassFailure();
          return;
        }
        if (*device_platform != *first_platform) {
          call_op->emitOpError()
              << "requires a single platform type. Expected platform: "
              << *first_platform << ". Actual platform of logical device "
              << device_id << ": " << *device_platform;
          signalPassFailure();
          return;
        }
      }
    }
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
