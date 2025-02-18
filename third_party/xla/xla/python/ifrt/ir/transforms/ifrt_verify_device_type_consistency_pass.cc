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

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/transforms/passes.h"

namespace xla {
namespace ifrt {

namespace {

#define GEN_PASS_DEF_IFRTVERIFYDEVICETYPECONSISTENCYPASS
#include "xla/python/ifrt/ir/transforms/passes.h.inc"

// Represents module type inferred from function.
enum class ModuleType { kUnknown, kXLA };

// Infers module type from the dialects in function. If there is conflict in the
// function or if the module type cannot be inferred, return `mlir::failure`.
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
  mlir::FailureOr<ModuleType> GetModuleType() const { return module_type_; }

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ModuleTypeAnalysis);

 private:
  mlir::FailureOr<ModuleType> module_type_;
};

ModuleTypeAnalysis::ModuleTypeAnalysis(mlir::Operation* op) {
  auto func = llvm::cast<mlir::func::FuncOp>(op);
  bool is_xla_module = false;

  mlir::WalkResult result =
      func.walk([&](mlir::Operation* op) -> mlir::WalkResult {
        if (auto* dialect = op->getDialect();
            dialect->getTypeID() ==
                mlir::TypeID::get<mlir::mhlo::MhloDialect>() ||
            dialect->getTypeID() ==
                mlir::TypeID::get<mlir::stablehlo::StablehloDialect>()) {
          is_xla_module = true;
        }
        return mlir::WalkResult::advance();
      });

  if (result.wasInterrupted()) {
    module_type_ = mlir::failure();
  } else if (is_xla_module) {
    module_type_ = ModuleType::kXLA;
  } else {
    module_type_ = ModuleType::kUnknown;
  }
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
  bool IsConsistentWithModuleType(ModuleType module_type,
                                  absl::string_view platform_name) const;
};

mlir::LogicalResult IfrtVerifyDeviceTypeConsistencyPass::initialize(
    mlir::MLIRContext* context) {
  for (const auto& platform_name : platform_names) {
    if (platform_name != "host" && platform_name != xla::TpuName() &&
        platform_name != xla::CudaName() && platform_name != xla::CpuName()) {
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
      return platform_name == xla::TpuName() ||
             platform_name == xla::CudaName();
    default:
      LOG(ERROR) << "Unexpected value for InferredDeviceType.";
      return false;
  }
}

void IfrtVerifyDeviceTypeConsistencyPass::runOnOperation() {
  mlir::ModuleOp module_op = getOperation();
  mlir::SymbolTableCollection symbol_table;
  mlir::WalkResult result = module_op.walk([&](CallOp call_op)
                                               -> mlir::WalkResult {
    llvm::ArrayRef<int> devices = call_op.getDevices();
    DCHECK(!devices.empty()) << "has empty device list";

    mlir::FailureOr<ModuleType> callee_module_type =
        getChildAnalysis<ModuleTypeAnalysis>(call_op.getCalleeOp(symbol_table))
            .GetModuleType();
    if (mlir::failed(callee_module_type)) {
      return mlir::WalkResult::interrupt();
    }

    // Use the first device ID to find platform name.
    int first_device_id = devices.front();
    if (first_device_id >= platform_names.size()) {
      return call_op->emitOpError()
             << "cannot find mapping for logical device id " << first_device_id
             << ". Mapping size: " << platform_names.size();
    }

    if (!IsConsistentWithModuleType(*callee_module_type,
                                    platform_names[first_device_id])) {
      return call_op->emitOpError()
             << "has platform: " << platform_names[first_device_id]
             << ", which is incompatible with the module type inferred from "
                "callee.";
    }

    for (int device_id : devices) {
      if (device_id >= platform_names.size()) {
        return call_op->emitOpError()
               << "cannot find mapping for logical device id " << device_id
               << ". Mapping size: " << platform_names.size();
      }
      if (platform_names[device_id] != platform_names[first_device_id]) {
        return call_op->emitOpError()
               << "requires a single platform type. Expected platform: "
               << platform_names[first_device_id]
               << ". Actual platform of logical device " << device_id << ": "
               << platform_names[device_id];
      }
    }
    return mlir::WalkResult::advance();
  });

  if (result.wasInterrupted()) {
    signalPassFailure();
    return;
  }
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyDeviceTypeConsistencyPass(
    IfrtVerifyDeviceTypeConsistencyPassOptions options) {
  return std::make_unique<IfrtVerifyDeviceTypeConsistencyPass>(options);
}

}  // namespace ifrt
}  // namespace xla
