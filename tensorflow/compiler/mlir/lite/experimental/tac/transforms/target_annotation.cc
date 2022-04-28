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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/targets.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/common/utils.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/device_transform.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/transforms/tac_pass.h"
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace tac {
namespace {

class TargetAnnotationPass : public TacFunctionPass<TargetAnnotationPass> {
 public:
  llvm::StringRef getArgument() const final { return "tfl-target-annotation"; }
  llvm::StringRef getDescription() const final {
    return "Add user specified target annotations to the TFL operations given "
           "operation capabilities, will default to CPU.";
  }
  // using TacFunctionPass::TacFunctionPass;
  TargetAnnotationPass() : TacFunctionPass(nullptr) {}
  TargetAnnotationPass(const TargetAnnotationPass& copy)
      : TacFunctionPass(copy.module_) {}
  explicit TargetAnnotationPass(llvm::ArrayRef<std::string> device_specs)
      : TacFunctionPass(nullptr) {
    device_specs_flag_ = device_specs;
  }

  explicit TargetAnnotationPass(const TacModule* module)
      : TacFunctionPass(module) {}

 private:
  void runOnFunction() override;
  void SetTargetAnnotation(Operation* op,
                           llvm::ArrayRef<std::string> device_specs,
                           OpBuilder* builder);

  ListOption<std::string> device_specs_flag_{
      *this, "device-specs",
      llvm::cl::desc(
          "comma separated list of device specs, like CPU, GPU, Hexagon."),
      llvm::cl::ZeroOrMore};
};

void SetAnnotation(Operation* op, std::string attribute, std::string annotation,
                   OpBuilder* builder) {
  // TODO(karimnosseir): Maybe set device capabilities to allow us to have
  // more flexbility when raise the subgraphs.
  auto default_target = builder->getStringAttr(annotation);
  op->setAttr(attribute, default_target);
}

void TargetAnnotationPass::SetTargetAnnotation(
    Operation* op, llvm::ArrayRef<std::string> device_specs,
    OpBuilder* builder) {
  const InferenceType inference_type = GetInferenceType(op);
  const std::string inference_type_str = GetInferenceString(inference_type);
  SetAnnotation(op, kInferenceType, inference_type_str, builder);
  bool device_is_set = false;
  // TODO(b/177376459): Remove the usage of device_specs.
  // TODO(b/177376459): Update if needed to make testing easy.
  if (!module_) {
    for (const auto& device : device_specs) {
      auto* hardware = this->GetTargetHardware(device);
      if (hardware == nullptr) continue;
      if (hardware->IsOpSupported(op)) {
        SetAnnotation(op, kDevice, device, builder);
        device_is_set = true;
        break;
      }
    }
  } else {
    for (const auto* hardware : module_->GetAvailableHardwares()) {
      if (hardware == nullptr) continue;
      if (hardware->IsOpSupported(op)) {
        SetAnnotation(op, kDevice, GetHardwareName(hardware), builder);
        device_is_set = true;
        break;
      }
    }
  }
  // default to CPU
  if (!device_is_set) {
    if (IsNonConstOp(op) && !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, func::FuncOp, CallableOpInterface>(op)) {
      SetAnnotation(op, kDevice, "CPU", builder);
      device_is_set = true;
    }
  }
  if (!device_is_set) {
    op->emitError("cannot set target device for this ops");
  }
}

void TargetAnnotationPass::runOnFunction() {
  auto func = getFunction();
  OpBuilder builder(func);

  func.walk([&](Operation* op) {
    // We only care about TFL dialect.
    if (IsNonConstOp(op) && NotTFLQuantDequantizeOp(op) &&
        !IsTerminatorOp(op) &&
        !llvm::isa<func::ReturnOp, func::FuncOp, CallOpInterface>(op)) {
      SetTargetAnnotation(op, device_specs_flag_, &builder);
    }
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTargetAnnotationPass(
    llvm::ArrayRef<std::string> device_specs) {
  return std::make_unique<TargetAnnotationPass>(device_specs);
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateTargetAnnotationPass(
    const TacModule* module) {
  return std::make_unique<TargetAnnotationPass>(module);
}

static PassRegistration<TargetAnnotationPass> pass;

}  // namespace tac
}  // namespace TFL
}  // namespace mlir
