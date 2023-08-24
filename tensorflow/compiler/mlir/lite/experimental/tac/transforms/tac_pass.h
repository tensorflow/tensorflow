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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_TAC_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_TAC_PASS_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/experimental/tac/hardwares/target_hardware.h"
#include "tensorflow/compiler/mlir/lite/experimental/tac/tac_module.h"

namespace mlir {
namespace TFL {
namespace tac {
// An OperationPass<> with access to the TAC module instance that the
// pass is running part of.
// See OperationPass<> comments for all details/restrictions of OperationPass.
//
// When adding new Pass to TAC, users should use this class as the base class
// as it provides access to the TAC module.
template <typename T>
class TacPass : public OperationPass<T> {
 public:
  using OperationPass<T>::OperationPass;
  explicit TacPass(const TacModule* module)
      : OperationPass<T>::OperationPass(mlir::TypeID::get<T>()),
        module_(module) {}

  ~TacPass() override = default;

  const TargetHardware* GetTargetHardware(
      const std::string& hardware_name) const {
    return module_ != nullptr
               ? module_->GetTargetHardware(hardware_name)
               : mlir::TFL::tac::GetTargetHardware(hardware_name);
  }

 protected:
  const TacModule* module_ = nullptr;  // Not owned.
};

// A FunctionPass but with access to TAC module.
// See FunctionPass comments for all details/restrictions of FunctionPass.
//
// When adding new Pass to TAC, users should use this class as the base class
// as it provides access to the TAC module.
template <typename T>
class TacFunctionPass : public TacPass<func::FuncOp> {
 public:
  using TacPass<func::FuncOp>::TacPass;

  ~TacFunctionPass() override = default;

  mlir::func::FuncOp getFunction() { return getOperation(); }

  virtual void runOnFunction() = 0;

  void runOnOperation() final {
    if (!getFunction().isExternal()) runOnFunction();
  }

 protected:
  // Returns the derived pass name.
  StringRef getName() const override { return llvm::getTypeName<T>(); }

  // A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<T>(*static_cast<const T*>(this));
  }
};

}  // namespace tac
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_TRANSFORMS_TAC_PASS_H_
