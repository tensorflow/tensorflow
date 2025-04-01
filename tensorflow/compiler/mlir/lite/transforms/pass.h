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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_H_

#include <memory>
#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassOptions.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/pass_options.h"
#include "tensorflow/compiler/mlir/lite/transforms/pass_options_setter.h"

// Forward declaration for the visitor interface
// class PassOptionsVisitor;

namespace mlir {
namespace TFL {

// Interface for setting options for TFLite Converter Pass/Pipeline Options.
class MutableOptionsPass {
 public:
  virtual ~MutableOptionsPass() = default;
  virtual void ApplyOptionsVisitor(const PassOptionsSetter &visitor) = 0;
};

// CRTP Class to ensure that the derived passes implement a Options struct
template <typename DerivedPass, typename DerivedPassOptions = EmptyPassOptions,
          typename OpType = mlir::ModuleOp>
class Pass : public PassWrapper<Pass<DerivedPass, DerivedPassOptions, OpType>,
                                mlir::OperationPass<OpType>>,
             public MutableOptionsPass {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(Pass);

  Pass() = default;
  Pass(const Pass &pass) {
    static_cast<DerivedPass *>(this)->GetOptions().copyOptionValuesFrom(
        pass.GetOptions());
  }
  explicit Pass(const DerivedPassOptions &options) {
    static_cast<DerivedPass *>(this)->GetOptions().copyOptionValuesFrom(
        options);
  }

  explicit Pass(const mlir::detail::PassOptions &options) {
    static_cast<DerivedPass *>(this)->GetOptions().copyOptionValuesFrom(
        options);
  }

  /// Functions to satisfy the mlir::Pass interface
  llvm::StringRef getArgument() const override {
    return DerivedPass::GetArgument();
  }

  llvm::StringRef getDescription() const override {
    return DerivedPass::GetDescription();
  }

  llvm::StringRef getName() const override { return DerivedPass::GetName(); }

  /// Support isa/dyn_cast functionality for the derived pass class.
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<DerivedPass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    auto pass =
        std::make_unique<DerivedPass>(*static_cast<const DerivedPass *>(this));
    pass->GetOptions().copyOptionValuesFrom(GetOptions());
    return std::move(pass);
  }
  void runOnOperation() override {}

  // ApplyOptionsVisitor method to `accept` the visitor
  void ApplyOptionsVisitor(const PassOptionsSetter &visitor) override {
    visitor.SetOptions(GetOptions());
  }

 protected:
  DerivedPassOptions &GetOptions() {
    return static_cast<DerivedPass *>(this)->options_;
  }

  const DerivedPassOptions &GetOptions() const {
    return static_cast<const DerivedPass *>(this)->options_;
  }

 private:
  DerivedPassOptions options_;
};
}  // namespace TFL
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_TRANSFORMS_PASS_H_
