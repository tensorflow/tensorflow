/* Copyright 2024 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_PASS_BOILERPLATE_H_
#define JAXLIB_MOSAIC_PASS_BOILERPLATE_H_

#include <memory>

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"

namespace jaxlib {
namespace mlir {

template <typename Derived, typename Op = void>
class Pass : public ::mlir::OperationPass<Op> {
 public:
  Pass() : ::mlir::OperationPass<Op>(::mlir::TypeID::get<Derived>()) {}
  Pass(const Pass &other) : ::mlir::OperationPass<Op>(other) {}
  Pass &operator=(const Pass &) = delete;
  Pass(Pass &&) = delete;
  Pass &operator=(Pass &&) = delete;
  ~Pass() = default;

  static constexpr ::llvm::StringLiteral getArgumentName() {
    return ::llvm::StringLiteral(Derived::kArgumentName);
  }
  ::llvm::StringRef getArgument() const override { return getArgumentName(); }
  ::llvm::StringRef getDescription() const override { return ""; }
  static constexpr ::llvm::StringLiteral getPassName() {
    return ::llvm::StringLiteral(Derived::kPassName);
  }
  ::llvm::StringRef getName() const override { return getPassName(); }
  static bool classof(const ::mlir::Pass *pass) {
    return pass->getTypeID() == ::mlir::TypeID::get<Derived>();
  }
  std::unique_ptr<::mlir::Pass> clonePass() const override {
    return std::make_unique<Derived>(*static_cast<const Derived *>(this));
  }
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {}

 private:
  using This =
      Pass<Derived, Op>;  // Can't have a comma in the macro instantiation

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(This)
};

}  // namespace mlir
}  // namespace jaxlib

#endif  // JAXLIB_MOSAIC_PASS_BOILERPLATE_H_
