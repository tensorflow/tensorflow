/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/ir/interfaces.h"

#include "llvm/ADT/ScopeExit.h"
#include "mlir/IR/DialectInterface.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/platform/test.h"

namespace mlir {
namespace tfg {
namespace {
TEST(TensorFlowRegistryInterface, TestDefaultImplementation) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  auto *dialect = context.getOrLoadDialect<TFGraphDialect>();

  OperationState state(UnknownLoc::get(&context), "tfg.Foo");
  state.addTypes(dialect->getControlType());

  Operation *op = Operation::create(state);
  auto cleanup = llvm::make_scope_exit([&] { op->destroy(); });
  ASSERT_TRUE(succeeded(verify(op)));
  auto iface = dyn_cast<TensorFlowRegistryInterface>(op);
  EXPECT_FALSE(iface);
}

TEST(TensorFlowRegisterInterface, TestCustomImplementation) {
  MLIRContext context(MLIRContext::Threading::DISABLED);
  DialectRegistry registry;
  registry.insert<TFGraphDialect>();

  struct CustomRegistryInterface : public TensorFlowRegistryInterfaceBase {
    using TensorFlowRegistryInterfaceBase::TensorFlowRegistryInterfaceBase;

    bool isStateful(Operation *op) const override {
      return op->getName().stripDialect() == "Foo";
    }
  };

  registry.addExtension(+[](mlir::MLIRContext *ctx, TFGraphDialect *dialect) {
    dialect->addInterfaces<CustomRegistryInterface>();
  });
  context.appendDialectRegistry(registry);

  auto *dialect = context.getOrLoadDialect<TFGraphDialect>();
  SmallVector<StringRef, 2> op_names = {"tfg.Foo", "tfg.Bar"};
  SmallVector<bool, 2> expected = {true, false};
  for (auto it : llvm::zip(op_names, expected)) {
    OperationState state(UnknownLoc::get(&context), std::get<0>(it));
    state.addTypes(dialect->getControlType());
    Operation *op = Operation::create(state);
    auto cleanup = llvm::make_scope_exit([&] { op->destroy(); });
    auto iface = dyn_cast<TensorFlowRegistryInterface>(op);
    ASSERT_TRUE(iface);
    EXPECT_EQ(iface.isStateful(), std::get<1>(it));
  }
}
}  // namespace
}  // namespace tfg
}  // namespace mlir
