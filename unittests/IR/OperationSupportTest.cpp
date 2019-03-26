//===- OperationSupportTest.cpp - Operation support unit tests ------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::detail;

namespace {
Operation *createOp(MLIRContext *context, bool resizableOperands,
                    ArrayRef<Value *> operands = llvm::None,
                    ArrayRef<Type> resultTypes = llvm::None) {
  return Operation::create(
      UnknownLoc::get(context), OperationName("foo.bar", context), operands,
      resultTypes, llvm::None, llvm::None, 0, resizableOperands, context);
}

TEST(OperandStorageTest, NonResizable) {
  MLIRContext context;
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*resizableOperands=*/false, /*operands=*/llvm::None,
               builder.getIntegerType(16));
  Value *operand = useOp->getResult(0);

  // Create a non-resizable instruction with one operand.
  Operation *user = createOp(&context, /*resizableOperands=*/false, operand,
                             builder.getIntegerType(16));

  // Sanity check the storage.
  EXPECT_EQ(user->hasResizableOperandsList(), false);

  // The same number of operands is okay.
  user->setOperands(operand);
  EXPECT_EQ(user->getNumOperands(), 1);

  // Removing is okay.
  user->setOperands(llvm::None);
  EXPECT_EQ(user->getNumOperands(), 0);

  // Destroy the instructions.
  user->destroy();
  useOp->destroy();
}

TEST(OperandStorageDeathTest, AddToNonResizable) {
  MLIRContext context;
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*resizableOperands=*/false, /*operands=*/llvm::None,
               builder.getIntegerType(16));
  Value *operand = useOp->getResult(0);

  // Create a non-resizable instruction with one operand.
  Operation *user = createOp(&context, /*resizableOperands=*/false, operand,
                             builder.getIntegerType(16));

  // Sanity check the storage.
  EXPECT_EQ(user->hasResizableOperandsList(), false);

  // Adding operands to a non resizable instruction should result in a failure.
  ASSERT_DEATH(user->setOperands({operand, operand}), "");
}

TEST(OperandStorageTest, Resizable) {
  MLIRContext context;
  Builder builder(&context);

  Operation *useOp =
      createOp(&context, /*resizableOperands=*/false, /*operands=*/llvm::None,
               builder.getIntegerType(16));
  Value *operand = useOp->getResult(0);

  // Create a resizable instruction with one operand.
  Operation *user = createOp(&context, /*resizableOperands=*/true, operand,
                             builder.getIntegerType(16));

  // Sanity check the storage.
  EXPECT_EQ(user->hasResizableOperandsList(), true);

  // The same number of operands is okay.
  user->setOperands(operand);
  EXPECT_EQ(user->getNumOperands(), 1);

  // Removing is okay.
  user->setOperands(llvm::None);
  EXPECT_EQ(user->getNumOperands(), 0);

  // Adding more operands is okay.
  user->setOperands({operand, operand, operand});
  EXPECT_EQ(user->getNumOperands(), 3);

  // Destroy the instructions.
  user->destroy();
  useOp->destroy();
}

} // end namespace
