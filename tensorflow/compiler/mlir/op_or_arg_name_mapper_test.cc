/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/op_or_arg_name_mapper.h"

#include <optional>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace tensorflow {
namespace {

TEST(OpOrArgNameMapperTest, GetMappedNameView) {
  mlir::MLIRContext context;
  OpOrArgLocNameMapper mapper;

  // Create a dummy operation.
  context.allowUnregisteredDialects();
  mlir::OperationState state(mlir::UnknownLoc::get(&context), "test.op");
  mlir::Operation *op = mlir::Operation::create(state);

  // Test case 1: Name not mapped yet.
  EXPECT_EQ(mapper.GetMappedNameView(op), std::nullopt);

  // Map a name.
  mapper.InitOpName(op, "test_op");

  // Test case 2: Name is mapped.
  std::optional<absl::string_view> name = mapper.GetMappedNameView(op);
  EXPECT_TRUE(name.has_value());
  EXPECT_EQ(*name, "test_op");

  // Clean up the operation.
  op->destroy();
}

}  // namespace
}  // namespace tensorflow
