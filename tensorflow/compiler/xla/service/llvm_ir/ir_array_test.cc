/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

#include "tensorflow/compiler/xla/test.h"

namespace xla {
namespace llvm_ir {
namespace {

TEST(IrArrayTest, TestShapeIsCompatible) {
  xla::Shape a = ShapeUtil::MakeShapeWithLayout(F32, {1, 10, 20}, {2, 1, 0});
  xla::Shape b = ShapeUtil::MakeShapeWithLayout(F32, {1, 10, 20}, {2, 0, 1});
  xla::Shape c = ShapeUtil::MakeShapeWithLayout(F32, {10, 1, 20}, {2, 1, 0});

  xla::Shape d = ShapeUtil::MakeShapeWithLayout(F32, {1, 10, 30}, {2, 1, 0});
  xla::Shape e = ShapeUtil::MakeShapeWithLayout(F32, {1, 10, 30}, {2, 0, 1});
  xla::Shape f = ShapeUtil::MakeShapeWithLayout(F32, {10, 1, 30}, {2, 1, 0});

  EXPECT_TRUE(IrArray::Index::ShapeIsCompatible(a, b));
  EXPECT_TRUE(IrArray::Index::ShapeIsCompatible(a, c));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, d));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, e));
  EXPECT_FALSE(IrArray::Index::ShapeIsCompatible(a, f));
}

}  // namespace
}  // namespace llvm_ir
}  // namespace xla
