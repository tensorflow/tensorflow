/* Copyright 2018 The OpenXLA Authors.

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

#include "llvm/IR/DataLayout.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla_data.pb.h"

class CpuByteSizeOfTest : public ::testing::Test {};

TEST_F(CpuByteSizeOfTest, ARM32) {
  llvm::DataLayout data_layout(
      "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64");
  auto tuple_shape =
      xla::ShapeUtil::MakeTupleShape({xla::ShapeUtil::MakeShape(xla::F32, {})});
  EXPECT_EQ(xla::llvm_ir::ByteSizeOf(tuple_shape, data_layout),
            data_layout.getPointerSize(0 /* default address space */));
}

TEST_F(CpuByteSizeOfTest, ARM64) {
  llvm::DataLayout data_layout("e-m:e-i64:64-i128:128-n32:64-S128");
  auto tuple_shape =
      xla::ShapeUtil::MakeTupleShape({xla::ShapeUtil::MakeShape(xla::F32, {})});
  EXPECT_EQ(xla::llvm_ir::ByteSizeOf(tuple_shape, data_layout),
            data_layout.getPointerSize(0 /* default address space */));
}
