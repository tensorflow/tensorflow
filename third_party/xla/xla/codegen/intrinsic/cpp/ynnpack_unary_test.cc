/* Copyright 2026 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_16_ll.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_32_ll.h"
#include "xla/codegen/intrinsic/cpp/ynnpack_unary_64_ll.h"

namespace xla::codegen {
namespace {
using ::testing::ContainsRegex;
using ::testing::Not;

TEST(YnnpackUnaryTest, BitcodeIsTargetIndependentAndVectorized) {
  for (const std::string* bitcode :
       {&llvm_ir::kYnnpackUnary16LlIr, &llvm_ir::kYnnpackUnary32LlIr,
        &llvm_ir::kYnnpackUnary64LlIr}) {
    if (bitcode->empty()) {
      continue;
    }
    llvm::LLVMContext context;
    std::unique_ptr<llvm::Module> module =
        ParseEmbeddedBitcode(context, *bitcode);
    std::string ir;
    llvm::raw_string_ostream stream(ir);
    module->print(stream, nullptr);

    EXPECT_THAT(ir, ContainsRegex("xla\\.log\\.v16f32"));
    EXPECT_THAT(ir, ContainsRegex("xla\\.log1p\\.v8f64"));
    EXPECT_THAT(ir, Not(ContainsRegex("llvm\\.x86")));
    EXPECT_THAT(ir, Not(ContainsRegex("llvm\\.aarch64")));
  }
}

}  // namespace
}  // namespace xla::codegen
