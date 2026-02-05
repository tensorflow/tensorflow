/* Copyright 2025 The OpenXLA Authors.

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
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"
#include "xla/codegen/intrinsic/cpp/tanh_ll.h"

namespace xla {
namespace codegen {
using ::testing::ContainsRegex;

namespace {

TEST(TanhTest, FloatTanhVectorized) {
  llvm::LLVMContext context;
  std::unique_ptr<llvm::Module> module =
      ParseEmbeddedBitcode(context, llvm_ir::kTanhLlIr);

  std::string ir;
  llvm::raw_string_ostream stream(ir);
  module->print(stream, nullptr);

  EXPECT_THAT(ir, ContainsRegex("fmul <4 x float>"));
  EXPECT_THAT(
      ir, ContainsRegex("fcmp olt <4 x float>.*float 0x3F3A36E2E0000000.*"));
}
}  // namespace
}  // namespace codegen
}  // namespace xla
