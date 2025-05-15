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

#if defined(INTEL_MKL)

#include "xla/service/cpu/onednn_memory_util.h"

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace xla {
namespace cpu {
namespace {

class MemoryUtilTest
    : public ::testing::Test,
      public ::testing::WithParamInterface<std::vector<int64_t>> {
 protected:
  constexpr static const char* test_pattern_ = R"(
    CHECK: %[[mref0:[0-9]+]] = insertvalue
    CHECK: %[[mref1:[0-9]+]] = insertvalue
    CHECK-SAME: [[arr:\[12 x i64\]]] } %[[mref0]], i64 255, 3
    CHECK: %{{[0-9]+}} = insertvalue
    CHECK-SAME: %[[mref1]], [[arr]] )";

  auto GetMemRefTestPattern(Shape shape) {
    std::ostringstream stream;
    stream << "[";
    absl::c_for_each(shape.dimensions(),
                     [&stream](auto x) { stream << "i64 " << x << ", "; });
    return absl::StrCat(test_pattern_, stream.str());
  }
};

TEST_P(MemoryUtilTest, VerifyMemRefTest) {
  std::string filecheck_input;
  llvm::LLVMContext context = llvm::LLVMContext();
  llvm::IRBuilder builder(context);
  llvm::raw_string_ostream ostream(filecheck_input);
  llvm::Module module("MemoryUtilTest", context);

  llvm::FunctionType* function_type = llvm::FunctionType::get(
      llvm::Type::getVoidTy(context), {builder.getPtrTy()}, false);
  llvm::Function* function = llvm::Function::Create(
      function_type, llvm::Function::LinkageTypes::ExternalLinkage,
      "memory_util_test", module);
  llvm::BasicBlock* bb = llvm::BasicBlock::Create(context, "BB", function);
  builder.SetInsertPoint(bb);

  Shape shape = ShapeUtil::MakeShape(F32, GetParam());
  llvm::Argument* ptr = function->getArg(0);
  llvm::Type* type = llvm_ir::PrimitiveTypeToIrType(F32, builder.getContext());

  if (shape.IsArray()) {
    for (auto dim : LayoutUtil::MinorToMajor(shape)) {
      type = llvm::ArrayType::get(type, shape.dimensions(dim));
    }
  }

  llvm_ir::IrArray ir_array(ptr, type, shape);
  auto alloca = GetAllocaAndEmitMemrefInfo(builder, ir_array);
  alloca.EmitLifetimeEnd();
  ostream << module;

  absl::StatusOr<bool> match =
      RunFileCheck(filecheck_input, GetMemRefTestPattern(shape));
  TF_ASSERT_OK(match.status());
  EXPECT_TRUE(match.value());
}

INSTANTIATE_TEST_SUITE_P(
    MemoryUtilTestSuite, MemoryUtilTest,
    ::testing::Values(std::vector<int64_t>({30}),
                      std::vector<int64_t>({30, 40}),
                      std::vector<int64_t>({30, 40, 50})),
    [](const ::testing::TestParamInfo<MemoryUtilTest::ParamType>& info) {
      return absl::StrCat("Rank_", info.param.size());
    });

}  // namespace
}  // namespace cpu
}  // namespace xla

#endif  // INTEL_MKL
