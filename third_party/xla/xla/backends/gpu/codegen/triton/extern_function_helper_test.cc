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

#include "xla/backends/gpu/codegen/triton/extern_function_helper.h"

#include <string>
#include <variant>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "xla/backends/gpu/codegen/triton/ir/triton_xla_ops.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::xla {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::xla::llvm_ir::CreateMlirModuleOp;

class ExternFunctionNameTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<LLVM::LLVMDialect>();
    context_.loadDialect<triton::TritonDialect>();
  }

  mlir::MLIRContext context_;
};

struct AtomicInstrNameTest : public ExternFunctionNameTest,
                             ::testing::WithParamInterface<bool> {
  bool HasMask() const { return GetParam(); }

  std::string GetName(absl::string_view prefix) {
    return absl::StrCat(prefix, HasMask() ? "_mask" : "_nomask");
  }
};

INSTANTIATE_TEST_SUITE_P(AtomicInstrNameTest, AtomicInstrNameTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& param) {
                           return param.param ? "mask" : "nomask";
                         });

// Test parsing GetThreadId instruction
TEST_F(ExternFunctionNameTest, ParseGetThreadId) {
  auto result = ParseExternFunctionName("xla_getthreadid");
  ASSERT_THAT(result, IsOk());
  EXPECT_TRUE(std::holds_alternative<GetThreadIdInstruction>(*result));
}

// Test parsing AtomicWrite instructions
TEST_P(AtomicInstrNameTest, ParseAtomicWriteRelaxedGpu) {
  auto result = ParseExternFunctionName(GetName("xla_atomicwrite_relaxed_gpu"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELAXED);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::GPU);
  EXPECT_EQ(instruction.has_mask, HasMask());
}

TEST_P(AtomicInstrNameTest, ParseAtomicWriteReleaseSystem) {
  auto result =
      ParseExternFunctionName(GetName("xla_atomicwrite_release_system"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::SYSTEM);
  EXPECT_EQ(instruction.has_mask, HasMask());
}

TEST_P(AtomicInstrNameTest, ParseAtomicWriteAcqRelCta) {
  auto result = ParseExternFunctionName(GetName("xla_atomicwrite_acqrel_cta"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicWriteInstruction>(*result));
  auto& instruction = std::get<AtomicWriteInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE_RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::CTA);
  EXPECT_EQ(instruction.has_mask, HasMask());
}

// Test parsing AtomicSpinWait instructions
TEST_P(AtomicInstrNameTest, ParseAtomicSpinWaitRelaxedGpuEq) {
  auto result =
      ParseExternFunctionName(GetName("xla_atomicspinwait_relaxed_gpu_eq"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::RELAXED);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::GPU);
  EXPECT_EQ(instruction.comparator, Comparator::EQ);
  EXPECT_EQ(instruction.has_mask, HasMask());
}

TEST_P(AtomicInstrNameTest, ParseAtomicSpinWaitAcquireSystemLt) {
  auto result =
      ParseExternFunctionName(GetName("xla_atomicspinwait_acquire_system_lt"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::SYSTEM);
  EXPECT_EQ(instruction.comparator, Comparator::LT);
  EXPECT_EQ(instruction.has_mask, HasMask());
}

TEST_P(AtomicInstrNameTest, ParseAtomicSpinWaitAcqRelCtaEq) {
  auto result =
      ParseExternFunctionName(GetName("xla_atomicspinwait_acqrel_cta_eq"));
  ASSERT_THAT(result, IsOk());

  ASSERT_TRUE(std::holds_alternative<AtomicSpinWaitInstruction>(*result));
  auto& instruction = std::get<AtomicSpinWaitInstruction>(*result);
  EXPECT_EQ(instruction.semantic, triton::MemSemantic::ACQUIRE_RELEASE);
  EXPECT_EQ(instruction.scope, triton::MemSyncScope::CTA);
  EXPECT_EQ(instruction.comparator, Comparator::EQ);
}

// Test parsing invalid function names
TEST_F(ExternFunctionNameTest, ParseInvalidFunctionName) {
  auto result = ParseExternFunctionName("invalid_function_name");
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Invalid extern function name")));
}

TEST_P(AtomicInstrNameTest, ParseInvalidAtomicWrite) {
  auto result = ParseExternFunctionName(GetName("xla_atomicwrite_invalid_gpu"));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown memory semantic")));
}

TEST_P(AtomicInstrNameTest, ParseInvalidScope) {
  auto result =
      ParseExternFunctionName(GetName("xla_atomicwrite_relaxed_invalid"));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown memory sync scope")));
}

TEST_P(AtomicInstrNameTest, ParseInvalidComparator) {
  auto result = ParseExternFunctionName(
      GetName("xla_atomicspinwait_relaxed_gpu_invalid"));
  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("Unknown comparator")));
}

// Test serialization
TEST_F(ExternFunctionNameTest, SerializeGetThreadId) {
  GetThreadIdInstruction instruction;
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, "xla_getthreadid");
}

TEST_P(AtomicInstrNameTest, SerializeAtomicWrite) {
  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELEASE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .has_mask= */ HasMask()};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, GetName("xla_atomicwrite_release_gpu"));
}

TEST_P(AtomicInstrNameTest, SerializeAtomicWriteAcqRel) {
  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::ACQUIRE_RELEASE,
      /* .scope= */ triton::MemSyncScope::SYSTEM,
      /* .has_mask= */ HasMask()};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, GetName("xla_atomicwrite_acqrel_system"));
}

TEST_P(AtomicInstrNameTest, SerializeAtomicSpinWait) {
  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::ACQUIRE,
      /* .scope= */ triton::MemSyncScope::CTA,
      /* .comparator= */ Comparator::LT,
      /* .has_mask= */ HasMask()};
  std::string result = SerializeExternFunctionName(instruction);
  EXPECT_EQ(result, GetName("xla_atomicspinwait_acquire_cta_lt"));
}

// Test round-trip (parse then serialize)
TEST_F(ExternFunctionNameTest, RoundTripGetThreadId) {
  std::string original = "xla_getthreadid";
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

TEST_P(AtomicInstrNameTest, RoundTripAtomicWrite) {
  std::string original = GetName("xla_atomicwrite_relaxed_gpu");
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

TEST_P(AtomicInstrNameTest, RoundTripAtomicSpinWait) {
  std::string original = GetName("xla_atomicspinwait_acquire_system_lt");
  auto parsed = ParseExternFunctionName(original);
  ASSERT_THAT(parsed, IsOk());
  std::string serialized = SerializeExternFunctionName(*parsed);
  EXPECT_EQ(original, serialized);
}

// Test memory semantic validation
TEST_F(ExternFunctionNameTest, ValidateGetThreadIdAlwaysValid) {
  GetThreadIdInstruction instruction;
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteRelaxed) {
  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELAXED,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteRelease) {
  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELEASE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicWriteAcquireInvalid) {
  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::ACQUIRE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("RELAXED or RELEASE")));
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitRelaxed) {
  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELAXED,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .comparator= */ Comparator::EQ,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitAcquire) {
  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::ACQUIRE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .comparator= */ Comparator::EQ,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction), IsOk());
}

TEST_F(ExternFunctionNameTest, ValidateAtomicSpinWaitReleaseInvalid) {
  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELEASE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .comparator= */ Comparator::EQ,
      /* .has_mask= */ false,
  };
  EXPECT_THAT(ValidateMemorySemantic(instruction),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("RELAXED or ACQUIRE")));
}

// Test LLVM operation creation - verify correct inline assembly
TEST_F(ExternFunctionNameTest, CreateGetThreadIdOpsCUDA) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  GetThreadIdInstruction instruction;
  LLVMOpCreationParams params{/* .builder= */ builder,
                              /* .loc= */ builder.getUnknownLoc(),
                              /* .target= */ TargetBackend::CUDA,
                              /* .operands= */ {}};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Verify inline assembly was created
  auto* defining_op = result.getDefiningOp();
  ASSERT_TRUE(defining_op != nullptr);
  auto asm_op = mlir::dyn_cast<LLVM::InlineAsmOp>(defining_op);
  ASSERT_TRUE(asm_op != nullptr);
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("mov.u32"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("%tid.x"));
}

TEST_F(ExternFunctionNameTest, CreateAtomicWriteOpsVerifyPTXAssembly) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicWriteInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELEASE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .has_mask= */ false,
  };

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{/* .builder= */ builder,
                              /* .loc= */ builder.getUnknownLoc(),
                              /* .target= */ TargetBackend::CUDA,
                              /* .operands= */ operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Result is a poison value, find the inline assembly in the block
  LLVM::InlineAsmOp asm_op;
  entry_block->walk([&](LLVM::InlineAsmOp op) {
    asm_op = op;
    return mlir::WalkResult::interrupt();
  });
  ASSERT_TRUE(asm_op != nullptr);
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("st.global"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("gpu"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("release"));
}

TEST_F(ExternFunctionNameTest, CreateAtomicSpinWaitOpsVerifyPTXAssembly) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::ACQUIRE,
      /* .scope= */ triton::MemSyncScope::GPU,
      /* .comparator= */ Comparator::EQ,
      /* .has_mask= */ false,
  };

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{/* .builder= */ builder,
                              /* .loc= */ builder.getUnknownLoc(),
                              /* .target= */ TargetBackend::CUDA,
                              /* .operands= */ operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);
  EXPECT_TRUE(result.getType().isInteger(32));

  // Result is a poison value, find the inline assembly in the block
  LLVM::InlineAsmOp asm_op;
  entry_block->walk([&](LLVM::InlineAsmOp op) {
    asm_op = op;
    return mlir::WalkResult::interrupt();
  });
  ASSERT_TRUE(asm_op != nullptr);
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("ld.global"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("gpu"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("acquire"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("setp.eq"));
}

TEST_F(ExternFunctionNameTest, CreateAtomicSpinWaitOpsVerifyComparator) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      CreateMlirModuleOp(mlir::UnknownLoc::get(&context_));
  mlir::OpBuilder builder(&context_);
  builder.setInsertionPointToEnd(module->getBody());

  auto i32_type = builder.getI32Type();
  auto ptr_type = LLVM::LLVMPointerType::get(&context_);
  auto func_type = LLVM::LLVMFunctionType::get(i32_type, {ptr_type, i32_type});
  auto func = LLVM::LLVMFuncOp::create(builder, builder.getUnknownLoc(),
                                       "test_func", func_type);
  auto* entry_block = func.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry_block);

  AtomicSpinWaitInstruction instruction{
      /* .semantic= */ triton::MemSemantic::RELAXED,
      /* .scope= */ triton::MemSyncScope::SYSTEM,
      /* .comparator= */ Comparator::LT};

  llvm::SmallVector<mlir::Value> operands = {entry_block->getArgument(0),
                                             entry_block->getArgument(1)};
  LLVMOpCreationParams params{/* .builder= */ builder,
                              /* .loc= */ builder.getUnknownLoc(),
                              /* .target= */ TargetBackend::CUDA,
                              /* .operands= */ operands};

  mlir::Value result = CreateLLVMOpsForInstruction(instruction, params);
  ASSERT_TRUE(result != nullptr);

  // Result is a poison value, find the inline assembly in the block
  LLVM::InlineAsmOp asm_op;
  entry_block->walk([&](LLVM::InlineAsmOp op) {
    asm_op = op;
    return mlir::WalkResult::interrupt();
  });
  ASSERT_TRUE(asm_op != nullptr);
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("ld.global"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("sys"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("relaxed"));
  EXPECT_THAT(asm_op.getAsmString().str(), HasSubstr("setp.lt"));
}

}  // namespace
}  // namespace mlir::triton::xla
