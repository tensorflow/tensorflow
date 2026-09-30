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

#include "xla/service/llvm_ir/llvm_util.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "llvm/ADT/StringRef.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"

namespace xla::llvm_ir {
namespace {

llvm::Instruction* FindInstruction(llvm::Function& fn, llvm::StringRef name) {
  for (llvm::Instruction& inst : llvm::instructions(fn)) {
    if (inst.getName() == name) {
      return &inst;
    }
  }
  return nullptr;
}

std::vector<const llvm::Instruction*> InstructionOrder(
    const llvm::Function& fn) {
  std::vector<const llvm::Instruction*> order;
  for (const llvm::Instruction& inst : llvm::instructions(fn)) {
    order.push_back(&inst);
  }
  return order;
}

struct SinkFMulTestCase {
  std::string test_name;
  std::string ir;
  bool should_sink;
};

class SinkContractableFMulToFAddFSubTest
    : public ::testing::TestWithParam<SinkFMulTestCase> {};

TEST_P(SinkContractableFMulToFAddFSubTest,
       SinksOnlyContractableSingleUsePairs) {
  const SinkFMulTestCase& tc = GetParam();
  llvm::LLVMContext context;
  llvm::SMDiagnostic diagnostic;
  std::unique_ptr<llvm::Module> module =
      llvm::parseAssemblyString(tc.ir, diagnostic, context);
  ASSERT_NE(module, nullptr) << diagnostic.getMessage().str();

  llvm::Function* fn = module->getFunction("test_fn");
  ASSERT_NE(fn, nullptr);
  const std::vector<const llvm::Instruction*> order_before =
      InstructionOrder(*fn);

  SinkContractableFMulToFAddFSub(*module);

  llvm::Instruction* mul = FindInstruction(*fn, "mul");
  llvm::Instruction* user = FindInstruction(*fn, "user");
  ASSERT_NE(mul, nullptr);
  ASSERT_NE(user, nullptr);
  if (tc.should_sink) {
    EXPECT_EQ(mul->getNextNode(), user) << DumpToString(fn);
    if (llvm::Instruction* mul0 = FindInstruction(*fn, "mul0")) {
      EXPECT_EQ(mul0->getNextNode(), mul) << DumpToString(fn);
    }
  } else {
    // Nothing at all should have moved.
    EXPECT_EQ(InstructionOrder(*fn), order_before) << DumpToString(fn);
  }
}

INSTANTIATE_TEST_SUITE_P(
    SinkContractableFMulToFAddFSubTests, SinkContractableFMulToFAddFSubTest,
    ::testing::Values(SinkFMulTestCase{
                          /*test_name=*/"FAddOperand0SingleUse",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FAddOperand1SingleUse",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd contract float %c, %mul
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FSubOperand0SingleUse",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fsub contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FSubOperand1SingleUse",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fsub contract float %c, %mul
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FAddBothOperandsSingleUse",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, float %d, ptr %p) {
              entry:
                %mul0 = fmul contract float %a, %b
                %mul = fmul contract float %c, %d
                %load = load float, ptr %p, align 4
                %user = fadd contract float %mul0, %mul
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"VectorSingleUse",
                          /*ir=*/R"(
              define <4 x float> @test_fn(<4 x float> %a, <4 x float> %b, <4 x float> %c, ptr %p) {
              entry:
                %mul = fmul contract <4 x float> %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd contract <4 x float> %mul, %c
                ret <4 x float> %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"AlreadyAdjacentUnchanged",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %load = load float, ptr %p, align 4
                %mul = fmul contract float %a, %b
                %user = fadd contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/true,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"NoContractOnFMulNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul float %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/false,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"NoContractOnUserNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/false,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"MultiUseNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fadd contract float %mul, %c
                %extra = fadd contract float %mul, %load
                ret float %extra
              }
            )",
                          /*should_sink=*/false,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FDivUserNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fdiv contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/false,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"FMulUserNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, ptr %p) {
              entry:
                %mul = fmul contract float %a, %b
                %load = load float, ptr %p, align 4
                %user = fmul contract float %mul, %c
                ret float %user
              }
            )",
                          /*should_sink=*/false,
                      },
                      SinkFMulTestCase{
                          /*test_name=*/"DifferentBasicBlocksNotSunk",
                          /*ir=*/R"(
              define float @test_fn(float %a, float %b, float %c, i1 %cond) {
              entry:
                %mul = fmul contract float %a, %b
                br i1 %cond, label %then, label %else
              then:
                %user = fadd contract float %mul, %c
                ret float %user
              else:
                ret float %c
              }
            )",
                          /*should_sink=*/false,
                      }),
    [](const ::testing::TestParamInfo<SinkFMulTestCase>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace xla::llvm_ir
