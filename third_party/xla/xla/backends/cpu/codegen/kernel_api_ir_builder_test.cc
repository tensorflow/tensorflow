/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/codegen/kernel_api_ir_builder.h"

#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class KernelApiIrBuilderTest : public HloTestBase {
 public:
  KernelApiIrBuilderTest()
      : module_("KernelApiIrBuilderTest", context_),
        kernel_api_ir_builder_(context_,
                               KernelApiIrBuilder::Options{true, 256}) {}

  llvm::IRBuilder<> getBuilder() { return llvm::IRBuilder<>(context_); }

  auto EmitKernelPrototype(const HloInstruction* instr,
                           const BufferAssignment* buffer_assignment) {
    return kernel_api_ir_builder_.EmitKernelPrototype(module_, instr,
                                                      buffer_assignment);
  }

  auto EmitKernelPrototype(
      absl::string_view name,
      absl::Span<const KernelApiIrBuilder::KernelParameter> arguments,
      absl::Span<const KernelApiIrBuilder::KernelParameter> results) {
    return kernel_api_ir_builder_.EmitKernelPrototype(module_, name, arguments,
                                                      results);
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        backend().compiler()->BufferSizeBytesFunction(),
        [](LogicalBuffer::Color) { return /*alignment=*/1; });
  }

  llvm::LLVMContext& context() { return context_; }
  std::string DumpToString() { return llvm_ir::DumpToString(&module_); }

 private:
  llvm::LLVMContext context_;
  llvm::Module module_;
  KernelApiIrBuilder kernel_api_ir_builder_;
};

namespace {

TEST_F(KernelApiIrBuilderTest, BuildKernelPrototype) {
  auto hlo = std::make_unique<HloModule>("test", HloModuleConfig());

  auto shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 2});

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice arg0(&alloc, /*offset=*/0, /*size=*/256);
  BufferAllocation::Slice arg1(&alloc, /*offset=*/256, /*size=*/256);
  BufferAllocation::Slice res0(&alloc, /*offset=*/512, /*size=*/256);
  BufferAllocation::Slice res1(&alloc, /*offset=*/768, /*size=*/256);

  std::vector<KernelApiIrBuilder::KernelParameter> arguments = {{shape, arg0},
                                                                {shape, arg1}};
  std::vector<KernelApiIrBuilder::KernelParameter> results = {{shape, res0},
                                                              {shape, res1}};

  TF_ASSERT_OK_AND_ASSIGN(auto prototype,
                          EmitKernelPrototype("test", arguments, results));
  llvm::IRBuilder<> builder = getBuilder();
  builder.SetInsertPoint(prototype.function->getEntryBlock().getTerminator());

  auto* zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context()), 0);
  llvm_ir::IrArray::Index index(zero, shape, &builder);

  // Emit loads from arguments and results buffers to test alias scope metadata.
  EXPECT_NE(prototype.arguments[0].EmitReadArrayElement(index, &builder),
            nullptr);
  EXPECT_NE(prototype.arguments[1].EmitReadArrayElement(index, &builder),
            nullptr);
  EXPECT_NE(prototype.results[0].EmitReadArrayElement(index, &builder),
            nullptr);
  EXPECT_NE(prototype.results[1].EmitReadArrayElement(index, &builder),
            nullptr);

  // clang-format off
  ASSERT_TRUE(*RunFileCheck(DumpToString(),
                            absl::StrCat(R"(
    CHECK: define ptr @test(ptr %0) #0 {

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThreadDim, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThread, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThread, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_KernelThread, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %XLA_CPU_KernelArg, {{.*}} i32 0, i32 0
    CHECK:      %[[ARG0:.+]] = load ptr, {{.*}}, !invariant.load ![[SCOPE0:.+]], !dereferenceable ![[DEREF_BYTES:.+]], !align ![[ALIGNMENT:.+]]

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %XLA_CPU_KernelArg, {{.*}} i32 1, i32 0
    CHECK:      %[[ARG1:.+]] = load ptr, {{.*}}, !invariant.load ![[SCOPE0]], !dereferenceable ![[DEREF_BYTES]], !align ![[ALIGNMENT]]

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %XLA_CPU_KernelArg, {{.*}} i32 2, i32 0
    CHECK:      %[[ARG2:.+]] = load ptr, {{.*}}, !invariant.load ![[SCOPE0]], !dereferenceable ![[DEREF_BYTES]], !align ![[ALIGNMENT]]

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %XLA_CPU_KernelArg, {{.*}} i32 3, i32 0
    CHECK:      %[[ARG3:.+]] = load ptr, {{.*}}, !invariant.load ![[SCOPE0]], !dereferenceable ![[DEREF_BYTES]], !align ![[ALIGNMENT]]

    CHECK-NEXT: %[[PTR0:.+]] = getelementptr inbounds float, ptr %[[ARG0]]
    CHECK:      load float, ptr %[[PTR0]], align 4,
    CHECK-SAME:                            !invariant.load ![[SCOPE0]],
    CHECK-SAME:                            !noalias ![[SCOPE1:.+]]

    CHECK-NEXT: %[[PTR1:.+]] = getelementptr inbounds float, ptr %[[ARG1]]
    CHECK:      load float, ptr %[[PTR1]], align 4,
    CHECK-SAME:                            !invariant.load ![[SCOPE0]],
    CHECK-SAME:                            !noalias ![[SCOPE1]]

    CHECK-NEXT: %[[PTR2:.+]] = getelementptr inbounds float, ptr %[[ARG2]]
    CHECK:      load float, ptr %[[PTR2]], align 4, !alias.scope ![[SCOPE2:.+]],
    CHECK:                                          !noalias ![[SCOPE3:.+]]

    CHECK-NEXT: %[[PTR3:.+]] = getelementptr inbounds float, ptr %[[ARG3]]
    CHECK:      load float, ptr %[[PTR3]], align 4, !alias.scope ![[SCOPE3]],
    CHECK:                                          !noalias ![[SCOPE2]]

    CHECK:      ret ptr null
    CHECK: }

    #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
    CHECK-DAG: ![[ALIGNMENT]] = !{i64 )", cpu_function_runtime::MinAlign(), R"(}
    CHECK-DAG: ![[SCOPE0]] = !{}
    CHECK-DAG: ![[SCOPE1]] = !{![[RES0:.+]], ![[RES1:.+]]}
    CHECK-DAG: ![[SCOPE2]] = !{![[RES0]]}
    CHECK-DAG: ![[SCOPE3]] = !{![[RES1]]}
    CHECK-DAG: ![[RES0]] = !{!"{{.*}}, offset:512, {{.*}}", ![[DOMAIN:.+]]}
    CHECK-DAG: ![[RES1]] = !{!"{{.*}}, offset:768, {{.*}}", ![[DOMAIN]]}
    CHECK-DAG: ![[DOMAIN]] = !{!"XLA host kernel test AA domain"}
  )")));
  // clang-format on

  // Match for dereferenceable metadata in separate check, because depending on
  // the alignment value, it may be the same scope as align, and may be a
  // separate one. It's impossible to match both these cases in one FileCheck.
  ASSERT_TRUE(*RunFileCheck(DumpToString(), R"(
    CHECK:      {{.+}} = load ptr, {{.*}}, !dereferenceable ![[DEREF_BYTES:.+]],
    CHECK: ![[DEREF_BYTES]] = !{i64 32}
  )"));
}

TEST_F(KernelApiIrBuilderTest, AllInvariantBuffers) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT add.0 = f32[2,2] add(p0, p1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignement.get()));

  ASSERT_EQ(prototype.invariant_arguments.size(), 2);
}

TEST_F(KernelApiIrBuilderTest, InvariantBufferPassedTwice) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT add.0 = f32[2,2] add(p0, p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignement.get()));

  // Invariant buffers contains indices of both arguments, even though it is the
  // same buffer slice.
  ASSERT_EQ(prototype.invariant_arguments.size(), 2);
}

TEST_F(KernelApiIrBuilderTest, NoInvariantBuffers) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m, input_output_alias={ {}: (0, {}, must-alias) }
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT add.0 = f32[2,2] add(p0, p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignement.get()));

  ASSERT_EQ(prototype.invariant_arguments.size(), 0);
}

TEST_F(KernelApiIrBuilderTest, MixedBuffers) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m, input_output_alias={ {}: (1, {}, must-alias) }
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT add.0 = f32[2,2] add(p0, p1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignement.get()));

  // The first argument is invariant, the second is not because it's aliased to
  // the output.
  EXPECT_EQ(prototype.invariant_arguments.size(), 1);
  EXPECT_TRUE(prototype.invariant_arguments.contains(0));
}

}  // namespace
}  // namespace xla::cpu
