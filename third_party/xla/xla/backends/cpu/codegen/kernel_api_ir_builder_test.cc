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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/Casting.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

template <bool ValidateBuffers>
class KernelApiIrBuilderTestBase : public HloHardwareIndependentTestBase {
 public:
  KernelApiIrBuilderTestBase()
      : module_("KernelApiIrBuilderTest", context_),
        kernel_api_ir_builder_(
            context_, KernelApiIrBuilder::Options{true, 256},
            ValidateBuffers ? KernelApiIrBuilder::BufferValidation::kDisjoint
                            : KernelApiIrBuilder::BufferValidation::kNone) {}

  llvm::IRBuilder<> getBuilder() { return llvm::IRBuilder<>(context_); }

  auto EmitKernelPrototype(
      const HloInstruction* instr, const BufferAssignment* buffer_assignment,
      const std::string& module_memory_region_name = "dummy_emitter") {
    return kernel_api_ir_builder_.EmitKernelPrototype(
        module_, instr, buffer_assignment, module_memory_region_name);
  }

  auto EmitKernelPrototype(
      absl::string_view name,
      absl::Span<const KernelApiIrBuilder::KernelParameter> arguments,
      absl::Span<const KernelApiIrBuilder::KernelParameter> results,
      const std::string& module_memory_region_name = "dummy_emitter") {
    return kernel_api_ir_builder_.EmitKernelPrototype(
        module_, name, arguments, results, module_memory_region_name);
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info_, [](LogicalBuffer::Color) { return /*alignment=*/1; });
  }

  void SetKernelFunctionAttributes(llvm::Function* function) {
    kernel_api_ir_builder_.SetKernelFunctionAttributes(function);
  }

  llvm::LLVMContext& context() { return context_; }
  std::string DumpToString() { return llvm_ir::DumpToString(&module_); }

  llvm::Module& module() { return module_; }

 private:
  llvm::LLVMContext context_;
  llvm::Module module_;
  KernelApiIrBuilder kernel_api_ir_builder_;
  AliasInfo alias_info_;
};

using KernelApiIrBuilderTest = KernelApiIrBuilderTestBase<true>;
using KernelApiIrBuilderTestNoBufferValidation =
    KernelApiIrBuilderTestBase<false>;

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
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 2
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
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignment.get()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignment.get()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignment.get()));

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
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelApiIrBuilder::KernelPrototype prototype,
      EmitKernelPrototype(hlo->entry_computation()->root_instruction(),
                          buffer_assignment.get()));

  // The first argument is invariant, the second is not because it's aliased to
  // the output.
  EXPECT_EQ(prototype.invariant_arguments.size(), 1);
  EXPECT_TRUE(prototype.invariant_arguments.contains(0));
}

TEST_F(KernelApiIrBuilderTestNoBufferValidation, PartialOverlap) {
  auto hlo = std::make_unique<HloModule>("test", HloModuleConfig());

  auto shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 2});

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice arg0(&alloc, /*offset=*/0, /*size=*/256);
  BufferAllocation::Slice arg1(&alloc, /*offset=*/256, /*size=*/256);
  BufferAllocation::Slice res0(&alloc, /*offset=*/288, /*size=*/512);
  BufferAllocation::Slice res1(&alloc, /*offset=*/768, /*size=*/256);
  BufferAllocation::Slice res2(&alloc, /*offset=*/1024, /*size=*/256);

  std::vector<KernelApiIrBuilder::KernelParameter> arguments = {{shape, arg0},
                                                                {shape, arg1}};
  std::vector<KernelApiIrBuilder::KernelParameter> results = {
      {shape, res0}, {shape, res1}, {shape, res2}};

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
  EXPECT_NE(prototype.results[2].EmitReadArrayElement(index, &builder),
            nullptr);

  // clang-format off
  ASSERT_TRUE(*RunFileCheck(DumpToString(),
                            absl::StrCat(R"(
    CHECK: define ptr @test(ptr %0) #0 {

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_NumWorkGroups, {{.*}} i32 2
    CHECK:      load i64
    CHECK:      load i64
    CHECK:      load i64

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 0
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 1
    CHECK:      getelementptr inbounds nuw %XLA_CPU_WorkGroupId, {{.*}} i32 2
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

    CHECK-NEXT: getelementptr inbounds nuw %XLA_CPU_KernelCallFrame, {{.*}} i32 3
    CHECK:      load ptr
    CHECK:      getelementptr %XLA_CPU_KernelArg, {{.*}} i32 4, i32 0
    CHECK:      %[[ARG4:.+]] = load ptr, {{.*}}, !invariant.load ![[SCOPE0]], !dereferenceable ![[DEREF_BYTES]], !align ![[ALIGNMENT]]

    CHECK-NEXT: %[[PTR0:.+]] = getelementptr inbounds float, ptr %[[ARG0]]
    CHECK:      load float, ptr %[[PTR0]], align 4,
    CHECK-SAME:                            !invariant.load ![[EMPTY_NODE:.+]],
    CHECK-SAME:                            !noalias ![[ARG0_NOALIAS:.+]]

    CHECK-NEXT: %[[PTR1:.+]] = getelementptr inbounds float, ptr %[[ARG1]]
    CHECK:      load float, ptr %[[PTR1]], align 4,
    CHECK-SAME:                            !noalias ![[ARG1_NOALIAS:.+]]

    CHECK-NEXT: %[[PTR2:.+]] = getelementptr inbounds float, ptr %[[ARG2]]
    CHECK:      load float, ptr %[[PTR2]], align 4, !alias.scope ![[ARG2_SCOPE:.+]],
    CHECK:                                          !noalias ![[ARG4_SCOPE:.+]]

    CHECK-NEXT: %[[PTR3:.+]] = getelementptr inbounds float, ptr %[[ARG3]]
    CHECK:      load float, ptr %[[PTR3]], align 4, !alias.scope ![[ARG3_SCOPE:.+]],
    CHECK:                                          !noalias ![[ARG4_SCOPE]]

    CHECK-NEXT: %[[PTR4:.+]] = getelementptr inbounds float, ptr %[[ARG4]]
    CHECK:      load float, ptr %[[PTR4]], align 4, !alias.scope ![[ARG4_SCOPE]],
    CHECK:                                          !noalias ![[ARG4_NOALIAS:.+]]

    CHECK:      ret ptr null
    CHECK: }

    #0 = { uwtable "frame-pointer"="all" "prefer-vector-width"="256" }
    CHECK-DAG: ![[ALIGNMENT]] = !{i64 )", cpu_function_runtime::MinAlign(), R"(}
    CHECK-DAG: ![[EMPTY_NODE]] = !{}
    CHECK-DAG: ![[DOMAIN:.+]] = !{!"XLA host kernel test AA domain"}
    CHECK-DAG: ![[RES0:.+]] = !{!"{{.*}}, offset:288, size:512}", ![[DOMAIN]]}
    CHECK-DAG: ![[RES1:.+]] = !{!"{{.*}}, offset:768, size:256}", ![[DOMAIN]]}
    CHECK-DAG: ![[RES2:.+]] = !{!"{{.*}}, offset:1024, size:256}", ![[DOMAIN]]}
    CHECK-DAG: ![[ARG2_SCOPE]] = !{![[RES0]]}
    CHECK-DAG: ![[ARG3_SCOPE]] = !{![[RES1]]}
    CHECK-DAG: ![[ARG4_SCOPE]] = !{![[RES2]]}
    CHECK-DAG: ![[ARG0_NOALIAS]] = !{![[RES0]], ![[RES1]], ![[RES2]]}
    CHECK-DAG: ![[ARG1_NOALIAS]] = !{![[RES1]], ![[RES2]]}
    CHECK-DAG: ![[ARG4_NOALIAS]] = !{![[RES0]], ![[RES1]]}
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

TEST_F(KernelApiIrBuilderTest, GetKernelParams) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);
  constexpr absl::string_view hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      p1 = f32[2,2] parameter(1)
      ROOT add.0 = f32[2,2] add(p0, p1)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignment, RunBufferAssignment(*hlo));
  const auto* root = hlo->entry_computation()->root_instruction();
  TF_ASSERT_OK_AND_ASSIGN(auto args,
                          KernelApiIrBuilder::GetKernelArgumentsParameters(
                              root, buffer_assignment.get()));
  EXPECT_EQ(args.size(), 2);
  EXPECT_THAT(args[0].shape.dimensions(), ::testing::ElementsAre(2, 2));
  EXPECT_THAT(args[1].shape.dimensions(), ::testing::ElementsAre(2, 2));
  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          KernelApiIrBuilder::GetKernelResultsParameters(
                              root, buffer_assignment.get()));
  EXPECT_EQ(results.size(), 1);
  EXPECT_THAT(results[0].shape.dimensions(), ::testing::ElementsAre(2, 2));
}

TEST_F(KernelApiIrBuilderTest, SetKernelFunctionAttributes) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);
  llvm::FunctionType* function_ty =
      llvm::FunctionType::get(llvm::PointerType::getUnqual(context),
                              llvm::PointerType::getUnqual(context),
                              /*isVarArg=*/false);
  llvm::Function* function = llvm::Function::Create(
      function_ty, llvm::GlobalValue::ExternalLinkage, "foo", *module);
  EXPECT_FALSE(function->hasFnAttribute("prefer-vector-width"));
  SetKernelFunctionAttributes(function);
  EXPECT_TRUE(function->hasFnAttribute("prefer-vector-width"));
}

TEST_F(KernelApiIrBuilderTest, SetModuleMemoryRegionName) {
  const std::string memory_region_name = "kernel_api_ir_builder_test";
  TF_ASSERT_OK_AND_ASSIGN(
      auto prototype, EmitKernelPrototype("test", {}, {}, memory_region_name));

  SetModuleMemoryRegionName(module(), memory_region_name);

  llvm::NamedMDNode* memory_region_name_md =
      module().getNamedMetadata(std::string(kMemoryRegionNameMetadataName));
  EXPECT_NE(memory_region_name_md, nullptr);
  EXPECT_GT(memory_region_name_md->getNumOperands(), 0);
  llvm::MDNode* node = memory_region_name_md->getOperand(0);
  EXPECT_NE(node, nullptr);
  EXPECT_GT(node->getNumOperands(), 0);
  llvm::MDString* md_str = llvm::dyn_cast<llvm::MDString>(node->getOperand(0));
  EXPECT_NE(md_str, nullptr);
  llvm::StringRef mem_region_name_str = md_str->getString();

  EXPECT_EQ(mem_region_name_str.str(), memory_region_name);
}

}  // namespace
}  // namespace xla::cpu
