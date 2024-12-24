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

#include "xla/service/cpu/ir_emitter2.h"

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "xla/cpu_function_runtime.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/ir_emitter.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape_util.h"
#include "xla/tests/filecheck.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::cpu {

class IrEmitter2Test : public HloTestBase {
 public:
  // This is a proxy function that allows us call private method
  // IrEmitter2::EmitKernelPrototype.
  static auto EmitKernelPrototype(
      IrEmitter2& ir_emitter,
      const std::vector<IrEmitter2::KernelParameter>& arguments,
      const std::vector<IrEmitter2::KernelParameter>& results) {
    return ir_emitter.EmitKernelPrototype("test", arguments, results);
  }

  absl::StatusOr<IrEmitter2> MakeIrEmitter2(llvm::Module& module,
                                            const HloModule& hlo) {
    TF_ASSIGN_OR_RETURN(
        buffer_assignment_,
        BufferAssigner::Run(
            &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
            backend().compiler()->BufferSizeBytesFunction(),
            [](LogicalBuffer::Color) { return /*alignment=*/1; }));

    target_machine_ = std::make_unique<TargetMachineFeaturesStub>(
        [](int64_t size) { return 1; });

    nested_ir_emitter_ = absl::WrapUnique(
        new IrEmitter(nullptr, hlo, *buffer_assignment_, &module, {}, {}, {},
                      target_machine_.get(), false));

    return IrEmitter2(hlo, &module, nested_ir_emitter_.get());
  }

  // TODO(abanas): This function could be static. It requires making the
  // underlying FindInstruction function static first.
  absl::StatusOr<IrEmitter2::KernelInfo> EmitElementalHostKernel(
      IrEmitter2& ir_emitter, HloModule& hlo,
      absl::string_view instruction_name) {
    HloInstruction* instruction = FindInstruction(&hlo, instruction_name);

    if (instruction == nullptr) {
      return absl::InternalError("Instruction not found");
    }
    TF_ASSIGN_OR_RETURN(IrEmitter2::KernelInfo kernel,
                        ir_emitter.EmitElementalHostKernel(instruction));
    return kernel;
  }

 private:
  // Dependencies of IrEmitter2. These are created in MakeIrEmitter2 and kept
  // alive for the duration of the test, because IrEmitter2 does not take
  // ownership of them.
  std::unique_ptr<BufferAssignment> buffer_assignment_;
  std::unique_ptr<TargetMachineFeaturesStub> target_machine_;
  std::unique_ptr<IrEmitter> nested_ir_emitter_;
};

namespace {

TEST_F(IrEmitter2Test, BuildKernelPrototype) {
  auto hlo = std::make_unique<HloModule>("test", HloModuleConfig());

  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  auto shape = ShapeUtil::MakeShape(PrimitiveType::F32, {4, 2});

  BufferAllocation alloc(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation::Slice arg0(&alloc, /*offset=*/0, /*size=*/256);
  BufferAllocation::Slice arg1(&alloc, /*offset=*/256, /*size=*/256);
  BufferAllocation::Slice res0(&alloc, /*offset=*/512, /*size=*/256);
  BufferAllocation::Slice res1(&alloc, /*offset=*/768, /*size=*/256);

  std::vector<IrEmitter2::KernelParameter> arguments = {{shape, arg0},
                                                        {shape, arg1}};
  std::vector<IrEmitter2::KernelParameter> results = {{shape, res0},
                                                      {shape, res1}};

  IrEmitter2 ir_emitter(*hlo, module.get(), /*nested_ir_emitter=*/nullptr);
  TF_ASSERT_OK_AND_ASSIGN(auto prototype,
                          EmitKernelPrototype(ir_emitter, arguments, results));

  llvm::IRBuilder<> b(context);
  b.SetInsertPoint(prototype.function->getEntryBlock().getTerminator());

  auto* zero = llvm::ConstantInt::get(llvm::Type::getInt32Ty(context), 0);
  llvm_ir::IrArray::Index index(zero, shape, &b);

  // Emit loads from arguments and results buffers to test alias scope metadata.
  EXPECT_NE(prototype.arguments[0].EmitReadArrayElement(index, &b), nullptr);
  EXPECT_NE(prototype.arguments[1].EmitReadArrayElement(index, &b), nullptr);
  EXPECT_NE(prototype.results[0].EmitReadArrayElement(index, &b), nullptr);
  EXPECT_NE(prototype.results[1].EmitReadArrayElement(index, &b), nullptr);

  // clang-format off
  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()),
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
  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK:      {{.+}} = load ptr, {{.*}}, !dereferenceable ![[DEREF_BYTES:.+]],
    CHECK: ![[DEREF_BYTES]] = !{i64 32}
  )"));
}

TEST_F(IrEmitter2Test, EmitElementalKernel) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT convert = s32[2,2] convert(p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "convert"));

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: define ptr @convert(ptr %0) #0 {
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

TEST_F(IrEmitter2Test, EmitParallelKernel) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[1,2,1,16384,256] parameter(0)
      ROOT convert = s32[1,2,1,16384,256] convert(p0),
        backend_config={"outer_dimension_partitions":["1","2","1","4"]}
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "convert"));

  ASSERT_TRUE(*RunFileCheck(llvm_ir::DumpToString(module.get()), R"(
    CHECK: @convert_parallel_bounds = private constant [8 x [4 x [2 x i64]]]

    CHECK: define ptr @convert(ptr %0) #0 {
    CHECK:   %lo_dim_0_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 0, i32 0
    CHECK:   %up_dim_0_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 0, i32 1
    CHECK:   %lo_dim_1_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 1, i32 0
    CHECK:   %up_dim_1_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 1, i32 1
    CHECK:   %lo_dim_2_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 2, i32 0
    CHECK:   %up_dim_2_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 2, i32 1
    CHECK:   %lo_dim_3_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 3, i32 0
    CHECK:   %up_dim_3_gep = getelementptr{{.*}} i32 0, i64 %tid_x, i32 3, i32 1
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

using IrEmitter2InvariantBuffersTest = IrEmitter2Test;

TEST_F(IrEmitter2InvariantBuffersTest, AllInvariantBuffers) {
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
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "add.0"));

  ASSERT_EQ(kernel.invariant_arguments.size(), 2);
}

TEST_F(IrEmitter2InvariantBuffersTest, InvariantBufferPassedTwice) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT add.0 = f32[2,2] add(p0, p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "add.0"));

  // Invariant buffers contains indices of both arguments, even though it is the
  // same buffer slice.
  ASSERT_EQ(kernel.invariant_arguments.size(), 2);
}

TEST_F(IrEmitter2InvariantBuffersTest, NoInvariantBuffers) {
  llvm::LLVMContext context;
  auto module = std::make_unique<llvm::Module>("test", context);

  const char* hlo_text = R"(
    HloModule m, input_output_alias={ {}: (0, {}, must-alias) }
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT add.0 = f32[2,2] add(p0, p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "add.0"));

  ASSERT_EQ(kernel.invariant_arguments.size(), 0);
}

TEST_F(IrEmitter2InvariantBuffersTest, MixedBuffers) {
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
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2 ir_emitter, MakeIrEmitter2(*module, *hlo));
  TF_ASSERT_OK_AND_ASSIGN(IrEmitter2::KernelInfo kernel,
                          EmitElementalHostKernel(ir_emitter, *hlo, "add.0"));

  // The first argument is invariant, the second is not because it's aliased to
  // the output.
  EXPECT_EQ(kernel.invariant_arguments.size(), 1);
  EXPECT_TRUE(kernel.invariant_arguments.contains(0));
}

}  // namespace
}  // namespace xla::cpu
