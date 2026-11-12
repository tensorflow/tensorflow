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

#include "xla/backends/cpu/codegen/emitters/slice_to_dynamic_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/CodeGen.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/cpu/codegen/fusion_compiler.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
#include "xla/codegen/mlir_kernel_source.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/logical_buffer.h"
#include "xla/types.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class SliceToDynamicKernelEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& module) {
    AliasInfo alias_info;
    return BufferAssigner::Run(
        &module, std::make_unique<DependencyHloOrdering>(&module),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        &alias_info, [](LogicalBuffer::Color) { return 1; },
        BufferAssigner::Options{});
  }

  absl::StatusOr<std::unique_ptr<FunctionLibrary>> CompileKernel(
      LlvmKernelSource llvm_source, absl::string_view kernel_name) {
    std::unique_ptr<IrCompiler> ir_compiler =
        IrCompiler::Create({},
                           IrCompiler::Options{
                               /*opt_level=*/llvm::CodeGenOptLevel::Default,
                               /*optimize_for_size=*/false,
                               TargetMachineOptions(GetDebugOptionsFromFlags()),
                           },
                           IrCompiler::CompilationHooks());
    ABSL_ASSIGN_OR_RETURN(JitCompiler jit,
                     JitCompiler::Create({}, std::move(ir_compiler)));
    ABSL_RETURN_IF_ERROR(jit.AddModule(std::move(llvm_source).thread_safe_module()));
    return std::move(jit).Compile(
        {FunctionLibrary::Sym<FunctionLibrary::Kernel>(kernel_name)});
  }
};

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesSliceToDynamic) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic
    ENTRY main {
      data = f32[2,3]{1,0} parameter(0)
      dim0 = s32[] parameter(1)
      dim1 = s32[] parameter(2)
      ROOT res = f32[<=2,<=3]{1,0} custom-call(data, dim0, dim1),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  KernelSpec spec = definition.spec();
  EXPECT_EQ(spec.name(), "res");
  EXPECT_THAT(spec.invariant_arguments(), UnorderedElementsAre(0, 1, 2));

  EXPECT_THAT(RunFileCheck(definition.source().ToString(), R"(
    CHECK: func.func @res(
    CHECK:   %[[D0:.*]] = tensor.extract %arg1[] : tensor<i32>
    CHECK:   xla.set_dynamic_dim_size %[[D0]] into %arg3[0] : tensor<2x3xf32>
    CHECK:   %[[D1:.*]] = tensor.extract %arg2[] : tensor<i32>
    CHECK:   xla.set_dynamic_dim_size %[[D1]] into %{{.*}}[1] : tensor<2x3xf32>
    CHECK:   scf.for
    CHECK:     scf.for
    CHECK:       %[[VAL:.*]] = tensor.extract %arg0[%{{.*}}, %{{.*}}] : tensor<2x3xf32>
    CHECK:       xla.linear_insert %[[VAL]] into %{{.*}}[%{{.*}}] : tensor<2x3xf32>
  )"),
              IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: define ptr @res(ptr{{.*}})
    CHECK: store i32
    CHECK: store float
  )"),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  std::array<float, 6> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  int32_t dim0 = 2;
  int32_t dim1 = 2;
  alignas(16) std::array<uint8_t, 32> output_bytes = {};

  std::vector<XLA_CPU_KernelArg> args = {
      {input.data(), input.size() * sizeof(float)},
      {&dim0, sizeof(dim0)},
      {&dim1, sizeof(dim1)},
      {output_bytes.data(), output_bytes.size()},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);

  std::array<float, 4> packed_values = {};
  std::memcpy(packed_values.data(), output_bytes.data(),
              packed_values.size() * sizeof(float));
  EXPECT_THAT(packed_values, ElementsAre(1.0f, 2.0f, 4.0f, 5.0f));

  std::array<int32_t, 2> metadata_dims = {};
  std::memcpy(metadata_dims.data(), output_bytes.data() + 24,
              metadata_dims.size() * sizeof(int32_t));
  EXPECT_THAT(metadata_dims, ElementsAre(2, 2));
}

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesUnalignedF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_f16
    ENTRY main {
      data = f16[5]{0} parameter(0)
      dim0 = s32[] parameter(1)
      ROOT res = f16[<=5]{0} custom-call(data, dim0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: define ptr @res(ptr{{.*}})
    CHECK: store i32 %{{.*}}, ptr {{.*}} align 1
    CHECK: store half
  )"),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  std::array<half, 5> input = {half(1.0f), half(2.0f), half(3.0f), half(4.0f),
                               half(5.0f)};
  int32_t dim0 = 3;
  alignas(16) std::array<uint8_t, 14> output_bytes = {};

  std::vector<XLA_CPU_KernelArg> args = {
      {input.data(), input.size() * sizeof(half)},
      {&dim0, sizeof(dim0)},
      {output_bytes.data(), output_bytes.size()},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);

  std::array<half, 3> packed_values = {};
  std::memcpy(packed_values.data(), output_bytes.data(),
              packed_values.size() * sizeof(half));
  EXPECT_THAT(packed_values, ElementsAre(half(1.0f), half(2.0f), half(3.0f)));

  int32_t metadata_dim = -1;
  std::memcpy(&metadata_dim, output_bytes.data() + 10, sizeof(metadata_dim));
  EXPECT_EQ(metadata_dim, 3);
}

TEST_F(SliceToDynamicKernelEmitterTest,
       EmitsAndExecutesMultiDimNonDefaultLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_layout
    ENTRY main {
      data = s32[3,3,2]{2,0,1} parameter(0)
      dim0 = s32[] parameter(1)
      dim1 = s32[] parameter(2)
      dim2 = s32[] parameter(3)
      ROOT res = s32[3,<=3,<=2]{2,0,1} custom-call(data, dim0, dim1, dim2),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  // Static shape [3,3,2] with layout {2,0,1}: offset(i0,i1,i2) = i2 + 2*(i0 +
  // 3*i1). Slice to dynamic bounds (3, 2, 1).
  std::array<int32_t, 18> input = {};
  for (int32_t i0 = 0; i0 < 3; ++i0) {
    for (int32_t i1 = 0; i1 < 3; ++i1) {
      for (int32_t i2 = 0; i2 < 2; ++i2) {
        input[i2 + 2 * (i0 + 3 * i1)] = i0 * 100 + i1 * 10 + i2 + 1;
      }
    }
  }
  std::array<int32_t, 3> dims = {3, 2, 1};
  alignas(16) std::array<uint8_t, 84> output_bytes = {};

  std::vector<XLA_CPU_KernelArg> args = {
      {input.data(), input.size() * sizeof(int32_t)},
      {&dims[0], sizeof(int32_t)},
      {&dims[1], sizeof(int32_t)},
      {&dims[2], sizeof(int32_t)},
      {output_bytes.data(), output_bytes.size()},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);

  std::array<int32_t, 6> packed_values = {};
  std::memcpy(packed_values.data(), output_bytes.data(),
              packed_values.size() * sizeof(int32_t));
  EXPECT_THAT(packed_values, ElementsAre(1, 101, 201, 11, 111, 211));

  std::array<int32_t, 3> metadata_dims = {};
  std::memcpy(metadata_dims.data(), output_bytes.data() + 72,
              metadata_dims.size() * sizeof(int32_t));
  EXPECT_THAT(metadata_dims, ElementsAre(3, 2, 1));
}

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesSubByteS4) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_s4
    ENTRY main {
      data = s4[5]{0} parameter(0)
      dim0 = s32[] parameter(1)
      ROOT res = s4[<=5]{0} custom-call(data, dim0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  std::array<uint8_t, 3> input_bytes = {0x21, 0x43, 0x05};
  int32_t dim0 = 3;
  alignas(16) std::array<uint8_t, 7> output_bytes = {};

  std::vector<XLA_CPU_KernelArg> args = {
      {input_bytes.data(), input_bytes.size()},
      {&dim0, sizeof(dim0)},
      {output_bytes.data(), output_bytes.size()},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);

  EXPECT_EQ(output_bytes[0], 0x21);
  EXPECT_EQ(output_bytes[1] & 0x0f, 0x03);
  int32_t metadata_dim = -1;
  std::memcpy(&metadata_dim, output_bytes.data() + 3, sizeof(metadata_dim));
  EXPECT_EQ(metadata_dim, 3);
}

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesZeroElementArray) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_zero_elems
    ENTRY main {
      data = s32[0,4]{1,0} parameter(0)
      dim0 = s32[] parameter(1)
      dim1 = s32[] parameter(2)
      ROOT res = s32[0,<=4]{1,0} custom-call(data, dim0, dim1),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  EXPECT_THAT(RunFileCheck(definition.source().ToString(), R"(
    CHECK: xla.set_dynamic_dim_size %{{.*}} into %{{.*}}[0]
    CHECK: xla.set_dynamic_dim_size %{{.*}} into %{{.*}}[1]
    CHECK-NOT: xla.linear_insert
  )"),
              IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  int32_t dim0 = 0;
  int32_t dim1 = 3;
  alignas(16) std::array<int32_t, 2> output_dims = {-1, -1};
  int32_t dummy_input = 0;
  std::vector<XLA_CPU_KernelArg> args = {
      {&dummy_input, 0},
      {&dim0, sizeof(dim0)},
      {&dim1, sizeof(dim1)},
      {output_dims.data(), output_dims.size() * sizeof(int32_t)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);
  EXPECT_THAT(output_dims, ElementsAre(0, 3));
}

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesRank0) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_rank0
    ENTRY main {
      data = s32[] parameter(0)
      ROOT res = s32[] custom-call(data), custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "res"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res"));

  int32_t input = 42;
  int32_t output = -1;
  std::vector<XLA_CPU_KernelArg> args = {
      {&input, sizeof(input)},
      {&output, sizeof(output)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ((*kernel)(&frame), nullptr);
  EXPECT_EQ(output, 42);
}

TEST_F(SliceToDynamicKernelEmitterTest, RejectsNonSliceToDynamic) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule not_slice_to_dynamic
    ENTRY main {
      p = f32[4] parameter(0)
      ROOT neg = f32[4] negate(p)
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  EXPECT_THAT(emitter.EmitKernelDefinition(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected SliceToDynamic custom call")));
}

TEST_F(SliceToDynamicKernelEmitterTest, RejectsUnsupportedSubByteS1) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule slice_to_dynamic_s1
    ENTRY main {
      data = s1[16]{0:E(1)} parameter(0)
      dim0 = s32[] parameter(1)
      ROOT res = s1[<=16]{0:E(1)} custom-call(data, dim0),
        custom_call_target="SliceToDynamic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  SliceToDynamicKernelEmitter emitter(*mlir_context, root, assignment.get());
  EXPECT_THAT(emitter.EmitKernelDefinition(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support sub-byte type s1")));
}

}  // namespace
}  // namespace xla::cpu
