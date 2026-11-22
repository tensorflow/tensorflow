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

#include "xla/backends/cpu/codegen/emitters/pad_to_static_kernel_emitter.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/base/attributes.h"
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

class PadToStaticKernelEmitterTest : public HloHardwareIndependentTestBase {
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

  ABSL_ATTRIBUTE_NO_SANITIZE_CFI static XLA_CPU_KernelError* CallKernel(
      FunctionLibrary::Kernel* kernel, XLA_CPU_KernelCallFrame* frame) {
    return (*kernel)(frame);
  }
};

TEST_F(PadToStaticKernelEmitterTest, EmitsAndExecutesUnalignedF16) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule pad_to_static_f16
    ENTRY main {
      param = f16[<=5]{0} parameter(0)
      ROOT pts = (f16[5]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  PadToStaticKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  KernelSpec spec = definition.spec();
  EXPECT_EQ(spec.name(), "pts");
  EXPECT_THAT(spec.invariant_arguments(), UnorderedElementsAre(0));

  EXPECT_THAT(RunFileCheck(definition.source().ToString(), R"(
    CHECK: func.func @pts(
    CHECK:   %[[DIM:.*]] = xla.get_dynamic_dim_size %arg0 0 : tensor<5xf16>
    CHECK:   tensor.insert %[[DIM]] into %arg2[] : tensor<i32>
    CHECK:   scf.for %{{.*}} = %c0 to %{{.*}} step %c1 iter_args(%{{.*}} = %arg1, %[[IDX:.*]] = %c0)
    CHECK:     %[[VAL:.*]] = xla.linear_extract %arg0[%[[IDX]]] : tensor<5xf16>
    CHECK:     tensor.insert %[[VAL]] into %{{.*}}[%{{.*}}] : tensor<5xf16>
  )"),
              IsOkAndHolds(true));

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  EXPECT_THAT(RunFileCheck(llvm_source.ToString(), R"(
    CHECK: define ptr @pts(ptr{{.*}})
    CHECK: load i32, ptr {{.*}} align 1
    CHECK: store half
  )"),
              IsOkAndHolds(true));

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "pts"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("pts"));

  alignas(16) std::array<uint8_t, 14> input_bytes = {};
  std::array<half, 5> input_vals = {half(1.0f), half(2.0f), half(3.0f),
                                    half(-9.0f), half(-9.0f)};
  std::memcpy(input_bytes.data(), input_vals.data(), 10);
  int32_t dyn_size = 3;
  std::memcpy(input_bytes.data() + 10, &dyn_size, sizeof(dyn_size));

  std::array<half, 5> output_vals = {};
  int32_t output_dim = -1;
  std::vector<XLA_CPU_KernelArg> args = {
      {input_bytes.data(), input_bytes.size()},
      {output_vals.data(), output_vals.size() * sizeof(half)},
      {&output_dim, sizeof(output_dim)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallKernel(kernel, &frame), nullptr);

  EXPECT_EQ(output_dim, 3);
  EXPECT_THAT(std::vector<half>(output_vals.begin(), output_vals.begin() + 3),
              ElementsAre(half(1.0f), half(2.0f), half(3.0f)));
}

TEST_F(PadToStaticKernelEmitterTest, EmitsAndExecutesMultiDimNonDefaultLayout) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule pad_to_static_layout
    ENTRY main {
      param = s32[3,<=3,<=2]{2,0,1} parameter(0)
      ROOT pts = (s32[3,3,2]{2,0,1}, s32[], s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  PadToStaticKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "pts"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("pts"));

  // Input is packed in {2,0,1} physical order for dynamic bounds (3, 2, 1):
  // (i1=0, i0=0..2, i2=0), (i1=1, i0=0..2, i2=0).
  std::array<int32_t, 18> input_data = {};
  input_data.fill(-1);
  input_data[0] = 1;
  input_data[1] = 101;
  input_data[2] = 201;
  input_data[3] = 11;
  input_data[4] = 111;
  input_data[5] = 211;
  std::array<int32_t, 3> input_dims = {3, 2, 1};
  alignas(16) std::array<uint8_t, 84> input_bytes = {};
  std::memcpy(input_bytes.data(), input_data.data(), 72);
  std::memcpy(input_bytes.data() + 72, input_dims.data(), 12);

  std::array<int32_t, 18> output_data = {};
  std::array<int32_t, 3> output_dims = {-1, -1, -1};
  std::vector<XLA_CPU_KernelArg> args = {
      {input_bytes.data(), input_bytes.size()},
      {output_data.data(), output_data.size() * sizeof(int32_t)},
      {&output_dims[0], sizeof(int32_t)},
      {&output_dims[1], sizeof(int32_t)},
      {&output_dims[2], sizeof(int32_t)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallKernel(kernel, &frame), nullptr);

  EXPECT_THAT(output_dims, ElementsAre(3, 2, 1));
  // Output static shape [3,3,2] with layout {2,0,1}: offset(i0,i1,i2) = i2 +
  // 2*(i0 + 3*i1).
  EXPECT_EQ(output_data[0], 1);
  EXPECT_EQ(output_data[2], 101);
  EXPECT_EQ(output_data[4], 201);
  EXPECT_EQ(output_data[6], 11);
  EXPECT_EQ(output_data[8], 111);
  EXPECT_EQ(output_data[10], 211);
}

TEST_F(PadToStaticKernelEmitterTest, EmitsAndExecutesSubByteS4) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule pad_to_static_s4
    ENTRY main {
      param = s4[<=5]{0} parameter(0)
      ROOT pts = (s4[5]{0}, s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  PadToStaticKernelEmitter emitter(*mlir_context, root, assignment.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<MlirKernelSource> definition,
                       emitter.EmitKernelDefinition());

  FusionCompiler compiler(mlir_context.get(),
                          FusionCompiler::Options{512, 1, true});
  ASSERT_OK_AND_ASSIGN(LlvmKernelSource llvm_source,
                       compiler.Compile(std::move(definition).TakeSource()));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<FunctionLibrary> library,
                       CompileKernel(std::move(llvm_source), "pts"));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("pts"));

  // Packed s4[<=5]: 3 data bytes (nibbles: 1, 2, 3, -1, -1) followed by s32 3.
  alignas(16) std::array<uint8_t, 7> input_bytes = {};
  input_bytes[0] = 0x21;  // elems 0=1, 1=2
  input_bytes[1] = 0x03;  // elem 2=3
  int32_t dyn_size = 3;
  std::memcpy(input_bytes.data() + 3, &dyn_size, sizeof(dyn_size));

  std::array<uint8_t, 3> output_bytes = {};
  int32_t output_dim = -1;
  std::vector<XLA_CPU_KernelArg> args = {
      {input_bytes.data(), input_bytes.size()},
      {output_bytes.data(), output_bytes.size()},
      {&output_dim, sizeof(output_dim)},
  };
  XLA_CPU_NumWorkGroups num_workgroups = {1, 1, 1};
  XLA_CPU_WorkGroupId workgroup_id = {0, 0, 0};
  XLA_CPU_KernelCallFrame frame = {&num_workgroups, &workgroup_id, args.size(),
                                   args.data()};
  EXPECT_EQ(CallKernel(kernel, &frame), nullptr);

  EXPECT_EQ(output_dim, 3);
  EXPECT_EQ(output_bytes[0], 0x21);
  EXPECT_EQ(output_bytes[1] & 0x0f, 0x03);
}

TEST_F(PadToStaticKernelEmitterTest, RejectsNonPadToStatic) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule not_pad_to_static
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
  PadToStaticKernelEmitter emitter(*mlir_context, root, assignment.get());
  EXPECT_THAT(emitter.EmitKernelDefinition(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected PadToStatic custom call")));
}

TEST_F(PadToStaticKernelEmitterTest, RejectsUnsupportedSubByteS1) {
  constexpr absl::string_view kHlo = R"hlo(
    HloModule pad_to_static_s1
    ENTRY main {
      param = s1[<=4,<=6]{1,0:E(1)} parameter(0)
      ROOT pts = (s1[4,6]{1,0:E(1)}, s32[], s32[]) custom-call(param),
        custom_call_target="PadToStatic"
    }
  )hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  std::unique_ptr<mlir::MLIRContext> mlir_context =
      FusionCompiler::CreateContext();
  PadToStaticKernelEmitter emitter(*mlir_context, root, assignment.get());
  EXPECT_THAT(emitter.EmitKernelDefinition(),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("does not support sub-byte type s1")));
}

}  // namespace
}  // namespace xla::cpu
