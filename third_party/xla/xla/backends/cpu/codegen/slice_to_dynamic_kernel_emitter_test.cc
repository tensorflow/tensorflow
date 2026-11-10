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

#include "xla/backends/cpu/codegen/slice_to_dynamic_kernel_emitter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "xla/backends/cpu/codegen/ir_compiler.h"
#include "xla/backends/cpu/codegen/jit_compiler.h"
#include "xla/backends/cpu/codegen/target_machine_features.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/kernel_c_api.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/codegen/kernel_spec.h"
#include "xla/codegen/llvm_kernel_source.h"
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
#include "xla/service/llvm_ir/llvm_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::cpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

class SliceToDynamicKernelEmitterTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    ASSERT_OK_AND_ASSIGN(
        target_machine_,
        IrCompiler::InferTargetMachine(
            llvm::TargetOptions(), llvm::CodeGenOptLevel::Default,
            TargetMachineOptions(GetDebugOptionsFromFlags())));
    features_ = std::make_unique<TargetMachineFeatures>(target_machine_.get());
  }

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

  std::unique_ptr<llvm::TargetMachine> target_machine_;
  std::unique_ptr<TargetMachineFeatures> features_;
};

TEST_F(SliceToDynamicKernelEmitterTest, EmitsAndExecutesSliceToDynamic) {
  constexpr absl::string_view kHlo = R"(
    HloModule slice_to_dynamic
    ENTRY main {
      data = f32[2,3]{1,0} parameter(0)
      dim0 = s32[] parameter(1)
      dim1 = s32[] parameter(2)
      ROOT res = f32[<=2,<=3]{1,0} custom-call(data, dim0, dim1),
        custom_call_target="SliceToDynamic"
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  SliceToDynamicKernelEmitter emitter(root, assignment.get(), features_.get());
  ASSERT_OK_AND_ASSIGN(KernelDefinition<LlvmKernelSource> definition,
                       emitter.EmitKernelDefinition());

  KernelSpec spec = definition.spec();
  EXPECT_EQ(spec.name(), "res_kernel");
  EXPECT_THAT(spec.invariant_arguments(), UnorderedElementsAre(0, 1, 2));

  LlvmKernelSource source = std::move(definition).TakeSource();
  EXPECT_THAT(RunFileCheck(llvm_ir::DumpToString(source.module()), R"(
    CHECK: define ptr @res_kernel(ptr{{.*}})
    CHECK: store i32 %dyn_dim_size
    CHECK: store float
  )"),
              IsOkAndHolds(true));

  std::unique_ptr<IrCompiler> ir_compiler = IrCompiler::Create(
      llvm::TargetOptions(),
      IrCompiler::Options{llvm::CodeGenOptLevel::Default,
                          /*optimize_for_size=*/false,
                          TargetMachineOptions(GetDebugOptionsFromFlags())},
      IrCompiler::CompilationHooks());
  ASSERT_OK_AND_ASSIGN(JitCompiler jit,
                       JitCompiler::Create({}, std::move(ir_compiler)));
  ASSERT_OK(jit.AddModule(std::move(source).thread_safe_module()));
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<FunctionLibrary> library,
      std::move(jit).Compile(
          {FunctionLibrary::Sym<FunctionLibrary::Kernel>("res_kernel")}));
  ASSERT_OK_AND_ASSIGN(
      FunctionLibrary::Kernel * kernel,
      library->ResolveFunction<FunctionLibrary::Kernel>("res_kernel"));

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

TEST_F(SliceToDynamicKernelEmitterTest, RejectsNonSliceToDynamic) {
  constexpr absl::string_view kHlo = R"(
    HloModule not_slice_to_dynamic
    ENTRY main {
      p = f32[4] parameter(0)
      ROOT neg = f32[4] negate(p)
    }
  )";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHlo));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BufferAssignment> assignment,
                       RunBufferAssignment(*module));
  const HloInstruction* root = module->entry_computation()->root_instruction();

  SliceToDynamicKernelEmitter emitter(root, assignment.get(), features_.get());
  EXPECT_THAT(emitter.EmitKernelDefinition(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Expected SliceToDynamic custom-call")));
}

}  // namespace
}  // namespace xla::cpu
