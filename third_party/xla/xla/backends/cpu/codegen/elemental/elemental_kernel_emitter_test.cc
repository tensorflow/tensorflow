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

#include "xla/backends/cpu/codegen/elemental/elemental_kernel_emitter.h"

#include <cstdint>
#include <memory>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Type.h"
#include "xla/codegen/kernel_definition.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/target_machine_features_stub.h"
#include "xla/service/logical_buffer.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

class ElementalKernelEmitterTest : public HloHardwareIndependentTestBase {
 public:
  ElementalKernelEmitterTest()
      : target_machine_features_([](int64_t size) { return 1; }) {}

  absl::StatusOr<KernelDefinition> EmitKernelDefinition(
      const HloInstruction* instr, const BufferAssignment* buffer_assignment) {
    ElementalKernelEmitter emitter(instr, buffer_assignment,
                                   &target_machine_features_);

    return emitter.EmitKernelDefinition();
  }

  absl::StatusOr<std::unique_ptr<BufferAssignment>> RunBufferAssignment(
      const HloModule& hlo) {
    return BufferAssigner::Run(
        &hlo, std::make_unique<DependencyHloOrdering>(&hlo),
        [](const BufferValue& buffer) {
          return CpuExecutable::ShapeSizeBytes(buffer.shape());
        },
        [](LogicalBuffer::Color) { return /*alignment=*/1; });
  }

 private:
  TargetMachineFeaturesStub target_machine_features_;
};

namespace {

TEST_F(ElementalKernelEmitterTest, EmitElementalKernel) {
  const char* hlo_text = R"(
    HloModule m
    ENTRY main {
      p0 = f32[2,2] parameter(0)
      ROOT convert = s32[2,2] convert(p0)
    })";

  TF_ASSERT_OK_AND_ASSIGN(auto hlo, ParseAndReturnUnverifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelDefinition kernel_definition,
      EmitKernelDefinition(hlo->entry_computation()->root_instruction(),
                           buffer_assignement.get()));

  ASSERT_TRUE(*RunFileCheck(kernel_definition.source().ToString(), R"(
    CHECK: define ptr @convert_kernel(ptr %0) #0 {
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

TEST_F(ElementalKernelEmitterTest, EmitParallelKernel) {
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
  TF_ASSERT_OK_AND_ASSIGN(auto buffer_assignement, RunBufferAssignment(*hlo));
  TF_ASSERT_OK_AND_ASSIGN(
      KernelDefinition kernel_definition,
      EmitKernelDefinition(hlo->entry_computation()->root_instruction(),
                           buffer_assignement.get()));

  ASSERT_TRUE(*RunFileCheck(kernel_definition.source().ToString(), R"(
    CHECK: @convert_parallel_bounds = private constant [8 x [4 x [2 x i64]]]

    CHECK: define ptr @convert_kernel(ptr %0) #0 {
    CHECK:   %[[X:.*]] = load i64, ptr %workgroup_id_x_gep, align 4
    CHECK:   %lo_dim_0_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 0, i32 0
    CHECK:   %up_dim_0_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 0, i32 1
    CHECK:   %lo_dim_1_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 1, i32 0
    CHECK:   %up_dim_1_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 1, i32 1
    CHECK:   %lo_dim_2_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 2, i32 0
    CHECK:   %up_dim_2_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 2, i32 1
    CHECK:   %lo_dim_3_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 3, i32 0
    CHECK:   %up_dim_3_gep = getelementptr{{.*}} i32 0, i64 %[[X]], i32 3, i32 1
    CHECK:   fptosi float {{.*}} to i32
    CHECK: }
  )"));
}

}  // namespace
}  // namespace xla::cpu
