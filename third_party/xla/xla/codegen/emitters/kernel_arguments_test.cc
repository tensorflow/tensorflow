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

#include "xla/codegen/emitters/kernel_arguments.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::emitters {
namespace {
using ::testing::ElementsAre;
using ::testing::SizeIs;

using KernelArgumentsTest = HloHardwareIndependentTestBase;

int64_t BufferSizeBytes(const BufferValue& buffer) {
  return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
}

TEST_F(KernelArgumentsTest, GetArgumentBufferSlices) {
  constexpr absl::string_view kHloString = R"(
    HloModule module

    ENTRY entry {
      param.0 = f32[1,2,3]{2,1,0} parameter(0)
      param.1 = f32[1,2,3]{2,1,0} parameter(1)
      ROOT add = f32[1,2,3]{2,1,0} add(param.0, param.1)
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(kHloString));
  AliasInfo alias_info;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 0; },
          /*allocate_buffers_for_constants=*/true));

  // Three allocations: one for each parameter, plus one for the output.
  EXPECT_THAT(assignment->Allocations(), SizeIs(3));

  TF_ASSERT_OK_AND_ASSIGN(
      auto kernel_arguments,
      KernelArguments::Create(*assignment, gpu::GetDefaultBufferAlignment(),
                              module->entry_computation()->root_instruction()));

  // Three arguments, one for each parameter, plus one for the output.
  EXPECT_THAT(kernel_arguments.args(), SizeIs(3));

  // Three slices, one for each parameter, plus one for the output.
  EXPECT_THAT(kernel_arguments.GetArgumentBufferSlices(), SizeIs(3));

  constexpr size_t kExpectedBufferSize = 1 * 2 * 3 * sizeof(float);
  EXPECT_THAT(
      kernel_arguments.GetArgumentBufferSlices(),
      ElementsAre(BufferAllocation::Slice(&assignment->Allocations()[1],
                                          /*offset=*/0, kExpectedBufferSize),
                  BufferAllocation::Slice(&assignment->Allocations()[2],
                                          /*offset=*/0, kExpectedBufferSize),
                  // The output is last in KernelArguments.
                  BufferAllocation::Slice(&assignment->Allocations()[0],
                                          /*offset=*/0, kExpectedBufferSize)));
  EXPECT_THAT(kernel_arguments.GetArgumentOutputFlags(),
              ElementsAre(false, false, true));
}

}  // namespace
}  // namespace xla::emitters
