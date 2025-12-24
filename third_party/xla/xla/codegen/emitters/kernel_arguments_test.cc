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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/buffer_value.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/logical_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla::emitters {
namespace {
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
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
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 0; },
          std::move(opts)));

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

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesTest) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[10] parameter(0)
  param1 = f32[20] parameter(1)
  param2 = f32[30] parameter(2)

  ROOT tuple_result = (f32[10], f32[20]) tuple(param0, param1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  AliasInfo alias_info;
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          [](const BufferValue& buffer) {
            return ShapeUtil::ByteSizeOf(buffer.shape(), sizeof(void*));
          },
          &alias_info, [](LogicalBuffer::Color) { return 1; },
          std::move(opts)));

  KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = 1;
  buffer_alignment.constant_buffer_align_bytes = 1;
  buffer_alignment.xla_allocated_buffer_align_bytes = 1;

  // Test 1: Create regular (non-interleaved) arguments for baseline
  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments regular_args,
      KernelArguments::Create(*buffer_assignment, buffer_alignment, root,
                              absl::Span<const int32_t>{}));

  // Test 2: Create interleaved arguments
  // Expected order: input0, output0, input1, output1
  std::vector<int32_t> interleaved_indices = {1, 3};
  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments interleaved_args,
      KernelArguments::Create(*buffer_assignment, buffer_alignment, root,
                              interleaved_indices));

  // Get buffer slices for comparison
  auto regular_slices = regular_args.GetArgumentBufferSlices();
  auto interleaved_slices = interleaved_args.GetArgumentBufferSlices();

  // Verify sizes
  ASSERT_EQ(regular_slices.size(), 4);      // 2 inputs + 2 outputs
  ASSERT_EQ(interleaved_slices.size(), 4);  // same total count

  // Verify interleaving worked by comparing buffer slices:
  // Regular order:     [input0, input1, output0, output1]
  // Interleaved order: [input0, output0, input1, output1]
  EXPECT_EQ(interleaved_slices[0],
            regular_slices[0]);  // input0 stays at position 0
  EXPECT_EQ(interleaved_slices[1],
            regular_slices[2]);  // output0 moves to position 1
  EXPECT_EQ(interleaved_slices[2],
            regular_slices[1]);  // input1 moves to position 2
  EXPECT_EQ(interleaved_slices[3],
            regular_slices[3]);  // output1 moves to position 3
}

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesEdgeCases) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)

  ROOT tuple_result = (f32[5]) tuple(param0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  AliasInfo alias_info;
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 1; },
          std::move(opts)));

  KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = 1;
  buffer_alignment.constant_buffer_align_bytes = 1;
  buffer_alignment.xla_allocated_buffer_align_bytes = 1;

  // Test 1: Create regular (non-interleaved) arguments for baseline
  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments regular_args,
      KernelArguments::Create(*buffer_assignment, buffer_alignment, root,
                              absl::Span<const int32_t>{}));

  // Test 2: Create interleaved arguments - output at beginning (position 0)
  // Expected order: output0, input0 (instead of input0, output0)
  std::vector<int32_t> interleaved_indices = {0};
  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments interleaved_args,
      KernelArguments::Create(*buffer_assignment, buffer_alignment, root,
                              interleaved_indices));

  // Get buffer slices for comparison
  auto regular_slices = regular_args.GetArgumentBufferSlices();
  auto interleaved_slices = interleaved_args.GetArgumentBufferSlices();

  // Verify sizes
  ASSERT_EQ(regular_slices.size(), 2);      // 1 input + 1 output
  ASSERT_EQ(interleaved_slices.size(), 2);  // same total count

  // Verify interleaving worked by comparing buffer slices:
  // Regular order:     [input0, output0]
  // Interleaved order: [output0, input0]
  EXPECT_EQ(interleaved_slices[0],
            regular_slices[1]);  // output0 moves to position 0
  EXPECT_EQ(interleaved_slices[1],
            regular_slices[0]);  // input0 moves to position 1

  // Also verify by checking shapes
  const auto& interleaved_shapes = interleaved_args.args();

  // Both should be f32[5] but in different order
  EXPECT_EQ(interleaved_shapes[0].shape(),
            ShapeUtil::MakeShape(F32, {5}));  // output0 at position 0
  EXPECT_EQ(interleaved_shapes[1].shape(),
            ShapeUtil::MakeShape(F32, {5}));  // input0 at position 1
}

TEST_F(KernelArgumentsTest, InterleavedOutputIndicesErrorCases) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)

  ROOT result = f32[5] add(param0, param0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  AliasInfo alias_info;
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 1; },
          std::move(opts)));

  KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = 1;
  buffer_alignment.constant_buffer_align_bytes = 1;
  buffer_alignment.xla_allocated_buffer_align_bytes = 1;

  // Test case: Output index out of bounds
  // root->operands() = {param0, param0} (2 operands, but same parameter used
  // twice) outputs = {result} (1 output) Total positions = 3, so index 5 is out
  // of bounds
  std::vector<int32_t> invalid_indices = {5};
  EXPECT_THAT(KernelArguments::Create(*buffer_assignment, buffer_alignment,
                                      root, invalid_indices),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Output index out of bounds")));
}

TEST_F(KernelArgumentsTest, EmptyInterleavedIndicesFallback) {
  const absl::string_view hlo_string = R"(
HloModule TestModule

ENTRY main {
  param0 = f32[5] parameter(0)
  ROOT result = f32[5] add(param0, param0)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  HloInstruction* root = module->entry_computation()->root_instruction();

  AliasInfo alias_info;
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> buffer_assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 1; },
          std::move(opts)));

  KernelArguments::BufferAlignment buffer_alignment;
  buffer_alignment.entry_parameter_align_bytes = 1;
  buffer_alignment.constant_buffer_align_bytes = 1;
  buffer_alignment.xla_allocated_buffer_align_bytes = 1;

  // Test case: Empty interleaved indices should fall back to regular Create
  std::vector<int32_t> empty_indices = {};

  TF_ASSERT_OK_AND_ASSIGN(
      KernelArguments kernel_args,
      KernelArguments::Create(*buffer_assignment, buffer_alignment, root,
                              empty_indices));

  // Should succeed and create arguments in regular order: inputs first, then
  // outputs
  ASSERT_EQ(kernel_args.args().size(), 3);  // 2 inputs + 1 output
}

TEST_F(KernelArgumentsTest, UnmanagedArguments) {
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
  BufferAssigner::Options opts;
  opts.allocate_buffers_for_constants = true;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<BufferAssignment> assignment,
      BufferAssigner::Run(
          module.get(), std::make_unique<DependencyHloOrdering>(module.get()),
          &BufferSizeBytes, &alias_info, [](LogicalBuffer::Color) { return 0; },
          std::move(opts)));
  // Input and output buffers are managed.
  EXPECT_THAT(assignment->Allocations(), SizeIs(3));
  auto unmanaged_arguments = std::vector{
      ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(U32, {}),
      ShapeUtil::MakeShape(F32, {24}), ShapeUtil::MakeShape(F32, {65536})};

  TF_ASSERT_OK_AND_ASSIGN(
      auto kernel_arguments,
      KernelArguments::Create(*assignment, gpu::GetDefaultBufferAlignment(),
                              module->entry_computation()->root_instruction(),
                              unmanaged_arguments));
  // 3 managed arguments + 4 unmanaged arguments.
  ASSERT_THAT(kernel_arguments.args(), SizeIs(7));

  constexpr size_t kExpectedBufferSize = 1 * 2 * 3 * sizeof(float);
  EXPECT_THAT(
      kernel_arguments.GetArgumentBufferSlices(),
      ElementsAre(BufferAllocation::Slice(&assignment->Allocations()[1],
                                          /*offset=*/0, kExpectedBufferSize),
                  BufferAllocation::Slice(&assignment->Allocations()[2],
                                          /*offset=*/0, kExpectedBufferSize),
                  // The output is last in KernelArguments.
                  BufferAllocation::Slice(&assignment->Allocations()[0],
                                          /*offset=*/0, kExpectedBufferSize),
                  BufferAllocation::Slice(), BufferAllocation::Slice(),
                  BufferAllocation::Slice(), BufferAllocation::Slice()));
  constexpr auto kManaged = KernelArgument::Kind::kManaged;
  constexpr auto kUnmanaged = KernelArgument::Kind::kUnmanaged;
  EXPECT_THAT(kernel_arguments.GetArgumentKinds(),
              ElementsAre(kManaged, kManaged, kManaged, kUnmanaged, kUnmanaged,
                          kUnmanaged, kUnmanaged));
}

}  // namespace
}  // namespace xla::emitters
