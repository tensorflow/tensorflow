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

#include "xla/service/gpu/transforms/command_buffer_conversion_pass.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/platform_util.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

std::string GetPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

se::StreamExecutor* GpuExecutor() {
  stream_executor::Platform* platform =
      se::PlatformManager::PlatformWithName(GetPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

TEST(CommandBufferConversionPassTest, ConvertsToCommandBufferThunk) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo();

  // Create a CopyThunk
  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  thunks.push_back(
      std::make_unique<CopyThunk>(thunk_info, slice0, slice0, 1024));

  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info;

  EXPECT_EQ(root_thunk->thunks().size(), 1);

  CommandBufferConversionPass pass;

  // CopyThunk should be converted to a CommandBufferThunk, because it is
  // supported in command buffers. The expected transformation is:
  // SequentialThunk(CopyThunk) ->
  // SequentialThunk(CommandBufferThunk(CopyThunk))
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root_thunk->thunks().size(), 1);

  const Thunk* thunk = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  auto* command_buffer_thunk = static_cast<const CommandBufferThunk*>(thunk);

  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer.size(), 1);

  EXPECT_EQ(thunks_in_command_buffer[0]->kind(), Thunk::kCopy);
}

TEST(CommandBufferConversionPassTest, PartiallyConvertsToCommandBufferThunk) {
  CommandBufferConversionPass pass;

  std::vector<std::unique_ptr<Thunk>> thunks;
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo();

  // Create a CopyThunk
  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  thunks.push_back(
      std::make_unique<CopyThunk>(thunk_info, slice0, slice0, 1024));

  // Create a GemmThunk
  se::StreamExecutor* executor = GpuExecutor();

  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      executor->GetDeviceDescription().gpu_compute_capability());
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation::Slice slice1(&alloc1, 0, 16 * 4);
  thunks.push_back(std::make_unique<GemmThunk>(
      thunk_info, config.value(), slice1, slice1, slice1, slice1, true));

  // Create another CopyThunk
  thunks.push_back(
      std::make_unique<CopyThunk>(thunk_info, slice0, slice0, 1024));

  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  DebugOptions debug_options;

  // Enable only FUSION, which means GemmThunk should not be converted.
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  se::DeviceDescription device_info;

  EXPECT_EQ(root_thunk->thunks().size(), 3);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);
  EXPECT_EQ(root_thunk->thunks().size(), 3);

  // Expected transformation: (Copy, Gemm, Copy) -> (CommandBuffer(Copy), Gemm,
  // CommandBuffer(Copy))
  EXPECT_EQ(root_thunk->thunks()[0]->kind(), Thunk::kCommandBuffer);
  EXPECT_EQ(root_thunk->thunks()[1]->kind(), Thunk::kGemm);
  EXPECT_EQ(root_thunk->thunks()[2]->kind(), Thunk::kCommandBuffer);

  // Check the content of the first command buffer thunk
  auto* command_buffer_thunk0 =
      dynamic_cast<const CommandBufferThunk*>(root_thunk->thunks()[0].get());
  ASSERT_NE(command_buffer_thunk0, nullptr);
  const auto& thunks_in_command_buffer0 =
      command_buffer_thunk0->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer0[0]->kind(), Thunk::kCopy);

  // Check the content of the second command buffer thunk
  auto* command_buffer_thunk1 =
      dynamic_cast<const CommandBufferThunk*>(root_thunk->thunks()[2].get());
  ASSERT_NE(command_buffer_thunk1, nullptr);
  const auto& thunks_in_command_buffer1 =
      command_buffer_thunk1->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer1[0]->kind(), Thunk::kCopy);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
