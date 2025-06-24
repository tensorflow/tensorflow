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
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

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

}  // namespace
}  // namespace gpu
}  // namespace xla
