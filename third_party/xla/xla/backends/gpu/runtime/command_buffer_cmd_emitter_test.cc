/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/command_buffer_cmd_emitter.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/kernel_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace xla::gpu {

TEST(CommandBufferCmdEmitterTest, ConcurrentAndSequentialExecutionGraphs) {
  BufferAllocation allocation(/*index=*/0, /*size=*/2048, /*color=*/0);
  BufferAllocation::Slice slice_a(&allocation, /*offset=*/0, /*size=*/1024);
  BufferAllocation::Slice slice_b(&allocation, /*offset=*/1024, /*size=*/2048);

  ThunkSequence thunks;
  emitters::KernelArgument arg_a(ShapeUtil::MakeShape(F32, {256}), slice_a);
  arg_a.set_written(true);
  auto thunk_a = std::make_unique<KernelThunk>(
      Thunk::ThunkInfo(),
      /*kernel_name=*/"kernel",
      /*kernel_arguments=*/
      emitters::KernelArguments(std::vector<emitters::KernelArgument>{arg_a}),
      /*launch_dimensions=*/LaunchDimensions(),
      /*cluster_dim=*/se::ClusterDim(),
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata(),
      /*zeroed_output_buffer_indices=*/std::vector<int64_t>{});
  thunks.push_back(std::move(thunk_a));

  emitters::KernelArgument arg_b(ShapeUtil::MakeShape(F32, {256}), slice_b);
  arg_b.set_written(true);
  auto thunk_b = std::make_unique<KernelThunk>(
      Thunk::ThunkInfo(),
      /*kernel_name=*/"kernel",
      /*kernel_arguments=*/
      emitters::KernelArguments(std::vector<emitters::KernelArgument>{arg_b}),
      /*launch_dimensions=*/LaunchDimensions(),
      /*cluster_dim=*/se::ClusterDim(),
      /*shmem_bytes=*/0,
      /*tma_metadata=*/se::gpu::TmaMetadata(),
      /*zeroed_output_buffer_indices=*/std::vector<int64_t>{});
  thunks.push_back(std::move(thunk_b));

  ConvertToCommandsOptions options;
  {
    // Synchronization mode kConcurrent creates a concurrent execution graph.
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrent;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a sequential execution
    // graph when all thunks are in different concurrent regions.
    thunks[0]->set_concurrent_region_id(42);
    thunks[1]->set_concurrent_region_id(43);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_TRUE(commands.execution_graph()->is_sequential());
  }

  {
    // Synchronization mode kConcurrentRegions creates a concurrent execution
    // graph when thunks are in the same regions.
    thunks[0]->set_concurrent_region_id(44);
    thunks[1]->set_concurrent_region_id(44);
    options.synchronization_mode =
        CommandExecutor::SynchronizationMode::kConcurrentRegions;
    TF_ASSERT_OK_AND_ASSIGN(CommandExecutor commands,
                            ConvertToCommands(thunks, options));
    EXPECT_FALSE(commands.execution_graph()->is_sequential());
  }
}

}  // namespace xla::gpu
