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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_module_config.h"
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

std::unique_ptr<AllGatherStartThunk> CreateAllGatherStartThunk() {
  auto create_replica_groups =
      [](const std::vector<std::vector<int64_t>>& replica_groups) {
        std::vector<ReplicaGroup> result;
        result.reserve(replica_groups.size());
        for (const auto& replica_group : replica_groups) {
          ReplicaGroup& group = result.emplace_back();
          for (auto id : replica_group) {
            group.add_replica_ids(id);
          }
        }
        return result;
      };

  std::vector<ReplicaGroup> replica_groups =
      create_replica_groups({{0, 1}, {2, 3}});

  HloModule module("test_module", HloModuleConfig());
  auto builder = HloComputation::Builder("test_builder");
  auto param_shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction* param_0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "p0"));

  HloInstruction* all_gather_start =
      builder.AddInstruction(HloInstruction::CreateAllGatherStart(
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::MakeShape(F32, {8, 4}), std::move(param_shape)}),
          {param_0}, /*all_gather_dimension=*/0, replica_groups,
          /*constrain_layout=*/false,
          /*channel_id=*/2, /*use_global_device_ids=*/false));

  Thunk::ThunkInfo thunk_info;
  std::vector<std::unique_ptr<Thunk>> thunks;

  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 16 * 4);
  BufferAllocation::Slice slice1(&alloc1, 0, 16 * 4);

  CollectiveThunk::Buffer buffer;
  buffer.source_buffer = slice0;
  buffer.destination_buffer = slice1;

  return std::make_unique<AllGatherStartThunk>(
      std::move(thunk_info),
      static_cast<const HloAllGatherInstruction*>(all_gather_start),
      std::vector<CollectiveThunk::Buffer>({buffer}), false);
}

std::unique_ptr<CopyThunk> CreateCopyThunk() {
  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  return std::make_unique<CopyThunk>(Thunk::ThunkInfo(), slice0, slice0, 1024);
}

std::unique_ptr<GemmThunk> CreateGemmThunk() {
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
  return std::make_unique<GemmThunk>(Thunk::ThunkInfo(), config.value(), slice1,
                                     slice1, slice1, slice1, true);
}

std::unique_ptr<CollectiveDoneThunk> CreateAllGatherDoneThunk(
    Thunk* start_thunk) {
  auto async_events =
      static_cast<const AllGatherStartThunk*>(start_thunk)->async_events();
  return std::make_unique<CollectiveDoneThunk>(
      Thunk::kAllGatherDone, Thunk::ThunkInfo(), std::move(async_events),
      AsyncStreamKind::kCollective);
}

TEST(CommandBufferConversionPassTest, ConvertsToCommandBufferThunk) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo();

  // Create a CopyThunk
  thunks.push_back(CreateCopyThunk());

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

  // Create a {CopyThunk, GemmThunk, CopyThunk}
  thunks.push_back(CreateCopyThunk());
  thunks.push_back(CreateGemmThunk());
  thunks.push_back(CreateCopyThunk());

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

TEST(CommandBufferConversionPassTest, ConvertsAsyncPairToCommandBuffer) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Create a start thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 2);

  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);

  // Expected transformation:
  // SequentialThunk(AllGatherStartThunk, CollectiveDoneThunk) ->
  // SequentialThunk(CommandBufferThunk(AllGatherStartThunk,
  // CollectiveDoneThunk))
  EXPECT_EQ(root_thunk->thunks().size(), 1);

  const Thunk* thunk = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  auto* command_buffer_thunk = static_cast<const CommandBufferThunk*>(thunk);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer.size(), 2);
  EXPECT_EQ(thunks_in_command_buffer[0]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[1]->kind(), Thunk::kAllGatherDone);
}

TEST(CommandBufferConversionPassTest,
     DontConvertAsyncsIfNonConvertibleThunkInBetween) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Create a start thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Create a non-convertible thunk
  thunks.push_back(CreateCopyThunk());

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 3);

  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));

  // Expected no transformation, because there is a non-convertible thunk in
  // between the asyncs.
  EXPECT_FALSE(changed);
}

TEST(CommandBufferConversionPassTest, ConvertCrossedAsyncs) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Create start thunk A
  thunks.push_back(CreateAllGatherStartThunk());
  // Create start thunk B
  thunks.push_back(CreateAllGatherStartThunk());
  // Create a done thunk A
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));
  // Create a done thunk B
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 4);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);

  // Expected transformation: Convert all 4 thunks into command buffer
  EXPECT_EQ(root_thunk->thunks().size(), 1);

  const Thunk* thunk = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  auto* command_buffer_thunk = static_cast<const CommandBufferThunk*>(thunk);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer.size(), 4);
  EXPECT_EQ(thunks_in_command_buffer[0]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[1]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[2]->kind(), Thunk::kAllGatherDone);
  EXPECT_EQ(thunks_in_command_buffer[3]->kind(), Thunk::kAllGatherDone);
}

TEST(CommandBufferConversionPassTest, ConvertNestedAsyncs) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Create start thunk A
  thunks.push_back(CreateAllGatherStartThunk());
  // Create start thunk B
  thunks.push_back(CreateAllGatherStartThunk());
  // Create a done thunk B
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));
  // Create a convertible thunk C
  thunks.push_back(CreateGemmThunk());
  // Create a done thunk A
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 5);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);

  // Expected transformation: Convert all 5 thunks into command buffer
  EXPECT_EQ(root_thunk->thunks().size(), 1);

  const Thunk* thunk = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  auto* command_buffer_thunk = static_cast<const CommandBufferThunk*>(thunk);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer.size(), 5);
  EXPECT_EQ(thunks_in_command_buffer[0]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[1]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[2]->kind(), Thunk::kAllGatherDone);
  EXPECT_EQ(thunks_in_command_buffer[3]->kind(), Thunk::kGemm);
  EXPECT_EQ(thunks_in_command_buffer[4]->kind(), Thunk::kAllGatherDone);
}

TEST(CommandBufferConversionPassTest, DontConvertAsyncsIfUnpairedStart) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Convertible CopyThunk in the beginning
  thunks.push_back(CreateCopyThunk());

  // Start A Thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Start B Thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Done A Thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  // Another convertible CopyThunk
  thunks.push_back(CreateCopyThunk());

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 5);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);

  // Expected transformation: {Copy, AllGatherStart0, AllGatherStart1,
  // AllGatherDone0, Copy} -> {CommandBuffer(Copy), AllGatherStart0,
  // AllGatherStart1, AllGatherDone0, CommandBuffer(Copy)}
  EXPECT_EQ(root_thunk->thunks().size(), 5);

  const Thunk* thunk0 = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk0->kind(), Thunk::kCommandBuffer);
  auto* command_buffer_thunk0 = static_cast<const CommandBufferThunk*>(thunk0);
  const auto& thunks_in_command_buffer0 =
      command_buffer_thunk0->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer0[0]->kind(), Thunk::kCopy);

  EXPECT_EQ(root_thunk->thunks()[1]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(root_thunk->thunks()[2]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(root_thunk->thunks()[3]->kind(), Thunk::kAllGatherDone);

  const Thunk* thunk4 = root_thunk->thunks()[4].get();
  EXPECT_EQ(thunk4->kind(), Thunk::kCommandBuffer);
  auto* command_buffer_thunk4 = static_cast<const CommandBufferThunk*>(thunk4);
  const auto& thunks_in_command_buffer4 =
      command_buffer_thunk4->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer4[0]->kind(), Thunk::kCopy);
}

TEST(CommandBufferConversionPassTest, ConvertsAsyncPairsMixedWithOtherThunks) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  // Create a start thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  // Convertible thunk in the middle
  thunks.push_back(CreateCopyThunk());

  // Create a start thunk
  thunks.push_back(CreateAllGatherStartThunk());

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  // Pack the thunks into a root thunk
  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  EXPECT_EQ(root_thunk->thunks().size(), 5);

  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass;
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_TRUE(changed);

  // Expected transformation:
  // SequentialThunk(AllGatherStartThunk0, CollectiveDoneThunk0, CopyThunk,
  // AllGatherStartThunk1, AllGatherDoneThunk1) ->
  // SequentialThunk(CommandBufferThunk(/*The same sequence of thunks*/))
  EXPECT_EQ(root_thunk->thunks().size(), 1);

  const Thunk* thunk = root_thunk->thunks()[0].get();
  EXPECT_EQ(thunk->kind(), Thunk::kCommandBuffer);

  auto* command_buffer_thunk = static_cast<const CommandBufferThunk*>(thunk);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_EQ(thunks_in_command_buffer.size(), 5);
  EXPECT_EQ(thunks_in_command_buffer[0]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[1]->kind(), Thunk::kAllGatherDone);
  EXPECT_EQ(thunks_in_command_buffer[2]->kind(), Thunk::kCopy);
  EXPECT_EQ(thunks_in_command_buffer[3]->kind(), Thunk::kAllGatherStart);
  EXPECT_EQ(thunks_in_command_buffer[4]->kind(), Thunk::kAllGatherDone);
}

TEST(CommandBufferConversionPassTest, DontConvertIfNotMinGraphSize) {
  std::vector<std::unique_ptr<Thunk>> thunks;
  Thunk::ThunkInfo thunk_info = Thunk::ThunkInfo();

  thunks.push_back(CreateCopyThunk());

  auto root_thunk =
      std::make_unique<SequentialThunk>(Thunk::ThunkInfo(), std::move(thunks));
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(2);

  se::DeviceDescription device_info;

  EXPECT_EQ(root_thunk->thunks().size(), 1);

  CommandBufferConversionPass pass;

  // The size of the sequence is less than the min graph size, so it should not
  // be converted to a command buffer.
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed, pass.Run(root_thunk.get(), debug_options, device_info));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
