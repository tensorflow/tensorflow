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

#include "xla/backends/gpu/runtime/command_buffer_conversion_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/all_gather_thunk.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/convolution_thunk.h"
#include "xla/backends/gpu/runtime/cudnn_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/execution_stream_id.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/replica_id_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Pointee;

MATCHER_P(ThunkKindIs, kind, "") { return arg.kind() == kind; }

template <typename... Matchers>
auto ThunkKindsAre(Matchers... matchers) {
  return ElementsAre(Pointee(ThunkKindIs(matchers))...);
}

std::string GetPlatformName() {
  return absl::AsciiStrToUpper(
      PlatformUtil::CanonicalPlatformName("gpu").value());
}

se::StreamExecutor* GpuExecutor() {
  stream_executor::Platform* platform =
      se::PlatformManager::PlatformWithName(GetPlatformName()).value();
  return platform->ExecutorForDevice(0).value();
}

std::unique_ptr<AllGatherThunk> CreateAllGatherThunk(
    const BufferAllocation& alloc0, const BufferAllocation& alloc1) {
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
              {ShapeUtil::MakeShape(F32, {8, 4}), param_shape}),
          {param_0}, /*all_gather_dimension=*/0, replica_groups,
          /*constrain_layout=*/false,
          /*channel_id=*/2, /*use_global_device_ids=*/false));

  ThunkSequence thunks;

  BufferAllocation::Slice slice0(&alloc0, 0, 16 * 4);
  BufferAllocation::Slice slice1(&alloc1, 0, 16 * 4);

  CollectiveThunk::Buffer buffer;
  buffer.source_buffer = {slice0, param_shape};
  buffer.destination_buffer = {slice1, param_shape};

  return std::make_unique<AllGatherThunk>(
      Thunk::ThunkInfo(),
      static_cast<const HloAllGatherInstruction*>(all_gather_start),
      std::vector<CollectiveThunk::Buffer>({buffer}), false);
}

std::unique_ptr<DeviceToDeviceCopyThunk> CreateCopyThunk(
    const BufferAllocation& alloc0) {
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  Shape shape = ShapeUtil::MakeShape(S32, {256});
  return std::make_unique<DeviceToDeviceCopyThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice0, shape},
      ShapedSlice{slice0, shape}, 1024);
}

std::unique_ptr<GemmThunk> CreateGemmThunk(const BufferAllocation& alloc1) {
  se::StreamExecutor* executor = GpuExecutor();
  auto config = GemmConfig::For(
      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 3}), {}, {1},
      ShapeUtil::MakeShape(PrimitiveType::F32, {3, 1}), {}, {0},
      ShapeUtil::MakeShape(PrimitiveType::F32, {1, 1}), 1.0, 0.0, 0.0,
      PrecisionConfig::ALG_UNSET, std::nullopt,
      se::blas::kDefaultComputePrecision, false, false,
      /*scale_mode=*/se::gpu::ScaleMode::kNone,
      executor->GetDeviceDescription().gpu_compute_capability());
  BufferAllocation::Slice slice1(&alloc1, 0, 16 * 4);
  return std::make_unique<GemmThunk>(Thunk::ThunkInfo(), config.value(), slice1,
                                     slice1, slice1, slice1, true);
}

std::unique_ptr<ConvolutionThunk> CreateConvolutionThunk(
    const BufferAllocation& alloc) {
  std::vector<ShapedSlice> operand_slices, result_slices;
  for (int i = 0, num = 3; i < num; i++) {
    operand_slices.emplace_back(
        ShapedSlice{BufferAllocation::Slice{&alloc, i * 16, 16}, Shape{}});
    result_slices.emplace_back(ShapedSlice{
        BufferAllocation::Slice{&alloc, (i + num) * 16, 16}, Shape{}});
  }

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_input_feature_dimension(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_input_spatial_dimensions(3);
  dnums.set_kernel_input_feature_dimension(0);
  dnums.set_kernel_output_feature_dimension(1);
  dnums.add_kernel_spatial_dimensions(2);
  dnums.add_kernel_spatial_dimensions(3);
  dnums.set_output_batch_dimension(0);
  dnums.set_output_feature_dimension(1);
  dnums.add_output_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(3);

  Window window;
  const auto dim0 = window.add_dimensions();
  const auto dim1 = window.add_dimensions();
  dim0->set_size(4);
  dim1->set_size(4);
  dim0->set_base_dilation(1);
  dim1->set_base_dilation(1);
  dim0->set_stride(1);
  dim1->set_stride(1);
  dim0->set_window_dilation(3);
  dim1->set_window_dilation(2);

  GpuConvDescriptor desc{
      .kind = CudnnConvKind::kForward,
      .backend_config = CudnnConvBackendConfig{},
      .operand0_shape = ShapeUtil::MakeShape(F32, {60, 38, 17, 13}),
      .operand1_shape = ShapeUtil::MakeShapeWithDenseLayout(F32, {38, 10, 4, 4},
                                                            {3, 2, 0, 1}),
      .result_shape = ShapeUtil::MakeShapeWithType<float>({64, 64, 64, 13}),
      .scratch_size = 128 * 1024,
      .window = window,
      .dnums = dnums,
      .feature_group_count = 1};
  auto thunk =
      ConvolutionThunk::Create(Thunk::ThunkInfo(), desc, operand_slices,
                               result_slices, result_slices.back().slice);
  TF_CHECK_OK(thunk.status());
  return std::move(thunk).value();
}

std::unique_ptr<AsyncStartThunk> WrapInAsyncStartThunk(
    std::unique_ptr<AllGatherThunk> start_thunk) {
  ThunkSequence sequence;
  sequence.push_back(std::move(start_thunk));
  return std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo(), CommunicationStreamId(0), std::move(sequence));
}

std::unique_ptr<AsyncDoneThunk> CreateAllGatherDoneThunk(Thunk* start_thunk) {
  auto async_execution =
      static_cast<const AsyncStartThunk*>(start_thunk)->async_execution();
  return std::make_unique<AsyncDoneThunk>(Thunk::ThunkInfo(),
                                          std::move(async_execution));
}

std::unique_ptr<WhileThunk> CreateWhileThunk(ThunkSequence condition_thunks,
                                             ThunkSequence body_thunks,
                                             const BufferAllocation& alloc) {
  BufferAllocation::Slice slice(&alloc, 0, 1024);

  return std::make_unique<WhileThunk>(Thunk::ThunkInfo(), slice,
                                      std::move(condition_thunks),
                                      std::move(body_thunks));
}

std::unique_ptr<ConditionalThunk> CreateConditionalThunk(
    std::vector<ThunkSequence> branch_thunks) {
  BufferAllocation alloc(0, 1024, 0);
  BufferAllocation::Slice slice(&alloc, 0, 1024);
  Shape shape = ShapeUtil::MakeShape(S32, {});

  return std::make_unique<ConditionalThunk>(
      Thunk::ThunkInfo(), ShapedSlice{slice, shape}, std::move(branch_thunks));
}

std::unique_ptr<CuDnnThunk> CreateCuDnnThunk(const BufferAllocation& alloc0) {
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  return std::make_unique<CuDnnThunk>(
      /*fingerprint=*/"fingerprint", Thunk::ThunkInfo(),
      /*args=*/
      std::vector<ShapedSlice>{{slice0, ShapeUtil::MakeShape(F32, {256})}},
      /*output_args=*/std::vector<bool>{true});
}

std::unique_ptr<PartitionIdThunk> CreatePartitionIdThunk(
    const BufferAllocation& alloc0) {
  BufferAllocation::Slice slice0(&alloc0, 0, 1024);
  return std::make_unique<PartitionIdThunk>(Thunk::ThunkInfo(), slice0);
}

class FakeErrorAllocator : public ThunkPassBufferAllocator {
 public:
  absl::StatusOr<BufferAllocation*> NewEmptyAllocation(int64_t size) override {
    return absl::InternalError("FakeErrorAllocator: Allocation failed");
  }
};

TEST(CommandBufferConversionPassTest, ConvertsToCommandBufferThunk) {
  ThunkSequence thunks;

  // Create a CopyThunk
  BufferAllocation alloc0(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc0));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info;
  FakeErrorAllocator allocator;

  CommandBufferConversionPass pass{"test"};

  // CopyThunk should be converted to a CommandBufferThunk, because it is
  // supported in command buffers. The expected transformation is:
  // SequentialThunk(CopyThunk) ->
  // SequentialThunk(CommandBufferThunk(CopyThunk))
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));
  EXPECT_THAT(thunks[0]->thunk_info().profile_annotation, "command_buffer_0");

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());

  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer, ThunkKindsAre(Thunk::kCopy));
}

TEST(CommandBufferConversionPassTest, PartiallyConvertsToCommandBufferThunk) {
  CommandBufferConversionPass pass{"test"};

  ThunkSequence thunks;

  // Create a {CopyThunk, GemmThunk, CopyThunk}
  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(CreateCopyThunk(alloc0));
  thunks.push_back(CreateGemmThunk(alloc1));
  thunks.push_back(CreateCopyThunk(alloc0));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  // Enable only FUSION, which means GemmThunk should not be converted.
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  se::DeviceDescription device_info;
  FakeErrorAllocator allocator;

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: (Copy, Gemm, Copy) -> (CommandBuffer(Copy), Gemm,
  // CommandBuffer(Copy))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kGemm,
                                    Thunk::kCommandBuffer));

  // Check the content of the first command buffer thunk
  auto* command_buffer_thunk0 =
      dynamic_cast<const CommandBufferThunk*>(thunks[0].get());
  ASSERT_NE(command_buffer_thunk0, nullptr);
  const auto& thunks_in_command_buffer0 =
      command_buffer_thunk0->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer0, ThunkKindsAre(Thunk::kCopy));

  // Check the content of the second command buffer thunk
  auto* command_buffer_thunk1 =
      dynamic_cast<const CommandBufferThunk*>(thunks[2].get());
  ASSERT_NE(command_buffer_thunk1, nullptr);
  const auto& thunks_in_command_buffer1 =
      command_buffer_thunk1->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer1, ThunkKindsAre(Thunk::kCopy));
}

TEST(CommandBufferConversionPassTest, ConvertConvolutionAndGemmThunks) {
  CommandBufferConversionPass pass{"test"};

  ThunkSequence thunks;

  // Create a {CopyThunk, GemmThunk, ConvolutionThunk}
  BufferAllocation alloc0(0, 1024, 0);
  BufferAllocation alloc1(1, 2048, 0);
  BufferAllocation alloc2(2, 2048, 0);
  thunks.push_back(CreateCopyThunk(alloc0));
  thunks.push_back(CreateGemmThunk(alloc1));
  thunks.push_back(CreateConvolutionThunk(alloc0));

  DebugOptions debug_options;

  // Enable only FUSION, which means GemmThunk should not be converted.
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CONVOLUTION);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: Convert all 5 thunks into command buffer
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer,
              ThunkKindsAre(Thunk::kCopy, Thunk::kGemm, Thunk::kConvolution));
}

TEST(CommandBufferConversionPassTest, ConvertsAsyncPairToCommandBuffer) {
  ThunkSequence thunks;
  // Create a start thunk
  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  CommandBufferConversionPass pass("test");
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation:
  // SequentialThunk(AsyncStartThunk, AsyncDoneThunk) ->
  // SequentialThunk(CommandBufferThunk(AsyncStartThunk,
  // AsyncDoneThunk))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer,
              ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest,
     DontConvertAsyncsIfNonConvertibleThunkInBetween) {
  ThunkSequence thunks;
  // Create a start thunk
  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));

  // Create a non-convertible thunk
  BufferAllocation alloc2(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc2));

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  CommandBufferConversionPass pass("test");
  // Expected no transformation, because there is a non-convertible thunk in
  // between the asyncs.
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(false));
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kCopy,
                                    Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest, ConvertCrossedAsyncs) {
  ThunkSequence thunks;
  // Create start thunk A
  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));
  // Create start thunk B
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));
  // Create a done thunk A
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));
  // Create a done thunk B
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass{"test"};
  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  FakeErrorAllocator allocator;
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: Convert all 4 thunks into command buffer
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer,
              ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart,
                            Thunk::kAsyncDone, Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest, ConvertNestedAsyncs) {
  ThunkSequence thunks;
  // Create start thunk A
  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));
  // Create start thunk B
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));
  // Create a done thunk B
  thunks.push_back(CreateAllGatherDoneThunk(thunks[0].get()));
  // Create a convertible thunk C
  BufferAllocation alloc2(1, 16 * 4, 0);
  thunks.push_back(CreatePartitionIdThunk(alloc2));
  // Create a done thunk A
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass{"test"};
  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  FakeErrorAllocator allocator;
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: Convert all 5 thunks into command buffer
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(
      thunks_in_command_buffer,
      ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncStart, Thunk::kAsyncDone,
                    Thunk::kPartitionId, Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest, DontConvertAsyncsIfUnpairedStart) {
  ThunkSequence thunks;
  // Convertible CopyThunk in the beginning
  BufferAllocation alloc0(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc0));

  // Start A Thunk
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation alloc2(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc1, alloc2)));

  // Start B Thunk
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc1, alloc2)));

  // Done A Thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks[1].get()));

  // Another convertible CopyThunk
  thunks.push_back(CreateCopyThunk(alloc0));

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  CommandBufferConversionPass pass{"test"};
  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  FakeErrorAllocator allocator;
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: {Copy, AsyncStart0, AsyncStart1,
  // AsyncDone0, Copy} -> {CommandBuffer(Copy), AsyncStart0,
  // AsyncStart1, AsyncDone0, CommandBuffer(Copy)}
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer, Thunk::kAsyncStart,
                                    Thunk::kAsyncStart, Thunk::kAsyncDone,
                                    Thunk::kCommandBuffer));

  const auto* command_buffer_thunk0 =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer0 =
      command_buffer_thunk0->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer0, ThunkKindsAre(Thunk::kCopy));

  const auto* command_buffer_thunk4 =
      static_cast<const CommandBufferThunk*>(thunks[4].get());
  const auto& thunks_in_command_buffer4 =
      command_buffer_thunk4->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer4, ThunkKindsAre(Thunk::kCopy));
}

TEST(CommandBufferConversionPassTest, ConvertsAsyncPairsMixedWithOtherThunks) {
  ThunkSequence thunks;
  // Create a start thunk
  BufferAllocation alloc0(1, 16 * 4, 0);
  BufferAllocation alloc1(1, 16 * 4, 0);
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  // Convertible thunk in the middle
  BufferAllocation alloc2(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc2));

  // Create a start thunk
  thunks.push_back(WrapInAsyncStartThunk(CreateAllGatherThunk(alloc0, alloc1)));

  // Create a done thunk
  thunks.push_back(CreateAllGatherDoneThunk(thunks.back().get()));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  CommandBufferConversionPass pass("test");
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation:
  // SequentialThunk(AsyncStartThunk0, AsyncDoneThunk0, CopyThunk,
  // AsyncStartThunk1, AsyncDoneThunk1) ->
  // SequentialThunk(CommandBufferThunk(/*The same sequence of thunks*/))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer,
              ThunkKindsAre(Thunk::kAsyncStart, Thunk::kAsyncDone, Thunk::kCopy,
                            Thunk::kAsyncStart, Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest, DontConvertIfNotMinGraphSize) {
  ThunkSequence thunks;

  BufferAllocation alloc0(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc0));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(2);

  se::DeviceDescription device_info;
  FakeErrorAllocator allocator;

  ASSERT_EQ(thunks.size(), 1);

  CommandBufferConversionPass pass{"test"};

  // The size of the sequence is less than the min graph size, so it should not
  // be converted to a command buffer.
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(false));
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCopy));
}

TEST(CommandBufferConversionPassTest, ConvertWhileThunk) {
  if (GetPlatformName() == "ROCM") {
    GTEST_SKIP() << "Not supported on ROCm";
  }
  CommandBufferConversionPass pass{"test"};

  ThunkSequence thunks;

  // Create condition and branch sequences
  ThunkSequence condition_thunks;
  BufferAllocation alloc0(0, 1024, 0);
  condition_thunks.push_back(CreateCopyThunk(alloc0));

  ThunkSequence body_thunks;
  BufferAllocation alloc1(1, 16 * 4, 0);
  body_thunks.push_back(CreateGemmThunk(alloc1));

  // Create a while thunk
  BufferAllocation alloc2(0, 1024, 0);
  thunks.push_back(CreateWhileThunk(std::move(condition_thunks),
                                    std::move(body_thunks), alloc2));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  ASSERT_EQ(thunks.size(), 1);

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: (While({Copy}, {Gemm})) ->
  // (CommandBuffer(While({Copy}, {Gemm})))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  // Check the content of the command buffer thunk
  auto* command_buffer_thunk =
      dynamic_cast<const CommandBufferThunk*>(thunks[0].get());
  ASSERT_NE(command_buffer_thunk, nullptr);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer, ThunkKindsAre(Thunk::kWhile));
  auto* while_thunk_transformed =
      dynamic_cast<const WhileThunk*>(thunks_in_command_buffer[0].get());
  ASSERT_NE(while_thunk_transformed, nullptr);
  EXPECT_THAT(while_thunk_transformed->condition_executor().thunks(),
              ThunkKindsAre(Thunk::kCopy));
  EXPECT_THAT(while_thunk_transformed->body_executor().thunks(),
              ThunkKindsAre(Thunk::kGemm));
}

TEST(CommandBufferConversionPassTest,
     DontConvertConditionalThunkWithNonConvertibleBranch) {
  // Check that if a branch of a conditional thunk is not convertible, the
  // conditional thunk is not convertible either, but the branches are attempted
  // to be converted independently.
  CommandBufferConversionPass pass("test");

  ThunkSequence thunks;

  // Create branch sequences
  ThunkSequence branch0_thunks;
  BufferAllocation alloc0(0, 1024, 0);
  branch0_thunks.push_back(CreateCopyThunk(alloc0));

  ThunkSequence branch1_thunks;
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation alloc2(1, 16 * 4, 0);
  BufferAllocation alloc3(1, 16 * 4, 0);
  branch1_thunks.push_back(CreateAllGatherThunk(alloc1, alloc2));
  branch1_thunks.push_back(CreateCopyThunk(alloc3));

  // Create a conditional thunk
  std::vector<ThunkSequence> branch_thunks;
  branch_thunks.push_back(std::move(branch0_thunks));
  branch_thunks.push_back(std::move(branch1_thunks));

  thunks.push_back(CreateConditionalThunk(std::move(branch_thunks)));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CONDITIONAL);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  ASSERT_EQ(thunks.size(), 1);

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation is: kConditional({kCopy}, {kAllGather, kCopy})
  // -> kConditional(kCommandBuffer(kCopy), {kAllGather,
  // kCommandBuffer(kCopy)}).
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kConditional));
  auto* conditional_thunk =
      dynamic_cast<const ConditionalThunk*>(thunks[0].get());
  ASSERT_NE(conditional_thunk, nullptr);
  EXPECT_THAT(conditional_thunk->branch_executors()[0].thunks(),
              ThunkKindsAre(Thunk::kCommandBuffer));
  EXPECT_THAT(conditional_thunk->branch_executors()[1].thunks(),
              ThunkKindsAre(Thunk::kAllGather, Thunk::kCommandBuffer));
}

TEST(CommandBufferConversionPassTest, ConvertWhileThunkWithAsyncPair) {
  if (GetPlatformName() == "ROCM") {
    GTEST_SKIP() << "Not supported on ROCm";
  }
  CommandBufferConversionPass pass{"test"};

  ThunkSequence thunks;

  // Create condition and branch sequences
  ThunkSequence condition_thunks;
  BufferAllocation alloc0(0, 1024, 0);
  condition_thunks.push_back(CreateCopyThunk(alloc0));

  ThunkSequence body_thunks;
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation alloc2(1, 16 * 4, 0);
  body_thunks.push_back(
      WrapInAsyncStartThunk(CreateAllGatherThunk(alloc1, alloc2)));
  body_thunks.push_back(CreateCopyThunk(alloc0));
  body_thunks.push_back(CreateAllGatherDoneThunk(body_thunks[0].get()));

  // Create a while thunk
  BufferAllocation alloc3(0, 1024, 0);
  thunks.push_back(CreateWhileThunk(std::move(condition_thunks),
                                    std::move(body_thunks), alloc3));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  ASSERT_EQ(thunks.size(), 1);

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // Expected transformation: (While({Copy}, {AsyncStart, Copy,
  // AsyncDone})) -> (CommandBuffer(While({Copy}, {AsyncStart, Copy,
  // AsyncDone})))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  // Check the content of the command buffer thunk
  auto* command_buffer_thunk =
      dynamic_cast<const CommandBufferThunk*>(thunks[0].get());
  ASSERT_NE(command_buffer_thunk, nullptr);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer, ThunkKindsAre(Thunk::kWhile));
  auto* while_thunk_transformed =
      dynamic_cast<const WhileThunk*>(thunks_in_command_buffer[0].get());
  ASSERT_NE(while_thunk_transformed, nullptr);
  EXPECT_THAT(while_thunk_transformed->condition_executor().thunks(),
              ThunkKindsAre(Thunk::kCopy));
  EXPECT_THAT(
      while_thunk_transformed->body_executor().thunks(),
      ThunkKindsAre(Thunk::kAsyncStart, Thunk::kCopy, Thunk::kAsyncDone));
}

TEST(CommandBufferConversionPassTest, ConvertsCuDnnThunkToCommandBufferThunk) {
  ThunkSequence thunks;

  // Create a CuDnnThunk
  BufferAllocation alloc0(0, 1024, 0);
  thunks.push_back(CreateCuDnnThunk(alloc0));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUDNN);

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;

  ASSERT_EQ(thunks.size(), 1);

  CommandBufferConversionPass pass{"test"};

  // The expected transformation is: SequentialThunk(CuDnnThunk) ->
  // SequentialThunk(CommandBufferThunk(CuDnnThunk))
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));

  const auto* command_buffer_thunk =
      static_cast<const CommandBufferThunk*>(thunks[0].get());

  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer, ThunkKindsAre(Thunk::kCuDnn));
}

TEST(CommandBufferConversionPassTest, ConvertTheBodyOfWhileThunk) {
  CommandBufferConversionPass pass{"test"};

  ThunkSequence thunks;

  // Create condition and branch sequences
  ThunkSequence condition_thunks;
  BufferAllocation alloc0(0, 1024, 0);
  condition_thunks.push_back(CreateCopyThunk(alloc0));

  ThunkSequence body_thunks;
  BufferAllocation alloc1(1, 16 * 4, 0);
  BufferAllocation alloc2(1, 16 * 4, 0);
  BufferAllocation alloc3(1, 16 * 4, 0);
  // Add one non-convertible thunk to the body.
  body_thunks.push_back(CreateAllGatherThunk(alloc1, alloc2));
  body_thunks.push_back(CreateGemmThunk(alloc3));

  // Create a while thunk
  BufferAllocation alloc4(0, 1024, 0);
  thunks.push_back(CreateWhileThunk(std::move(condition_thunks),
                                    std::move(body_thunks), alloc4));

  DebugOptions debug_options = xla::GetDebugOptionsFromFlags();

  debug_options.set_xla_gpu_graph_min_graph_size(1);
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::WHILE);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::CUBLAS);
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);
  debug_options.set_xla_gpu_graph_min_graph_size(1);
  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  FakeErrorAllocator allocator;
  ASSERT_EQ(thunks.size(), 1);

  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));

  // While thunk is not converted itself, because it has a non-convertible thunk
  // in its body, but the body is partially converted. Expected transformation:
  // (While({Copy}, {AllGatherStart,Gemm})) ->
  // ((While({Copy}, {AllGatherStart, CommandBuffer(Gemm))})))
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kWhile));

  // Check the content of the while thunk
  auto* while_thunk = dynamic_cast<const WhileThunk*>(thunks[0].get());
  ASSERT_NE(while_thunk, nullptr);
  const auto& thunks_in_while_thunk_body =
      while_thunk->body_executor().thunks();
  EXPECT_THAT(thunks_in_while_thunk_body,
              ThunkKindsAre(Thunk::kAllGather, Thunk::kCommandBuffer));
  auto* command_buffer_thunk = dynamic_cast<const CommandBufferThunk*>(
      thunks_in_while_thunk_body[1].get());
  ASSERT_NE(command_buffer_thunk, nullptr);
  const auto& thunks_in_command_buffer =
      command_buffer_thunk->thunks()->thunks();
  EXPECT_THAT(thunks_in_command_buffer, ThunkKindsAre(Thunk::kGemm));
}

TEST(CommandBufferConversionPassTest, ConvertAsyncStartDonePair) {
  ThunkSequence thunks;

  // Create AsyncStartThunk with an empty nested sequence.
  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo(), ComputationStreamId(0), ThunkSequence{});
  auto async_execution = start_thunk->async_execution();
  thunks.push_back(std::move(start_thunk));

  // Create AsyncDoneThunk paired with the start thunk.
  thunks.push_back(
      std::make_unique<AsyncDoneThunk>(Thunk::ThunkInfo(), async_execution));

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  DebugOptions debug_options;
  debug_options.clear_xla_gpu_enable_command_buffer();
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::FUSION);

  FakeErrorAllocator allocator;
  CommandBufferConversionPass pass("test");
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(true));
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kCommandBuffer));
}

TEST(CommandBufferConversionPassTest,
     DontConvertAsyncStartDoneIfNonConvertibleThunkInBetween) {
  ThunkSequence thunks;

  // Create AsyncStartThunk with an empty nested sequence.
  auto start_thunk = std::make_unique<AsyncStartThunk>(
      Thunk::ThunkInfo(), ComputationStreamId(0), ThunkSequence{});
  auto async_execution = start_thunk->async_execution();
  thunks.push_back(std::move(start_thunk));

  // Create a non-convertible thunk in between.
  BufferAllocation alloc0(0, 1024, 0);
  thunks.push_back(CreateCopyThunk(alloc0));

  // Create AsyncDoneThunk paired with the start thunk.
  thunks.push_back(
      std::make_unique<AsyncDoneThunk>(Thunk::ThunkInfo(), async_execution));

  se::DeviceDescription device_info = TestGpuDeviceInfo::CudaOrRocmDeviceInfo();
  DebugOptions debug_options;
  debug_options.add_xla_gpu_enable_command_buffer(DebugOptions::COLLECTIVES);

  FakeErrorAllocator allocator;
  CommandBufferConversionPass pass("test");
  // Expected no transformation, because there is a non-convertible thunk in
  // between the async start and done.
  ASSERT_THAT(pass.Run(&thunks, debug_options, /*hlo_module=*/nullptr,
                       device_info, allocator),
              IsOkAndHolds(false));
  EXPECT_THAT(thunks, ThunkKindsAre(Thunk::kAsyncStart, Thunk::kCopy,
                                    Thunk::kAsyncDone));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
