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

#include "xla/backends/gpu/runtime/select_k_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_buffer_thunk.h"
#include "xla/backends/gpu/runtime/command_executor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_id.h"
#include "xla/codegen/emitters/kernel_arguments.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/literal_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::tsl::proto_testing::EqualsProto;

absl::StatusOr<se::StreamExecutor*> GpuExecutor() {
  ABSL_ASSIGN_OR_RETURN(std::string canonical_name,
                   PlatformUtil::CanonicalPlatformName("gpu"));
  std::string name = absl::AsciiStrToUpper(canonical_name);
  ABSL_ASSIGN_OR_RETURN(auto* platform, se::PlatformManager::PlatformWithName(name));
  return platform->ExecutorForDevice(0);
}

TEST(SelectKThunkTest, ToProto) {
  auto c1 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{.125f, 0.875f, .5f, .25f, 0.75f}}));
  auto topKInst = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 3}),
                                 ShapeUtil::MakeShape(S32, {1, 3})}),
      {c1.get()}, "__gpu$TopK");

  Thunk::ThunkInfo thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(topKInst.get(), ThunkId{456});

  std::vector<BufferAllocation> buffer_allocations = {
      {/*index=*/0, /*size=*/20, /*color=*/0},
      {/*index=*/1, /*size=*/12, /*color=*/0},
      {/*index=*/2, /*size=*/12, /*color=*/0}};

  BufferAllocation::Slice slice0(&buffer_allocations[0], /*offset=*/0,
                                 /*size=*/20);
  BufferAllocation::Slice slice1(&buffer_allocations[1], /*offset=*/0,
                                 /*size=*/12);
  BufferAllocation::Slice slice2(&buffer_allocations[2], /*offset=*/0,
                                 /*size=*/12);

  emitters::KernelArgument arg0(ShapeUtil::MakeShape(F32, {1, 5}), slice0);
  emitters::KernelArgument arg1(ShapeUtil::MakeShape(F32, {1, 3}), slice1);
  emitters::KernelArgument arg2(ShapeUtil::MakeShape(S32, {1, 3}), slice2);

  emitters::KernelArguments kernel_arguments({arg0, arg1, arg2});

  SelectKThunk thunk(std::move(thunk_info), 1, 5, 3, F32, kernel_arguments);

  ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info { profile_annotation: "custom-call" thunk_id: 456 }
                select_k_thunk {
                  args { buffer_allocation_index: 0 size: 20 }
                  args { buffer_allocation_index: 1 size: 12 }
                  args { buffer_allocation_index: 2 size: 12 }
                  batch_size: 1
                  num_elements: 5
                  k: 3
                  dtype: F32
                }
              )pb"));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<SelectKThunk> deserialized,
      SelectKThunk::FromProto(thunk.thunk_info(), proto.select_k_thunk(),
                              buffer_allocations));
  EXPECT_THAT(deserialized->ToProto(), IsOkAndHolds(EqualsProto(proto)));
}

TEST(SelectKThunkTest, BufferUses) {
  auto c1 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{.125f, 0.875f, .5f, .25f, 0.75f}}));
  auto topKInst = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 3}),
                                 ShapeUtil::MakeShape(S32, {1, 3})}),
      {c1.get()}, "__gpu$TopK");

  Thunk::ThunkInfo thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(topKInst.get(), ThunkId{456});

  std::vector<BufferAllocation> buffer_allocations = {
      {/*index=*/0, /*size=*/20, /*color=*/0},
      {/*index=*/1, /*size=*/12, /*color=*/0},
      {/*index=*/2, /*size=*/12, /*color=*/0}};

  BufferAllocation::Slice slice0(&buffer_allocations[0], /*offset=*/0,
                                 /*size=*/20);
  BufferAllocation::Slice slice1(&buffer_allocations[1], /*offset=*/0,
                                 /*size=*/12);
  BufferAllocation::Slice slice2(&buffer_allocations[2], /*offset=*/0,
                                 /*size=*/12);

  Shape shape0 = ShapeUtil::MakeShape(F32, {1, 5});
  Shape shape1 = ShapeUtil::MakeShape(F32, {1, 3});
  Shape shape2 = ShapeUtil::MakeShape(S32, {1, 3});

  emitters::KernelArgument arg0(shape0, slice0);
  emitters::KernelArgument arg1(shape1, slice1);
  emitters::KernelArgument arg2(shape2, slice2);
  emitters::KernelArguments kernel_arguments({arg0, arg1, arg2});

  SelectKThunk thunk(std::move(thunk_info), /*batch_size=*/1,
                     /*num_elements=*/5, /*k=*/3, /*dtype=*/F32,
                     kernel_arguments);

  EXPECT_THAT(thunk.buffer_uses(),
              ElementsAre(BufferUse::Read(slice0, shape0),
                          BufferUse::Write(slice1, shape1),
                          BufferUse::Write(slice2, shape2)));
}

TEST(SelectKThunkTest, BufferUsesWithScratch) {
  auto c1 = HloInstruction::CreateConstant(
      LiteralUtil::CreateR2<float>({{.125f, 0.875f, .5f, .25f, 0.75f}}));
  auto topKInst = HloInstruction::CreateCustomCall(
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {1, 3}),
                                 ShapeUtil::MakeShape(S32, {1, 3}),
                                 ShapeUtil::MakeShape(U8, {1024})}),
      {c1.get()}, "__gpu$TopK");

  Thunk::ThunkInfo thunk_info =
      Thunk::ThunkInfo::WithProfileAnnotation(topKInst.get(), ThunkId{456});

  std::vector<BufferAllocation> buffer_allocations = {
      {/*index=*/0, /*size=*/20, /*color=*/0},
      {/*index=*/1, /*size=*/12, /*color=*/0},
      {/*index=*/2, /*size=*/12, /*color=*/0},
      {/*index=*/3, /*size=*/1024, /*color=*/0}};

  BufferAllocation::Slice slice0(&buffer_allocations[0], /*offset=*/0,
                                 /*size=*/20);
  BufferAllocation::Slice slice1(&buffer_allocations[1], /*offset=*/0,
                                 /*size=*/12);
  BufferAllocation::Slice slice2(&buffer_allocations[2], /*offset=*/0,
                                 /*size=*/12);
  BufferAllocation::Slice slice3(&buffer_allocations[3], /*offset=*/0,
                                 /*size=*/1024);

  Shape shape0 = ShapeUtil::MakeShape(F32, {1, 5});
  Shape shape1 = ShapeUtil::MakeShape(F32, {1, 3});
  Shape shape2 = ShapeUtil::MakeShape(S32, {1, 3});
  Shape shape3 = ShapeUtil::MakeShape(U8, {1024});

  emitters::KernelArgument arg0(shape0, slice0);
  emitters::KernelArgument arg1(shape1, slice1);
  emitters::KernelArgument arg2(shape2, slice2);
  emitters::KernelArgument arg3(shape3, slice3);
  emitters::KernelArguments kernel_arguments({arg0, arg1, arg2, arg3});

  SelectKThunk thunk(std::move(thunk_info), /*batch_size=*/1,
                     /*num_elements=*/5, /*k=*/3, /*dtype=*/F32,
                     kernel_arguments);

  EXPECT_THAT(thunk.buffer_uses(),
              ElementsAre(BufferUse::Read(slice0, shape0),
                          BufferUse::Write(slice1, shape1),
                          BufferUse::Write(slice2, shape2),
                          BufferUse::Write(slice3, shape3)));
}

TEST(SelectKThunkTest, CommandBuffer) {
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * stream_executor, GpuExecutor());
  ASSERT_OK_AND_ASSIGN(auto stream, stream_executor->CreateStream());

  if (!stream_executor->GetDeviceDescription()
           .cuda_compute_capability()
           .IsAtLeastAmpere()) {
    GTEST_SKIP() << "Requires at least an Ampere GPU.";
  }

  constexpr int kBatchSize = 1;
  constexpr int kNumElements = 4096;
  constexpr int kTopK = 32;
  constexpr size_t kScratchSize = 32 * 1024 * 1024;

  BufferAllocation alloc_input(/*index=*/0,
                               kBatchSize * kNumElements * sizeof(float),
                               /*color=*/0);
  BufferAllocation alloc_output_val(
      /*index=*/1, kBatchSize * kTopK * sizeof(float), /*color=*/0);
  BufferAllocation alloc_output_idx(
      /*index=*/2, kBatchSize * kTopK * sizeof(int32_t), /*color=*/0);
  BufferAllocation alloc_scratch(/*index=*/3, kScratchSize, /*color=*/0);

  BufferAllocation::Slice slice_input(&alloc_input, 0, alloc_input.size());
  BufferAllocation::Slice slice_output_val(&alloc_output_val, 0,
                                           alloc_output_val.size());
  BufferAllocation::Slice slice_output_idx(&alloc_output_idx, 0,
                                           alloc_output_idx.size());
  BufferAllocation::Slice slice_scratch(&alloc_scratch, 0,
                                        alloc_scratch.size());

  emitters::KernelArgument arg_input(
      ShapeUtil::MakeShape(F32, {kBatchSize, kNumElements}), slice_input);
  emitters::KernelArgument arg_output_val(
      ShapeUtil::MakeShape(F32, {kBatchSize, kTopK}), slice_output_val);
  emitters::KernelArgument arg_output_idx(
      ShapeUtil::MakeShape(S32, {kBatchSize, kTopK}), slice_output_idx);
  emitters::KernelArgument arg_scratch(ShapeUtil::MakeShape(U8, {kScratchSize}),
                                       slice_scratch);

  emitters::KernelArguments kernel_arguments(
      {arg_input, arg_output_val, arg_output_idx, arg_scratch});

  auto select_k_thunk = std::make_unique<SelectKThunk>(
      Thunk::ThunkInfo(), kBatchSize, kNumElements, kTopK, F32,
      kernel_arguments);

  CommandSequence commands;
  commands.Append(select_k_thunk.get());
  ASSERT_OK_AND_ASSIGN(CommandExecutor executor,
                       CommandExecutor::Create(
                           std::move(commands),
                           CommandExecutor::SynchronizationMode::kSerialize));

  ThunkSequence thunk_sequence;
  thunk_sequence.push_back(std::move(select_k_thunk));
  auto sequential_thunk = std::make_unique<SequentialThunk>(
      Thunk::ThunkInfo(), std::move(thunk_sequence));
  CommandBufferThunk thunk(std::move(executor), Thunk::ThunkInfo(),
                           std::move(sequential_thunk));

  std::vector<se::DeviceAddressBase> operands;
  operands.reserve(4);

  std::vector<float> host_in(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    host_in[i] = static_cast<float>(i);
  }
  se::DeviceAddress<float> input =
      stream_executor->AllocateArray<float>(kBatchSize * kNumElements);
  ASSERT_OK(
      stream->Memcpy(&input, host_in.data(), sizeof(float) * host_in.size()));

  se::DeviceAddress<float> output_val =
      stream_executor->AllocateArray<float>(kBatchSize * kTopK);
  ASSERT_OK(stream->MemZero(&output_val, output_val.size()));

  se::DeviceAddress<int32_t> output_idx =
      stream_executor->AllocateArray<int32_t>(kBatchSize * kTopK);
  ASSERT_OK(stream->MemZero(&output_idx, output_idx.size()));

  se::DeviceAddressBase scratch = stream_executor->Allocate(kScratchSize);

  operands.push_back(input);
  operands.push_back(output_val);
  operands.push_back(output_idx);
  operands.push_back(scratch);

  ServiceExecutableRunOptions run_options;
  stream_executor::StreamExecutorAddressAllocator allocator(stream_executor);
  BufferAllocations allocations(operands, 0, &allocator);

  Thunk::ExecuteParams params = Thunk::ExecuteParams::Create(
      run_options, allocations, stream.get(), stream.get(), nullptr, nullptr,
      nullptr, /*additional_compute_streams=*/{},
      /*execution_scoped_state=*/nullptr,
      /*persistent_alloc_indices=*/absl::Span<const BufferAllocation::Index>());

  Thunk::ExecutableSource source = {/*text=*/"", /*binary=*/{}};
  Thunk::InitializeParams initialize_params;
  initialize_params.executor = stream_executor;
  initialize_params.src = source;
  initialize_params.buffer_allocations = &allocations;
  initialize_params.stream = stream.get();
  initialize_params.command_buffer_trace_stream = stream.get();
  initialize_params.persistent_alloc_indices =
      absl::Span<const BufferAllocation::Index>();
  ASSERT_OK(thunk.Initialize(initialize_params));

  // First execution: traces CUDA graph with SelectKThunk and executes it.
  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  std::vector<float> dst_val(kTopK, 0.0f);
  std::vector<int32_t> dst_idx(kTopK, -1);
  ASSERT_OK(stream->Memcpy(dst_val.data(), output_val, sizeof(float) * kTopK));
  ASSERT_OK(
      stream->Memcpy(dst_idx.data(), output_idx, sizeof(int32_t) * kTopK));

  std::vector<float> expected_val(kTopK);
  std::vector<int32_t> expected_idx(kTopK);
  for (int i = 0; i < kTopK; ++i) {
    expected_val[i] = static_cast<float>(kNumElements - 1 - i);
    expected_idx[i] = kNumElements - 1 - i;
  }
  EXPECT_EQ(dst_val, expected_val);
  EXPECT_EQ(dst_idx, expected_idx);

  // Second execution (replay): update input data in place.
  std::vector<float> host_in2(kNumElements);
  for (int i = 0; i < kNumElements; ++i) {
    host_in2[i] = static_cast<float>(kNumElements - i);
  }
  ASSERT_OK(
      stream->Memcpy(&input, host_in2.data(), sizeof(float) * host_in2.size()));

  ASSERT_OK(thunk.ExecuteOnStream(params));
  ASSERT_OK(stream->BlockHostUntilDone());

  ASSERT_OK(stream->Memcpy(dst_val.data(), output_val, sizeof(float) * kTopK));
  ASSERT_OK(
      stream->Memcpy(dst_idx.data(), output_idx, sizeof(int32_t) * kTopK));

  std::vector<float> expected_val2(kTopK);
  std::vector<int32_t> expected_idx2(kTopK);
  for (int i = 0; i < kTopK; ++i) {
    expected_val2[i] = static_cast<float>(kNumElements - i);
    expected_idx2[i] = i;
  }
  EXPECT_EQ(dst_val, expected_val2);
  EXPECT_EQ(dst_idx, expected_idx2);
}

}  // namespace
}  // namespace xla::gpu
