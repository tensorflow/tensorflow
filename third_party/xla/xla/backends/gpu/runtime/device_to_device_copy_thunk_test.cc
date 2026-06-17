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

#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/ascii.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/command_state.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/platform_util.h"
#include "xla/service/service_executable_run_options.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

static se::StreamExecutor* GpuExecutor() {
  auto name =
      absl::AsciiStrToUpper(PlatformUtil::CanonicalPlatformName("gpu").value());
  auto* platform = se::PlatformManager::PlatformWithName(name).value();
  return platform->ExecutorForDevice(0).value();
}

//===----------------------------------------------------------------------===//
// Proto round-trip tests
//===----------------------------------------------------------------------===//

TEST(DeviceToDeviceCopyThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice =
      BufferAllocation::Slice(&alloc0, /*offset=*/128, /*size=*/256);
  auto dst_slice = BufferAllocation::Slice(&alloc1, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(S32, {64});

  DeviceToDeviceCopyThunk thunk(thunk_info, {src_slice, shape},
                                {dst_slice, shape}, 256);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info { profile_annotation: "profile_annotation" }
                device_to_device_copy_thunk {
                  copy_thunk {
                    source_buffer {
                      slice { offset: 128 size: 256 }
                      shape {
                        dimensions: 64
                        element_type: S32
                        is_dynamic_dimension: false
                        layout {
                          minor_to_major: 0
                          tail_padding_alignment_in_elements: 1
                        }
                      }
                    }
                    destination_buffer {
                      slice { size: 256 buffer_allocation_index: 1 }
                      shape {
                        dimensions: 64
                        element_type: S32
                        is_dynamic_dimension: false
                        layout {
                          minor_to_major: 0
                          tail_padding_alignment_in_elements: 1
                        }
                      }
                    }
                    mem_size: 256
                  }
                }
              )pb"));
}

TEST(DeviceToDeviceCopyThunkTest, FromProto) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info { profile_annotation: "profile_annotation" }
        device_to_device_copy_thunk {
          copy_thunk {
            source_buffer {
              slice { offset: 128 size: 256 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            destination_buffer {
              slice { size: 256 buffer_allocation_index: 1 }
              shape {
                dimensions: 64
                element_type: S32
                is_dynamic_dimension: false
                layout {
                  minor_to_major: 0
                  tail_padding_alignment_in_elements: 1
                }
              }
            }
            mem_size: 256
          }
        }
      )pb");

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DeviceToDeviceCopyThunk> thunk,
      DeviceToDeviceCopyThunk::FromProto(
          thunk_info, proto.device_to_device_copy_thunk(), buffer_allocations));

  Shape shape = ShapeUtil::MakeShape(S32, {64});
  EXPECT_EQ(*thunk.get(),
            DeviceToDeviceCopyThunk(
                thunk_info,
                {BufferAllocation::Slice(&buffer_allocations[0],
                                         /*offset=*/128, /*size=*/256),
                 shape},
                {BufferAllocation::Slice(&buffer_allocations[1], /*offset=*/0,
                                         /*size=*/256),
                 shape},
                256));
}

//===----------------------------------------------------------------------===//
// Command buffer tests (Record)
//===----------------------------------------------------------------------===//

// Records a D2D copy into a primary command buffer and verifies the data
// is copied correctly after submission.
TEST(DeviceToDeviceCopyThunkTest, RecordCommandBuffer) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> dst = executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&src, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&dst, byte_length));

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, byte_length);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, byte_length);

  DeviceToDeviceCopyThunk thunk(Thunk::ThunkInfo(), {slice_src, shape},
                                {slice_dst, shape}, byte_length);

  se::StreamExecutorAddressAllocator allocator(executor);
  BufferAllocations buffer_allocations({src, dst}, 0, &allocator);

  ServiceExecutableRunOptions run_options;
  Thunk::ExecuteParams execute_params = Thunk::ExecuteParams::Create(
      run_options, buffer_allocations, stream.get(),
      /*command_buffer_trace_stream=*/nullptr,
      /*collective_params=*/nullptr,
      /*collective_cliques=*/nullptr,
      /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(execute_params, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  std::vector<int32_t> result(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result.data(), dst, byte_length));
  EXPECT_EQ(result, std::vector<int32_t>(length, 42));
}

// Records a D2D copy into a command buffer, then updates the command buffer to
// copy to a different destination and verifies the update takes effect.
TEST(DeviceToDeviceCopyThunkTest, RecordCommandBufferUpdate) {
  se::StreamExecutor* executor = GpuExecutor();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());

  int64_t length = 4;
  int64_t byte_length = sizeof(int32_t) * length;
  Shape shape = ShapeUtil::MakeShape(S32, {length});

  se::DeviceAddress<int32_t> src = executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> dst_first =
      executor->AllocateArray<int32_t>(length, 0);
  se::DeviceAddress<int32_t> dst_second =
      executor->AllocateArray<int32_t>(length, 0);

  TF_ASSERT_OK(stream->Memset32(&src, 42, byte_length));
  TF_ASSERT_OK(stream->MemZero(&dst_first, byte_length));
  TF_ASSERT_OK(stream->MemZero(&dst_second, byte_length));

  BufferAllocation alloc_src(/*index=*/0, byte_length, /*color=*/0);
  BufferAllocation alloc_dst(/*index=*/1, byte_length, /*color=*/0);
  BufferAllocation::Slice slice_src(&alloc_src, 0, byte_length);
  BufferAllocation::Slice slice_dst(&alloc_dst, 0, byte_length);

  DeviceToDeviceCopyThunk thunk(Thunk::ThunkInfo(), {slice_src, shape},
                                {slice_dst, shape}, byte_length);

  se::StreamExecutorAddressAllocator allocator(executor);
  ServiceExecutableRunOptions run_options;

  // First recording: src -> dst_first.
  BufferAllocations allocs_first({src, dst_first}, 0, &allocator);
  Thunk::ExecuteParams params_first =
      Thunk::ExecuteParams::Create(run_options, allocs_first, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  CommandStateManager state;
  Command::RecordParams record_params = {state};

  TF_ASSERT_OK_AND_ASSIGN(
      auto command_buffer,
      executor->CreateCommandBuffer(se::CommandBuffer::Mode::kPrimary));
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* cmd,
      thunk.Record(params_first, record_params,
                   Command::RecordCreate{/*dependencies=*/{}},
                   command_buffer.get()));
  ASSERT_NE(cmd, nullptr);
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  std::vector<int32_t> result_first(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result_first.data(), dst_first, byte_length));
  EXPECT_EQ(result_first, std::vector<int32_t>(length, 42));

  // Update recording: src -> dst_second using the same command node.
  BufferAllocations allocs_second({src, dst_second}, 0, &allocator);
  Thunk::ExecuteParams params_second =
      Thunk::ExecuteParams::Create(run_options, allocs_second, stream.get(),
                                   /*command_buffer_trace_stream=*/nullptr,
                                   /*collective_params=*/nullptr,
                                   /*collective_cliques=*/nullptr,
                                   /*collective_memory=*/nullptr);

  TF_ASSERT_OK(command_buffer->Update());
  TF_ASSERT_OK_AND_ASSIGN(
      const se::CommandBuffer::Command* updated_cmd,
      thunk.Record(params_second, record_params, Command::RecordUpdate{cmd},
                   command_buffer.get()));
  EXPECT_EQ(updated_cmd, cmd);  // same command node is reused
  TF_ASSERT_OK(command_buffer->Finalize());
  TF_ASSERT_OK(command_buffer->Submit(stream.get()));

  std::vector<int32_t> result_second(length, 0);
  TF_ASSERT_OK(stream->Memcpy(result_second.data(), dst_second, byte_length));
  EXPECT_EQ(result_second, std::vector<int32_t>(length, 42));

  // dst_first should still hold the value from the first Submit (not zeroed or
  // overwritten by the update), confirming the copy was redirected.
  std::vector<int32_t> result_first_after(length, 0);
  TF_ASSERT_OK(
      stream->Memcpy(result_first_after.data(), dst_first, byte_length));
  EXPECT_EQ(result_first_after, std::vector<int32_t>(length, 42));
}

}  // namespace
}  // namespace xla::gpu
