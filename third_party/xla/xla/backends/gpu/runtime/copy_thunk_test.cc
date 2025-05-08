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

#include "xla/backends/gpu/runtime/copy_thunk.h"

#include <cstdint>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla::gpu {
namespace {

constexpr ExecutionStreamId kExecutionStreamId{123};
constexpr absl::string_view kProfileAnnotation = "profile_annotation";

Thunk::ThunkInfo SampleThunkInfo() {
  Thunk::ThunkInfo thunk_info;
  thunk_info.execution_stream_id = kExecutionStreamId;
  thunk_info.profile_annotation = kProfileAnnotation;
  return thunk_info;
}

void verify_thunk_proto(const ThunkProto& proto) {
  ASSERT_TRUE(proto.has_thunk_info());
  EXPECT_EQ(proto.thunk_info().execution_stream_id(), kExecutionStreamId);
  EXPECT_EQ(proto.thunk_info().profile_annotation(), kProfileAnnotation);
}

void verify_copy_thunk_proto(const CopyThunkProto& copy_thunk_proto,
                             const BufferAllocation::Slice& src_slice,
                             const BufferAllocation::Slice& dst_slice,
                             const uint64_t mem_size) {
  ASSERT_TRUE(copy_thunk_proto.has_source_buffer());
  EXPECT_EQ(copy_thunk_proto.source_buffer().offset(), src_slice.offset());
  EXPECT_EQ(copy_thunk_proto.source_buffer().size(), src_slice.size());
  EXPECT_EQ(copy_thunk_proto.source_buffer().buffer_allocation_index(),
            src_slice.index());

  ASSERT_TRUE(copy_thunk_proto.has_destination_buffer());
  EXPECT_EQ(copy_thunk_proto.destination_buffer().offset(), dst_slice.offset());
  EXPECT_EQ(copy_thunk_proto.destination_buffer().size(), dst_slice.size());
  EXPECT_EQ(copy_thunk_proto.destination_buffer().buffer_allocation_index(),
            dst_slice.index());

  EXPECT_EQ(copy_thunk_proto.mem_size(), mem_size);
}

TEST(CopyThunkTest, ToProto) {
  const uint64_t mem_size = 256;
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice = BufferAllocation::Slice(&alloc0, 128, 384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, 0, 256);

  CopyThunk thunk(SampleThunkInfo(), src_slice, dst_slice, mem_size);

  ThunkProto proto;
  TF_ASSERT_OK(thunk.ToProto(&proto));
  ASSERT_TRUE(proto.has_copy_thunk());
  verify_thunk_proto(proto);
  verify_copy_thunk_proto(proto.copy_thunk(), src_slice, dst_slice, mem_size);
}

TEST(DeviceToHostCopyThunkProtoTest, ToProto) {
  const uint64_t mem_size = 256;
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice = BufferAllocation::Slice(&alloc0, 128, 384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, 0, 256);

  DeviceToHostCopyThunk thunk(SampleThunkInfo(), src_slice, dst_slice, mem_size,
                              /*events=*/nullptr,
                              /*instr=*/nullptr);
  ThunkProto proto;
  TF_ASSERT_OK(thunk.ToProto(&proto));
  verify_thunk_proto(proto);
  ASSERT_TRUE(proto.has_device_to_host_copy_thunk());
  ASSERT_TRUE(proto.device_to_host_copy_thunk().has_copy_thunk());
  verify_copy_thunk_proto(proto.device_to_host_copy_thunk().copy_thunk(),
                          src_slice, dst_slice, mem_size);
}

TEST(HostToDeviceCopyThunkProtoTest, ToProto) {
  const uint64_t mem_size = 256;
  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice = BufferAllocation::Slice(&alloc0, 128, 384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, 0, 256);

  HostToDeviceCopyThunk thunk(SampleThunkInfo(), src_slice, dst_slice, mem_size,
                              /*events=*/nullptr,
                              /*instr=*/nullptr);
  ThunkProto proto;
  TF_ASSERT_OK(thunk.ToProto(&proto));
  verify_thunk_proto(proto);
  ASSERT_TRUE(proto.has_host_to_device_copy_thunk());
  ASSERT_TRUE(proto.host_to_device_copy_thunk().has_copy_thunk());
  verify_copy_thunk_proto(proto.host_to_device_copy_thunk().copy_thunk(),
                          src_slice, dst_slice, mem_size);
}

}  // namespace
}  // namespace xla::gpu
