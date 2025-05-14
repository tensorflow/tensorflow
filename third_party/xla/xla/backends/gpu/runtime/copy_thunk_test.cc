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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;

TEST(CopyThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice =
      BufferAllocation::Slice(&alloc0, /*offset=*/128, /*size=*/384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, /*offset=*/0, /*size=*/256);

  CopyThunk thunk(thunk_info, src_slice, dst_slice, /*mem_size=*/256);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info {
                  profile_annotation: "profile_annotation"
                  execution_stream_id: 123
                }
                copy_thunk {
                  source_buffer { offset: 128 size: 384 }
                  destination_buffer { size: 256 buffer_allocation_index: 1 }
                  mem_size: 256
                }
              )pb"));
}

TEST(DeviceToHostCopyThunkProtoTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice =
      BufferAllocation::Slice(&alloc0, /*offset=*/128, /*size=*/384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, /*offset=*/0, /*size=*/256);

  DeviceToHostCopyThunk thunk(thunk_info, src_slice, dst_slice,
                              /*mem_size=*/256,
                              /*events=*/nullptr,
                              /*instr=*/nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info {
                  profile_annotation: "profile_annotation"
                  execution_stream_id: 123
                }
                device_to_host_copy_thunk {
                  copy_thunk {
                    source_buffer { offset: 128 size: 384 }
                    destination_buffer { size: 256 buffer_allocation_index: 1 }
                    mem_size: 256
                  }
                }
              )pb"));
}

TEST(HostToDeviceCopyThunkProtoTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice =
      BufferAllocation::Slice(&alloc0, /*offset=*/128, /*size=*/384);
  auto dst_slice = BufferAllocation::Slice(&alloc1, /*offset=*/0, /*size=*/256);

  HostToDeviceCopyThunk thunk(thunk_info, src_slice, dst_slice,
                              /*mem_size=*/256,
                              /*events=*/nullptr,
                              /*instr=*/nullptr);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info {
                  profile_annotation: "profile_annotation"
                  execution_stream_id: 123
                }
                host_to_device_copy_thunk {
                  copy_thunk {
                    source_buffer { offset: 128 size: 384 }
                    destination_buffer { size: 256 buffer_allocation_index: 1 }
                    mem_size: 256
                  }
                }
              )pb"));
}

}  // namespace
}  // namespace xla::gpu
