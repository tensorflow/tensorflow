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

#include "xla/backends/gpu/runtime/host_to_device_copy_thunk.h"

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(HostToDeviceCopyThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  BufferAllocation alloc0(/*index=*/0, /*size=*/1024, /*color=*/0);
  BufferAllocation alloc1(/*index=*/1, /*size=*/1024, /*color=*/0);
  auto src_slice =
      BufferAllocation::Slice(&alloc0, /*offset=*/128, /*size=*/256);
  auto dst_slice = BufferAllocation::Slice(&alloc1, /*offset=*/0, /*size=*/256);
  Shape shape = ShapeUtil::MakeShape(S32, {64});

  HostToDeviceCopyThunk thunk(thunk_info, {src_slice, shape},
                              {dst_slice, shape},
                              /*mem_size=*/256,
                              /*events=*/nullptr,
                              /*instr_id=*/-1);
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(proto, EqualsProto(R"pb(
                thunk_info {
                  profile_annotation: "profile_annotation"
                  execution_stream_id: 123
                }
                host_to_device_copy_thunk {
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

TEST(HostToDeviceCopyThunkTest, FromProto) {
  ThunkProto proto = ParseTextProtoOrDie<ThunkProto>(
      R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        host_to_device_copy_thunk {
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
  thunk_info.execution_stream_id = 123;
  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};
  Shape shape = ShapeUtil::MakeShape(S32, {64});

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<HostToDeviceCopyThunk> thunk,
      HostToDeviceCopyThunk::FromProto(
          thunk_info, proto.host_to_device_copy_thunk(), buffer_allocations));

  EXPECT_EQ(*thunk.get(),
            HostToDeviceCopyThunk(
                thunk_info,
                {BufferAllocation::Slice(&buffer_allocations[0],
                                         /*offset=*/128, /*size=*/256),
                 shape},
                {BufferAllocation::Slice(&buffer_allocations[1], /*offset=*/0,
                                         /*size=*/256),
                 shape},
                /*mem_size=*/256,
                /*events=*/nullptr,
                /*instr_id=*/-1));
}

}  // namespace
}  // namespace xla::gpu
