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

#include "xla/backends/gpu/runtime/dynamic_memcpy_thunk.h"

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"
#include "xla/tsl/util/proto/proto_matchers.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::EqualsProto;
using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(DynamicMemcpyThunkTest, ToProto) {
  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  Shape shape = ShapeUtil::MakeShape(F32, {256});

  DynamicMemcpyThunk thunk(thunk_info,
                           /*source_buffer=*/
                           {BufferAllocation::Slice(&buffer_allocations[0],
                                                    /*offset=*/0,
                                                    /*size=*/1024),
                            shape},
                           /*destination_buffer=*/
                           {BufferAllocation::Slice(&buffer_allocations[1],
                                                    /*offset=*/0,
                                                    /*size=*/1024),
                            shape},
                           /*mem_size=*/256,
                           {DynamicMemcpyThunk::Offsets{
                               /*depends_on_loop=*/true,
                               /*src_offsets=*/std::vector<int64_t>{4, 8},
                               /*dst_offsets=*/std::vector<int64_t>{16, 32},
                           }});
  TF_ASSERT_OK_AND_ASSIGN(ThunkProto proto, thunk.ToProto());
  EXPECT_THAT(
      proto, EqualsProto(R"pb(
        thunk_info {
          profile_annotation: "profile_annotation"
          execution_stream_id: 123
        }
        dynamic_memcpy_thunk {
          source_buffer {
            slice { offset: 0 size: 1024 buffer_allocation_index: 0 }
            shape {
              element_type: F32
              dimensions: 256
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          destination_buffer {
            slice { offset: 0 size: 1024 buffer_allocation_index: 1 }
            shape {
              element_type: F32
              dimensions: 256
              layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
              is_dynamic_dimension: false
            }
          }
          mem_size: 256
          offsets {
            depends_on_loop: true
            src_offsets: 4
            src_offsets: 8
            dst_offsets: 16
            dst_offsets: 32
          }
        }
      )pb"));
}

TEST(DynamicMemcpyThunkTest, FromProto) {
  Shape shape = ShapeUtil::MakeShape(F32, {256});
  auto dynamic_memcpy_thunk_proto = ParseTextProtoOrDie<
      DynamicMemcpyThunkProto>(
      R"pb(
        source_buffer {
          slice { offset: 0 size: 1024 buffer_allocation_index: 0 }
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        destination_buffer {
          slice { offset: 0 size: 1024 buffer_allocation_index: 1 }
          shape {
            element_type: F32
            dimensions: 256
            layout { minor_to_major: 0 tail_padding_alignment_in_elements: 1 }
            is_dynamic_dimension: false
          }
        }
        mem_size: 256
        offsets {
          depends_on_loop: true
          src_offsets: 4
          src_offsets: 8
          dst_offsets: 16
          dst_offsets: 32
        }
      )pb");

  Thunk::ThunkInfo thunk_info{};
  thunk_info.profile_annotation = "profile_annotation";
  thunk_info.execution_stream_id = 123;

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/1024, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/1024, /*color=*/0)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<DynamicMemcpyThunk> thunk,
      DynamicMemcpyThunk::FromProto(thunk_info, dynamic_memcpy_thunk_proto,
                                    buffer_allocations));

  DynamicMemcpyThunk::Offsets reference_offsets{
      /*depends_on_loop=*/true,
      /*src_offsets=*/std::vector<int64_t>{4, 8},
      /*dst_offsets=*/std::vector<int64_t>{16, 32}};
  EXPECT_EQ(thunk->offsets(), reference_offsets);
  EXPECT_EQ(thunk->source(),
            (ShapedSlice{BufferAllocation::Slice(&buffer_allocations[0],
                                                 /*offset=*/0,
                                                 /*size=*/1024),
                         shape}));
  EXPECT_EQ(thunk->destination(),
            (ShapedSlice{BufferAllocation::Slice(&buffer_allocations[1],
                                                 /*offset=*/0,
                                                 /*size=*/1024),
                         shape}));
  EXPECT_EQ(thunk->mem_size(), 256);
}

}  // namespace
}  // namespace xla::gpu
