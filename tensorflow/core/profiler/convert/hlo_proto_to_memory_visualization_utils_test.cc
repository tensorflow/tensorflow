/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/convert/hlo_proto_to_memory_visualization_utils.h"

#include <string>

#include <gmock/gmock.h>
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/profiler/protobuf/memory_viewer_preprocess.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"

namespace tensorflow {
namespace profiler {
namespace {

// 1 buffer allocation of 1MB
// 2 logical buffers, each is 0.5MB
static constexpr char kHLOBase[] = R"pb(
  hlo_module {
    name: "test_module"
    entry_computation_name: "test_computation"
    computations {
      name: "test_computation"
      instructions {
        name: "fusion.1"
        id: 0
        shape { tuple_shapes { element_type: U64 } }
      }
      instructions {
        name: "fusion.2"
        id: 1
        shape { tuple_shapes { element_type: U64 } }
      }
    }
  }
  buffer_assignment {
    buffer_allocations {
      index: 0
      size: 1048576
      color: 0
      assigned { logical_buffer_id: 1 offset: 0 size: 524288 }
      assigned { logical_buffer_id: 2 offset: 524288 size: 524288 }
    }
    logical_buffers {
      id: 1
      size: 524288
      color: 0
      defined_at { instruction_id: 0 shape_index: 0 }
    }
    logical_buffers {
      id: 2
      size: 524288
      color: 0
      defined_at { instruction_id: 1 shape_index: 0 }
    }
    heap_simulator_traces { %s }
  }
)pb";

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWith_1) {
  // Allocate and then share, the memory usage is not doubled.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: FREE buffer_id: 2 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOBase, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  ASSERT_TRUE(
      proto_utils::ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessResult preprocess_result,
      ConvertHloProtoToPreprocessResult(hlo_proto, /*small_buffer_size=*/0));
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.5);
}

TEST(MemoryViewerTest, TestHeapSimulatorTraceShareWith_2) {
  // Allocate, free and then share, the memory usage is not doubled.
  static constexpr char kHeapSimulatorTrace[] = R"pb(
    events { kind: ALLOC buffer_id: 1 }
    events { kind: FREE buffer_id: 1 }
    events { kind: SHARE_WITH buffer_id: 2 share_with_canonical_id: 1 }
    events { kind: FREE buffer_id: 2 }
  )pb";
  std::string hlo_string = absl::StrFormat(kHLOBase, kHeapSimulatorTrace);
  xla::HloProto hlo_proto;
  ASSERT_TRUE(
      proto_utils::ParseTextFormatFromString(hlo_string, &hlo_proto).ok());
  TF_ASSERT_OK_AND_ASSIGN(
      PreprocessResult preprocess_result,
      ConvertHloProtoToPreprocessResult(hlo_proto, /*small_buffer_size=*/0));
  EXPECT_EQ(preprocess_result.peak_heap_mib(), 0.5);
  EXPECT_FALSE(preprocess_result.allocation_timeline().empty());
}

}  // namespace
}  // namespace profiler
}  // namespace tensorflow
