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

#include "xla/backends/gpu/runtime/select_k_thunk_proto_deserialization.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/parse_text_proto.h"

namespace xla::gpu {
namespace {

using ::tsl::proto_testing::ParseTextProtoOrDie;

TEST(SelectKThunkProtoDeserializationTest, SelectKThunkBackwardCompatibility) {
  SelectKThunkProto proto = ParseTextProtoOrDie<SelectKThunkProto>(R"pb(
    args { buffer_allocation_index: 0 offset: 0 size: 16384 }
    args { buffer_allocation_index: 1 offset: 0 size: 128 }
    args { buffer_allocation_index: 2 offset: 0 size: 128 }
    args { buffer_allocation_index: 3 offset: 0 size: 33554432 }
    batch_size: 1
    num_elements: 4096
    k: 32
    dtype: F32
  )pb");

  std::vector<BufferAllocation> buffer_allocations = {
      BufferAllocation(/*index=*/0, /*size=*/16384, /*color=*/0),
      BufferAllocation(/*index=*/1, /*size=*/128, /*color=*/0),
      BufferAllocation(/*index=*/2, /*size=*/128, /*color=*/0),
      BufferAllocation(/*index=*/3, /*size=*/33554432, /*color=*/0),
  };

  Thunk::ThunkInfo thunk_info;
  thunk_info.profile_annotation = "select_k";
  thunk_info.thunk_id = 7;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Thunk> deserialized,
      DeserializeSelectKThunkProto(thunk_info, proto, buffer_allocations,
                                   "CUDA", se::GpuComputeCapability()));
  EXPECT_EQ(deserialized->kind(), Thunk::kCustomCall);
  const auto* custom_call_thunk =
      dynamic_cast<const CustomCallThunk*>(deserialized.get());
  ASSERT_NE(custom_call_thunk, nullptr);
  EXPECT_EQ(custom_call_thunk->target_name(), kTopKCustomCallTarget);
  EXPECT_EQ(custom_call_thunk->operands().size(), 1);
  EXPECT_EQ(custom_call_thunk->results().size(), 3);
}

}  // namespace
}  // namespace xla::gpu
