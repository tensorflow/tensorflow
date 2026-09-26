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

#include "xla/backends/gpu/runtime/execution_stream_id.h"

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"

namespace xla::gpu {
namespace {

TEST(ComputationStreamIdTest, AbslStringify) {
  ComputationStreamId id(42);
  EXPECT_EQ(absl::StrCat(id), "computation_stream[42]");
}

TEST(CommunicationStreamIdTest, AbslStringify) {
  CommunicationStreamId id(7);
  EXPECT_EQ(absl::StrCat(id), "communication_stream[7]");
}

TEST(ExecutionStreamIdTest, AbslStringifyComputation) {
  ExecutionStreamId id(ComputationStreamId(42));
  EXPECT_EQ(absl::StrCat(id), "computation_stream[42]");
}

TEST(ExecutionStreamIdTest, AbslStringifyCommunication) {
  ExecutionStreamId id(CommunicationStreamId(7));
  EXPECT_EQ(absl::StrCat(id), "communication_stream[7]");
}

TEST(ExecutionStreamIdTest, Accessors) {
  ExecutionStreamId compute(ComputationStreamId(3));
  EXPECT_TRUE(compute.is_computation());
  EXPECT_FALSE(compute.is_communication());
  EXPECT_EQ(compute.computation_id(), ComputationStreamId(3));

  ExecutionStreamId comm(CommunicationStreamId(5));
  EXPECT_TRUE(comm.is_communication());
  EXPECT_FALSE(comm.is_computation());
  EXPECT_EQ(comm.communication_id(), CommunicationStreamId(5));
}

TEST(ExecutionStreamIdTest, Equality) {
  EXPECT_EQ(ExecutionStreamId(ComputationStreamId(1)),
            ExecutionStreamId(ComputationStreamId(1)));
  EXPECT_NE(ExecutionStreamId(ComputationStreamId(1)),
            ExecutionStreamId(ComputationStreamId(2)));
  EXPECT_NE(ExecutionStreamId(ComputationStreamId(1)),
            ExecutionStreamId(CommunicationStreamId(1)));
}

// ---- MemcpyStreamId tests ------------------------------------------------

TEST(MemcpyStreamIdTest, Constants) {
  EXPECT_EQ(kMemcpyD2HStreamId.value(), 0u);
  EXPECT_EQ(kMemcpyH2DStreamId.value(), 1u);
  EXPECT_NE(kMemcpyD2HStreamId, kMemcpyH2DStreamId);
}

TEST(MemcpyStreamIdTest, AbslStringifyD2H) {
  EXPECT_EQ(absl::StrCat(kMemcpyD2HStreamId), "memcpy_d2h_stream");
}

TEST(MemcpyStreamIdTest, AbslStringifyH2D) {
  EXPECT_EQ(absl::StrCat(kMemcpyH2DStreamId), "memcpy_h2d_stream");
}

TEST(ExecutionStreamIdTest, MemcpyAccessors) {
  ExecutionStreamId d2h(kMemcpyD2HStreamId);
  EXPECT_TRUE(d2h.is_memcpy());
  EXPECT_FALSE(d2h.is_computation());
  EXPECT_FALSE(d2h.is_communication());
  EXPECT_EQ(d2h.memcpy_id(), kMemcpyD2HStreamId);

  ExecutionStreamId h2d(kMemcpyH2DStreamId);
  EXPECT_TRUE(h2d.is_memcpy());
  EXPECT_FALSE(h2d.is_computation());
  EXPECT_FALSE(h2d.is_communication());
  EXPECT_EQ(h2d.memcpy_id(), kMemcpyH2DStreamId);
}

TEST(ExecutionStreamIdTest, AbslStringifyMemcpy) {
  EXPECT_EQ(absl::StrCat(ExecutionStreamId(kMemcpyD2HStreamId)),
            "memcpy_d2h_stream");
  EXPECT_EQ(absl::StrCat(ExecutionStreamId(kMemcpyH2DStreamId)),
            "memcpy_h2d_stream");
}

TEST(ExecutionStreamIdTest, EqualityWithMemcpy) {
  EXPECT_EQ(ExecutionStreamId(kMemcpyD2HStreamId),
            ExecutionStreamId(kMemcpyD2HStreamId));
  EXPECT_EQ(ExecutionStreamId(kMemcpyH2DStreamId),
            ExecutionStreamId(kMemcpyH2DStreamId));
  EXPECT_NE(ExecutionStreamId(kMemcpyD2HStreamId),
            ExecutionStreamId(kMemcpyH2DStreamId));
  EXPECT_NE(ExecutionStreamId(kMemcpyD2HStreamId),
            ExecutionStreamId(ComputationStreamId(0)));
  EXPECT_NE(ExecutionStreamId(kMemcpyD2HStreamId),
            ExecutionStreamId(CommunicationStreamId(0)));
}

}  // namespace
}  // namespace xla::gpu
