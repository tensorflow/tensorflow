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

#include "xla/python/ifrt/buffer_hash_util.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

template <typename T>
absl::Span<const char> MakeBufferSpan(const std::vector<T>& data) {
  return absl::MakeConstSpan(
      reinterpret_cast<const char*>(  // NOLINT(reinterpret_cast-array-type)
          data.data()),
      data.size() * sizeof(T));
}

TEST(BufferHashUtilTest, HashBufferPhysical) {
  std::vector<int32_t> buffer0 = {1, 2, 3, 4};
  std::vector<int32_t> buffer1 = {1, 2, 3, 5};
  xla::Layout layout0 = xla::LayoutUtil::MakeDescendingLayout(1);
  xla::Layout layout1 = xla::LayoutUtil::MakeLayout(/*minor_to_major=*/{1, 0},
                                                    /*tiles=*/{Tile({4, 1})});

  ASSERT_OK_AND_ASSIGN(uint64_t hash0,
                       HashBufferPhysical(MakeBufferSpan(buffer0), layout0));
  ASSERT_OK_AND_ASSIGN(uint64_t hash1,
                       HashBufferPhysical(MakeBufferSpan(buffer1), layout0));
  EXPECT_NE(hash0, hash1);

  ASSERT_OK_AND_ASSIGN(uint64_t hash2,
                       HashBufferPhysical(MakeBufferSpan(buffer1), layout1));
  EXPECT_NE(hash0, hash2);
  EXPECT_NE(hash1, hash2);
}

TEST(BufferHashUtilTest, HashBufferLogical) {
  std::vector<int32_t> buffer0 = {1, 2, 3, 4};
  std::vector<int32_t> buffer1 = {1, 2, 3, 5};

  ASSERT_OK_AND_ASSIGN(
      uint64_t hash0,
      HashBufferLogical(MakeBufferSpan(buffer0), /*element_byte_size=*/4,
                        IndexDomain(Index({0}), Shape({4}))));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash1,
      HashBufferLogical(MakeBufferSpan(buffer1), /*element_byte_size=*/4,
                        IndexDomain(Index({0}), Shape({4}))));
  EXPECT_NE(hash0, hash1);

  ASSERT_OK_AND_ASSIGN(
      uint64_t hash2,
      HashBufferLogical(MakeBufferSpan(buffer1), /*element_byte_size=*/4,
                        IndexDomain(Index({0}), Shape({4}))));
  EXPECT_EQ(hash1, hash2);
  EXPECT_NE(hash0, hash2);
}

TEST(BufferHashUtilTest, GetReplicaGroupIds) {
  std::vector<IndexDomain> index_domains_fully_replicated = {
      IndexDomain(Index({0}), Shape({4})),
      IndexDomain(Index({0}), Shape({4})),
  };
  EXPECT_THAT(GetReplicaGroupIds(index_domains_fully_replicated),
              testing::ElementsAre(0, 0));

  std::vector<IndexDomain> index_domains_partitioned = {
      IndexDomain(Index({0}), Shape({2})),
      IndexDomain(Index({2}), Shape({2})),
  };
  EXPECT_THAT(GetReplicaGroupIds(index_domains_partitioned),
              testing::ElementsAre(0, 1));

  std::vector<IndexDomain> index_domains_partially_replicated = {
      IndexDomain(Index({0}), Shape({2})),
      IndexDomain(Index({2}), Shape({2})),
      IndexDomain(Index({0}), Shape({2})),
      IndexDomain(Index({2}), Shape({2})),
  };
  EXPECT_THAT(GetReplicaGroupIds(index_domains_partially_replicated),
              testing::ElementsAre(0, 1, 0, 1));
}
TEST(BufferHashUtilTest, AggregateShardHashesPhysical) {
  std::vector<int32_t> buffer0 = {1, 2};
  std::vector<int32_t> buffer1 = {3, 4};
  std::vector<int32_t> buffer2 = {1, 3};
  std::vector<int32_t> buffer3 = {2, 4};
  xla::Layout layout;

  ASSERT_OK_AND_ASSIGN(uint64_t hash0,
                       HashBufferPhysical(MakeBufferSpan(buffer0), layout));
  ASSERT_OK_AND_ASSIGN(uint64_t hash1,
                       HashBufferPhysical(MakeBufferSpan(buffer1), layout));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash01,
      AggregateShardHashes({hash0, hash1}, /*replica_group_ids=*/{0, 1},
                           Client::HashMode::kPhysical));

  ASSERT_OK_AND_ASSIGN(uint64_t hash2,
                       HashBufferPhysical(MakeBufferSpan(buffer2), layout));
  ASSERT_OK_AND_ASSIGN(uint64_t hash3,
                       HashBufferPhysical(MakeBufferSpan(buffer3), layout));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash23,
      AggregateShardHashes({hash2, hash3}, /*replica_group_ids=*/{0, 1},
                           Client::HashMode::kPhysical));

  EXPECT_NE(hash01, hash23);
}

TEST(BufferHashUtilTest, AggregateShardHashesLogical) {
  std::vector<int32_t> buffer0 = {1, 2};
  std::vector<int32_t> buffer1 = {3, 4};
  std::vector<int32_t> buffer2 = {1, 3};
  std::vector<int32_t> buffer3 = {2, 4};

  ASSERT_OK_AND_ASSIGN(
      uint64_t hash0,
      HashBufferLogical(MakeBufferSpan(buffer0), /*element_byte_size=*/4,
                        IndexDomain(Index({0, 0}), Shape({1, 2}))));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash1,
      HashBufferLogical(MakeBufferSpan(buffer1), /*element_byte_size=*/4,
                        IndexDomain(Index({1, 0}), Shape({1, 2}))));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash01,
      AggregateShardHashes({hash0, hash1}, /*replica_group_ids=*/{0, 1},
                           Client::HashMode::kLogical));

  ASSERT_OK_AND_ASSIGN(
      uint64_t hash2,
      HashBufferLogical(MakeBufferSpan(buffer2), /*element_byte_size=*/4,
                        IndexDomain(Index({0, 0}), Shape({2, 1}))));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash3,
      HashBufferLogical(MakeBufferSpan(buffer3), /*element_byte_size=*/4,
                        IndexDomain(Index({0, 1}), Shape({2, 1}))));
  ASSERT_OK_AND_ASSIGN(
      uint64_t hash23,
      AggregateShardHashes({hash2, hash3}, /*replica_group_ids=*/{0, 1},
                           Client::HashMode::kLogical));

  EXPECT_EQ(hash01, hash23);
}

TEST(BufferHashUtilTest, AggregateShardHashesReplicaMismatch) {
  EXPECT_THAT(
      AggregateShardHashes(/*shard_hashes=*/{10, 20},
                           /*replica_group_ids=*/{0, 0},
                           Client::HashMode::kLogical),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Hash mismatch across replicas for shards 0 and 1 "
                         "(replica group ID 0). hashes: 10 vs. 20")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
