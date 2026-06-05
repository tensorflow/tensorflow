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

#include "xla/python/pjrt_ifrt/pjrt_buffer_hash_util.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "xla/python/ifrt/buffer_hash_util.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/shape.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/env.h"

namespace xla {
namespace ifrt {
namespace {

// Wraps `SchedClosure` to make it compatible with `tsl::Executor`.
class SchedClosureExecutor : public tsl::Executor {
 public:
  void Execute(Task task) override {
    tsl::Env::Default()->SchedClosure(
        [task = std::move(task)]() mutable { std::move(task)(); });
  }
};

TEST(PjRtBufferHashUtilTest, HashPjRtBuffers) {
  xla::CpuClientOptions options;
  options.cpu_device_count = 1;
  ASSERT_OK_AND_ASSIGN(auto xla_pjrt_client,
                       xla::GetXlaPjrtCpuClient(std::move(options)));
  SchedClosureExecutor executor;

  xla::PjRtDevice* xla_pjrt_device =
      xla_pjrt_client->addressable_devices().at(0);
  ASSERT_OK_AND_ASSIGN(auto* memory_space,
                       xla_pjrt_device->default_memory_space());

  xla::Literal literal = xla::LiteralUtil::CreateR2<float>(
      {{1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f}});

  ASSERT_OK_AND_ASSIGN(auto pjrt_buffer, xla_pjrt_client->BufferFromHostLiteral(
                                             literal, memory_space));
  ASSERT_OK(pjrt_buffer->GetReadyFuture().Await());

  std::vector<PjRtBuffer*> raw_buffers = {pjrt_buffer.get()};

  Shape shape({2, 3});
  std::vector<IndexDomain> index_domains = {IndexDomain(Index({0, 0}), shape)};

  ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> hash,
                       HashPjRtBuffers(executor, raw_buffers, index_domains,
                                       Client::HashMode::kPhysical)
                           .Await());
  EXPECT_EQ(hash.size(), 1);

  ASSERT_OK_AND_ASSIGN(hash,
                       HashPjRtBuffers(executor, raw_buffers, index_domains,
                                       Client::HashMode::kLogical)
                           .Await());
  EXPECT_EQ(hash.size(), 1);
}

TEST(PjRtBufferHashUtilTest, HashPjRtBuffersSharding) {
  xla::CpuClientOptions options;
  options.cpu_device_count = 2;
  ASSERT_OK_AND_ASSIGN(auto xla_pjrt_client,
                       xla::GetXlaPjrtCpuClient(std::move(options)));
  SchedClosureExecutor executor;

  xla::PjRtDevice* xla_pjrt_device0 =
      xla_pjrt_client->addressable_devices().at(0);
  ASSERT_OK_AND_ASSIGN(auto* memory_space0,
                       xla_pjrt_device0->default_memory_space());
  xla::PjRtDevice* xla_pjrt_device1 =
      xla_pjrt_client->addressable_devices().at(1);
  ASSERT_OK_AND_ASSIGN(auto* memory_space1,
                       xla_pjrt_device1->default_memory_space());

  Shape shape({4});
  Shape shard_shape({2});

  // Buffer 0: Full shard.
  xla::Literal literal0 =
      xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f, 3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer0,
      xla_pjrt_client->BufferFromHostLiteral(literal0, memory_space0));
  ASSERT_OK(pjrt_buffer0->GetReadyFuture().Await());

  // Buffer 1: Shard 0 out of 2
  xla::Literal literal1 = xla::LiteralUtil::CreateR1<float>({1.0f, 2.0f});
  ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer_shard0,
      xla_pjrt_client->BufferFromHostLiteral(literal1, memory_space0));
  ASSERT_OK(pjrt_buffer_shard0->GetReadyFuture().Await());

  // Buffer 2: Shard 1 out of 2
  xla::Literal literal2 = xla::LiteralUtil::CreateR1<float>({3.0f, 4.0f});
  ASSERT_OK_AND_ASSIGN(
      auto pjrt_buffer_shard1,
      xla_pjrt_client->BufferFromHostLiteral(literal2, memory_space1));
  ASSERT_OK(pjrt_buffer_shard1->GetReadyFuture().Await());

  // Physical mode.
  {
    std::vector<PjRtBuffer*> raw_buffers = {pjrt_buffer0.get()};
    std::vector<IndexDomain> domains = {IndexDomain(Index({0}), shape)};
    ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> hash0,
                         HashPjRtBuffers(executor, raw_buffers, domains,
                                         Client::HashMode::kPhysical)
                             .Await());
    ASSERT_EQ(hash0.size(), 1);

    std::vector<PjRtBuffer*> raw_buffers_shards = {pjrt_buffer_shard0.get(),
                                                   pjrt_buffer_shard1.get()};
    std::vector<IndexDomain> domains_shards = {
        IndexDomain(Index({0}), shard_shape),
        IndexDomain(Index({2}), shard_shape)};
    ASSERT_OK_AND_ASSIGN(
        std::vector<uint64_t> hashes_shards,
        HashPjRtBuffers(executor, raw_buffers_shards, domains_shards,
                        Client::HashMode::kPhysical)
            .Await());
    ASSERT_EQ(hashes_shards.size(), 2);
    std::vector<int> replica_group_ids = GetReplicaGroupIds(domains_shards);
    ASSERT_OK_AND_ASSIGN(uint64_t hash12,
                         AggregateShardHashes(hashes_shards, replica_group_ids,
                                              Client::HashMode::kPhysical));

    // Different shardings give different physical hashes.
    EXPECT_NE(hash0.front(), hash12);
  }

  // Logical mode.
  {
    std::vector<PjRtBuffer*> raw_buffers = {pjrt_buffer0.get()};
    std::vector<IndexDomain> domains = {IndexDomain(Index({0}), shape)};
    ASSERT_OK_AND_ASSIGN(std::vector<uint64_t> hash0,
                         HashPjRtBuffers(executor, raw_buffers, domains,
                                         Client::HashMode::kLogical)
                             .Await());
    ASSERT_EQ(hash0.size(), 1);

    std::vector<PjRtBuffer*> raw_buffers_shards = {pjrt_buffer_shard0.get(),
                                                   pjrt_buffer_shard1.get()};
    std::vector<IndexDomain> domains_shards = {
        IndexDomain(Index({0}), shard_shape),
        IndexDomain(Index({2}), shard_shape)};
    ASSERT_OK_AND_ASSIGN(
        std::vector<uint64_t> hashes_shards,
        HashPjRtBuffers(executor, raw_buffers_shards, domains_shards,
                        Client::HashMode::kLogical)
            .Await());
    ASSERT_EQ(hashes_shards.size(), 2);
    std::vector<int> replica_group_ids = GetReplicaGroupIds(domains_shards);
    ASSERT_OK_AND_ASSIGN(uint64_t hash12,
                         AggregateShardHashes(hashes_shards, replica_group_ids,
                                              Client::HashMode::kLogical));

    // The same content gives same logical hash.
    EXPECT_EQ(hash0.front(), hash12);
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
