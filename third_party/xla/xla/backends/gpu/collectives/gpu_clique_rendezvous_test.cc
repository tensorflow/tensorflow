/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/gpu_clique_rendezvous.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"

namespace xla::gpu {

tsl::thread::ThreadPool CreateThreadPool(int32_t size) {
  return tsl::thread::ThreadPool(tsl::Env::Default(), "rendezvous_test", size);
}

TEST(GpuCliqueRendezvousTest, TwoParticipants) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  GpuCliqueKey key({id0, id1}, /*num_local_participants=*/2);

  auto task = [&](int32_t id) {
    return [&, id] {
      auto rendezvous = GpuCliqueRendezvous::Join(key, RankId(id), int32_t{id});
      ASSERT_OK(rendezvous);

      GpuCliqueRendezvous& data = **rendezvous;
      ASSERT_EQ(data.clique_key(), key);
      ASSERT_EQ(*data.at<int32_t>(RankId(0)), 0);
      ASSERT_EQ(*data.at<int32_t>(RankId(1)), 1);
    };
  };

  auto thread_pool = CreateThreadPool(2);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
}

TEST(GpuCliqueRendezvousTest, MoveOnlyType) {
  using MoveOnly = std::unique_ptr<int32_t>;

  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  GpuCliqueKey key({id0, id1}, /*num_local_participants=*/2);

  auto task = [&](int32_t id) {
    return [&, id] {
      auto rendezvous = GpuCliqueRendezvous::Join(
          key, RankId(id), std::make_unique<int32_t>(id * 10));
      ASSERT_OK(rendezvous);

      GpuCliqueRendezvous& data = **rendezvous;
      ASSERT_EQ(*data.at<MoveOnly>(RankId(0))->get(), 0);
      ASSERT_EQ(*data.at<MoveOnly>(RankId(1))->get(), 10);
    };
  };

  auto thread_pool = CreateThreadPool(2);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
}

TEST(GpuCliqueRendezvousTest, ThreeParticipants) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);
  GlobalDeviceId id2 = GlobalDeviceId(2);

  GpuCliqueKey key({id0, id1, id2}, /*num_local_participants=*/3);

  auto task = [&](int32_t id) {
    return [&, id] {
      auto rendezvous =
          GpuCliqueRendezvous::Join(key, RankId(id), std::string(1, 'a' + id));
      ASSERT_OK(rendezvous);

      GpuCliqueRendezvous& data = **rendezvous;
      ASSERT_EQ(data.at<std::string>(RankId(0))->get(), "a");
      ASSERT_EQ(data.at<std::string>(RankId(1))->get(), "b");
      ASSERT_EQ(data.at<std::string>(RankId(2))->get(), "c");
    };
  };

  auto thread_pool = CreateThreadPool(3);
  thread_pool.Schedule(task(0));
  thread_pool.Schedule(task(1));
  thread_pool.Schedule(task(2));
}

TEST(GpuCliqueRendezvousTest, NonLocalCliqueReturnsError) {
  GlobalDeviceId id0 = GlobalDeviceId(0);
  GlobalDeviceId id1 = GlobalDeviceId(1);

  // num_local_participants < total participants → non-local clique
  GpuCliqueKey key({id0, id1}, /*num_local_participants=*/1);

  auto rendezvous = GpuCliqueRendezvous::Join(key, RankId(0), int32_t{42});
  ASSERT_FALSE(rendezvous.ok());
  EXPECT_THAT(rendezvous.status().message(), ::testing::HasSubstr("non-local"));
}

}  // namespace xla::gpu
