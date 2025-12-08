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

#include <any>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
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

  GpuCliqueKey key({id0, id1}, /*num_local_participants=*/2, false);

  auto task = [&](int32_t id) {
    return [&, id] {
      auto rendezvous = GpuCliqueRendezvous::Join(key, RankId(id),
                                                  std::make_any<int32_t>(id));
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

}  // namespace xla::gpu
