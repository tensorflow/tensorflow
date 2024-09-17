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

#include "xla/pjrt/cpu/gloo_collectives.h"

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#if defined(__linux__)
#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#elif defined(__APPLE__)
#include "gloo/transport/uv/device.h"
#endif  // defined(__linux__)
#include "xla/executable_run_options.h"
#include "xla/pjrt/cpu/gloo_kv_store.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {

namespace {
using ::testing::Each;
using ::testing::Eq;

constexpr int kNumParticipants = 2;
constexpr size_t kBufferSize = 256;
constexpr absl::Duration kTimeout = absl::Seconds(5);

absl::StatusOr<std::shared_ptr<CollectivesCommunicator>> GetCommunicator(
    size_t kNumParticipants, absl::Span<GlobalDeviceId const> global_devices,
    const std::shared_ptr<xla::KeyValueStoreInterface>& kv_store, int rank) {
  auto collectives = std::make_shared<cpu::GlooCollectives>(
      std::make_unique<cpu::GlooKeyValueStore>(kv_store),
#if defined(__linux__)
      gloo::transport::tcp::CreateDevice(gloo::transport::tcp::attr()));
#elif defined(__APPLE__)
      gloo::transport::uv::CreateDevice(gloo::transport::uv::attr()));
#endif  // defined(__linux__)
  return collectives->GetCommunicator(global_devices, rank);
}

RendezvousKey MakeRendezvousKey(std::vector<GlobalDeviceId> global_devices) {
  return RendezvousKey(RunId(0), global_devices, kNumParticipants,
                       RendezvousKey::CollectiveOpKind::kCrossModule,
                       /*op_id=*/0);
}

// TODO(cobley) - add tests for other collectives.

absl::StatusOr<std::vector<uint8_t>> AllReduce(
    const std::shared_ptr<xla::KeyValueStoreInterface>& kv_store,
    const std::vector<uint8_t>& input_buffer,
    std::vector<GlobalDeviceId> global_devices, int rank) {
  std::vector<uint8_t> output_buffer(kBufferSize);
  RendezvousKey rendezvous_key = MakeRendezvousKey(global_devices);
  TF_ASSIGN_OR_RETURN(
      auto communicator,
      GetCommunicator(kNumParticipants, global_devices, kv_store, rank));

  TF_RETURN_IF_ERROR(communicator->AllReduce(
      rendezvous_key, xla::ReductionKind::SUM, xla::PrimitiveType::U8,
      kBufferSize, input_buffer.data(), output_buffer.data(), kTimeout));

  return output_buffer;
}

TEST(GlooCollectives, AllReduce) {
  std::vector<GlobalDeviceId> global_devices;
  global_devices.reserve(kNumParticipants);
  for (int rank = 0; rank < kNumParticipants; ++rank) {
    global_devices.push_back(GlobalDeviceId(rank));
  }

  auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();

  // Create a vector of output buffers with one buffer per participant.
  std::vector<absl::StatusOr<std::vector<uint8_t>>> output_buffers(
      kNumParticipants);

  {
    // Perform the collective with each participant in a separate thread.
    tsl::thread::ThreadPool thread_pool(
        tsl::Env::Default(), "AllReduceParticipants", kNumParticipants);
    for (int rank = 0; rank < kNumParticipants; ++rank) {
      thread_pool.Schedule(
          [rank, &output_buffers, &kv_store, &global_devices]() {
            std::vector<uint8_t> input_buffer(kBufferSize, rank + 1);
            output_buffers[rank] =
                AllReduce(kv_store, input_buffer, global_devices, rank);
          });
    }
  }
  // thread_pool is now out of scope, so all threads have joined.

  // Verify that all participants successfully executed the collective.
  for (int rank = 0; rank < kNumParticipants; ++rank) {
    TF_ASSERT_OK(output_buffers[rank].status());
  }
  // Verify that all participants received the expected result.
  for (int rank = 0; rank < kNumParticipants; ++rank) {
    EXPECT_THAT(output_buffers[rank].value(),
                Each(Eq(kNumParticipants * (kNumParticipants + 1) / 2)));
  }
}
}  // namespace
}  // namespace xla::cpu
