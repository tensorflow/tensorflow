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

#include "xla/backends/cpu/collectives/gloo_collectives.h"

#include <unistd.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_clique_key.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/backends/cpu/collectives/gloo_kv_store.h"
#include "xla/backends/cpu/collectives/gloo_reduce_scatter.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/executable_run_options.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/runtime/device_id.h"
#include "xla/service/collective_rendezvous.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/test.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

#if defined(__linux__)
#include "gloo/transport/tcp/attr.h"
#include "gloo/transport/tcp/device.h"
#elif defined(__APPLE__)
#include "gloo/transport/uv/device.h"
#endif  // defined(__linux__)

namespace xla::cpu {

namespace {
using ::testing::Each;
using ::testing::Eq;

constexpr int kNumParticipants = 2;
constexpr size_t kBufferSize = 256;
constexpr absl::Duration kTimeout = absl::Seconds(5);

absl::StatusOr<std::unique_ptr<Communicator>> GetCommunicator(
    absl::Span<GlobalDeviceId const> global_devices,
    const std::shared_ptr<xla::KeyValueStoreInterface>& kv_store, int rank) {
  auto collectives = std::make_shared<cpu::GlooCollectives>(
      std::make_unique<cpu::GlooKeyValueStore>(kv_store),
#if defined(__linux__)
      gloo::transport::tcp::CreateDevice(gloo::transport::tcp::attr()));
#elif defined(__APPLE__)
      gloo::transport::uv::CreateDevice(gloo::transport::uv::attr()));
#endif  // defined(__linux__)

  CpuCliqueKey clique_key(global_devices);
  CpuCollectives::DeviceRank device_rank(nullptr, RankId(rank));

  ABSL_ASSIGN_OR_RETURN(
      auto communicators,
      collectives->CreateCommunicators(clique_key, std::nullopt, {device_rank},
                                       CpuCollectives::Config()));

  return std::move(communicators[0]);
}

RendezvousKey MakeRendezvousKey(std::vector<GlobalDeviceId> global_devices) {
  return RendezvousKey(RunId(0), global_devices, global_devices.size(),
                       RendezvousKey::CollectiveOpKind::kCrossModule,
                       /*op_id=*/0);
}

// TODO(cobley) - add tests for other collectives.

template <typename T>
static se::DeviceAddressBase AsDeviceMemory(const std::vector<T>& data) {
  return se::DeviceAddressBase(const_cast<T*>(data.data()),
                               data.size() * sizeof(T));
}

absl::StatusOr<std::vector<uint8_t>> AllReduce(
    const std::shared_ptr<xla::KeyValueStoreInterface>& kv_store,
    const std::vector<uint8_t>& input_buffer,
    std::vector<GlobalDeviceId> global_devices, int rank) {
  std::vector<uint8_t> output_buffer(kBufferSize);
  RendezvousKey rendezvous_key = MakeRendezvousKey(global_devices);
  ABSL_ASSIGN_OR_RETURN(auto communicator,
                   GetCommunicator(global_devices, kv_store, rank));

  CpuCollectives::Executor executor(rendezvous_key, kTimeout);
  auto event = communicator->AllReduce(
      AsDeviceMemory(input_buffer), AsDeviceMemory(output_buffer),
      xla::PrimitiveType::U8, kBufferSize, xla::ReductionKind::SUM, executor);

  ABSL_RETURN_IF_ERROR(event.Await());

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
    ASSERT_TRUE(output_buffers[rank].ok()) << output_buffers[rank].status();
    EXPECT_THAT(output_buffers[rank].value(),
                Each(Eq(kNumParticipants * (kNumParticipants + 1) / 2)));
  }
}

template <typename T>
absl::StatusOr<std::vector<T>> RunReduceScatter(
    const std::shared_ptr<xla::KeyValueStoreInterface>& kv_store,
    const std::vector<T>& input_buffer,
    std::vector<GlobalDeviceId> global_devices, int rank, PrimitiveType dtype,
    size_t chunk_count, ReductionKind reduction_kind) {
  std::vector<T> output_buffer(chunk_count);
  RendezvousKey rendezvous_key = MakeRendezvousKey(global_devices);
  ABSL_ASSIGN_OR_RETURN(auto communicator,
                   GetCommunicator(global_devices, kv_store, rank));

  CpuCollectives::Executor executor(rendezvous_key, kTimeout);
  auto event = communicator->ReduceScatter(
      AsDeviceMemory(input_buffer), AsDeviceMemory(output_buffer), dtype,
      chunk_count, reduction_kind, executor);

  ABSL_RETURN_IF_ERROR(event.Await());

  return output_buffer;
}

template <typename T>
void TestReduceScatter(int num_participants, size_t chunk_count,
                       PrimitiveType dtype,
                       ReductionKind reduction_kind = ReductionKind::SUM,
                       int num_iterations = 1) {
  std::vector<GlobalDeviceId> global_devices;
  global_devices.reserve(num_participants);
  for (int rank = 0; rank < num_participants; ++rank) {
    global_devices.push_back(GlobalDeviceId(rank));
  }

  auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
  const size_t total_count = chunk_count * num_participants;
  std::vector<absl::Status> statuses(num_participants, absl::OkStatus());

  auto elem_value = [](int r, int iter, size_t idx) -> T {
    return static_cast<T>(static_cast<int>((r + 1) + ((iter + idx) % 5)));
  };

  {
    tsl::thread::ThreadPool thread_pool(
        tsl::Env::Default(), "ReduceScatterParticipants", num_participants);
    for (int rank = 0; rank < num_participants; ++rank) {
      thread_pool.Schedule([rank, num_participants, chunk_count, total_count,
                            reduction_kind, num_iterations, elem_value,
                            &statuses, &kv_store, &global_devices, dtype]() {
        auto comm_or = GetCommunicator(global_devices, kv_store, rank);
        if (!comm_or.ok()) {
          statuses[rank] = comm_or.status();
          return;
        }
        std::vector<T> input_buffer(total_count);
        std::vector<T> output_buffer(chunk_count);

        for (int iter = 0; iter < num_iterations; ++iter) {
          for (size_t i = 0; i < total_count; ++i) {
            input_buffer[i] = elem_value(rank, iter, i);
          }
          RendezvousKey rendezvous_key = MakeRendezvousKey(global_devices);
          CpuCollectives::Executor executor(rendezvous_key, kTimeout);
          auto event = (*comm_or)->ReduceScatter(
              AsDeviceMemory(input_buffer), AsDeviceMemory(output_buffer),
              dtype, chunk_count, reduction_kind, executor);
          statuses[rank] = event.Await();
          if (!statuses[rank].ok()) {
            return;
          }
          for (size_t i = 0; i < chunk_count; ++i) {
            const size_t global_idx = rank * chunk_count + i;
            T expected = elem_value(0, iter, global_idx);
            for (int r = 1; r < num_participants; ++r) {
              const T v = elem_value(r, iter, global_idx);
              switch (reduction_kind) {
                case ReductionKind::SUM:
                  expected = expected + v;
                  break;
                case ReductionKind::PRODUCT:
                  expected = expected * v;
                  break;
                case ReductionKind::MIN:
                  if constexpr (!is_complex_v<T>) {
                    expected = std::min(expected, v);
                  }
                  break;
                case ReductionKind::MAX:
                  if constexpr (!is_complex_v<T>) {
                    expected = std::max(expected, v);
                  }
                  break;
              }
            }
            if (output_buffer[i] != expected) {
              statuses[rank] = absl::InternalError("Output mismatch");
              return;
            }
          }
        }
      });
    }
  }

  for (int rank = 0; rank < num_participants; ++rank) {
    TF_ASSERT_OK(statuses[rank])
        << "num_participants=" << num_participants << " rank=" << rank;
  }
}

class ReduceScatterSweepTest
    : public ::testing::TestWithParam<std::tuple<int, size_t>> {};

TEST_P(ReduceScatterSweepTest, PowerOfTwoAndNonPowerOfTwo) {
  auto [num_participants, chunk_count] = GetParam();
  TestReduceScatter<int32_t>(num_participants, chunk_count,
                             xla::PrimitiveType::S32);
}

INSTANTIATE_TEST_SUITE_P(
    GlooCollectives, ReduceScatterSweepTest,
    ::testing::Combine(::testing::Range(1, 17),
                       ::testing::Values(size_t{1}, size_t{3}, size_t{5},
                                         size_t{7}, size_t{13}, size_t{31},
                                         size_t{64}, size_t{127}, size_t{256},
                                         size_t{1000}, size_t{4096})));

TEST(GlooCollectives, ReduceScatterAliasedBuffer) {
  constexpr int kParticipants = 3;
  constexpr size_t kChunkCount = 7;
  constexpr size_t kTotalCount = kParticipants * kChunkCount;

  std::vector<GlobalDeviceId> global_devices;
  for (int rank = 0; rank < kParticipants; ++rank) {
    global_devices.push_back(GlobalDeviceId(rank));
  }

  for (bool partial_overlap : {false, true}) {
    auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
    std::vector<absl::Status> statuses(kParticipants);
    std::vector<std::vector<int32_t>> buffers(kParticipants);

    {
      tsl::thread::ThreadPool thread_pool(
          tsl::Env::Default(), "ReduceScatterAliased", kParticipants);
      for (int rank = 0; rank < kParticipants; ++rank) {
        thread_pool.Schedule([rank, partial_overlap, &statuses, &buffers,
                              &kv_store, &global_devices]() {
          buffers[rank].resize(kTotalCount);
          for (size_t i = 0; i < kTotalCount; ++i) {
            buffers[rank][i] = (rank + 1) * 1000 + static_cast<int32_t>(i);
          }
          auto comm_or = GetCommunicator(global_devices, kv_store, rank);
          if (!comm_or.ok()) {
            statuses[rank] = comm_or.status();
            return;
          }
          se::DeviceAddressBase send_mem(buffers[rank].data(),
                                         kTotalCount * sizeof(int32_t));
          const size_t recv_elem_offset =
              partial_overlap ? rank * kChunkCount : 0;
          se::DeviceAddressBase recv_mem(
              buffers[rank].data() + recv_elem_offset,
              kChunkCount * sizeof(int32_t));
          RendezvousKey rendezvous_key = MakeRendezvousKey(global_devices);
          CpuCollectives::Executor executor(rendezvous_key, kTimeout);
          auto event = (*comm_or)->ReduceScatter(
              send_mem, recv_mem, xla::PrimitiveType::S32, kChunkCount,
              xla::ReductionKind::SUM, executor);
          statuses[rank] = event.Await();
        });
      }
    }

    const int32_t rank_sum = kParticipants * (kParticipants + 1) / 2;
    for (int rank = 0; rank < kParticipants; ++rank) {
      TF_ASSERT_OK(statuses[rank]);
      const size_t recv_elem_offset = partial_overlap ? rank * kChunkCount : 0;
      for (size_t i = 0; i < kChunkCount; ++i) {
        const size_t global_idx = rank * kChunkCount + i;
        const int32_t expected =
            rank_sum * 1000 + static_cast<int32_t>(kParticipants * global_idx);
        EXPECT_EQ(buffers[rank][recv_elem_offset + i], expected);
      }
    }
  }
}

TEST(GlooCollectives, ReduceScatterLargeBuffer) {
  constexpr size_t kChunkCount = 1 << 20;  // 4 MiB per rank
  for (int num_participants : {4, 6}) {
    TestReduceScatter<uint32_t>(num_participants, kChunkCount,
                                xla::PrimitiveType::U32, ReductionKind::SUM,
                                /*num_iterations=*/3);
  }
}

TEST(GlooCollectives, ReduceScatterTypesAndReductionKinds) {
  constexpr int kParticipants = 3;
  constexpr size_t kChunkCount = 7;

  for (ReductionKind kind : {ReductionKind::SUM, ReductionKind::PRODUCT,
                             ReductionKind::MIN, ReductionKind::MAX}) {
    TestReduceScatter<int32_t>(kParticipants, kChunkCount,
                               xla::PrimitiveType::S32, kind);
    TestReduceScatter<float>(kParticipants, kChunkCount,
                             xla::PrimitiveType::F32, kind);
    TestReduceScatter<double>(kParticipants, kChunkCount,
                              xla::PrimitiveType::F64, kind);
    TestReduceScatter<xla::half>(kParticipants, kChunkCount,
                                 xla::PrimitiveType::F16, kind);
    TestReduceScatter<xla::bfloat16>(kParticipants, kChunkCount,
                                     xla::PrimitiveType::BF16, kind);
  }

  for (ReductionKind kind : {ReductionKind::SUM, ReductionKind::PRODUCT}) {
    TestReduceScatter<std::complex<float>>(kParticipants, kChunkCount,
                                           xla::PrimitiveType::C64, kind);
    TestReduceScatter<std::complex<double>>(kParticipants, kChunkCount,
                                            xla::PrimitiveType::C128, kind);
  }
}

TEST(GlooCollectives, ReduceScatterMinMaxOnComplexFails) {
  std::vector<GlobalDeviceId> global_devices = {GlobalDeviceId(0),
                                                GlobalDeviceId(1)};
  constexpr size_t kChunkCount = 4;

  for (PrimitiveType dtype :
       {xla::PrimitiveType::C64, xla::PrimitiveType::C128}) {
    for (ReductionKind kind : {ReductionKind::MIN, ReductionKind::MAX}) {
      auto kv_store = std::make_shared<xla::InMemoryKeyValueStore>();
      std::vector<absl::StatusOr<std::vector<std::complex<double>>>> results(2);
      {
        tsl::thread::ThreadPool thread_pool(tsl::Env::Default(),
                                            "ComplexMinMax", 2);
        for (int rank = 0; rank < 2; ++rank) {
          thread_pool.Schedule(
              [rank, &results, &kv_store, &global_devices, dtype, kind]() {
                std::vector<std::complex<double>> input(2 * kChunkCount,
                                                        {1.0, 2.0});
                results[rank] = RunReduceScatter<std::complex<double>>(
                    kv_store, input, global_devices, rank, dtype, kChunkCount,
                    kind);
              });
        }
      }
      for (int rank = 0; rank < 2; ++rank) {
        EXPECT_EQ(results[rank].status().code(),
                  absl::StatusCode::kInvalidArgument);
      }
    }
  }
}

TEST(GlooCollectives, ReverseLastNBits) {
  EXPECT_EQ(internal::ReverseLastNBits(0, 0), 0u);
  EXPECT_EQ(internal::ReverseLastNBits(0xffffffffu, 0), 0u);

  // n = 1: only lowest bit is kept; high bits are ignored.
  EXPECT_EQ(internal::ReverseLastNBits(0xa5a5a5a0u | 0b0u, 1), 0b0u);
  EXPECT_EQ(internal::ReverseLastNBits(0x5a5a5a50u | 0b1u, 1), 0b1u);

  // n = 2: 00 -> 00, 01 -> 10, 10 -> 01, 11 -> 11; high bits are ignored.
  EXPECT_EQ(internal::ReverseLastNBits(0xdeadbee0u | 0b00u, 2), 0b00u);
  EXPECT_EQ(internal::ReverseLastNBits(0xfeedfac0u | 0b01u, 2), 0b10u);
  EXPECT_EQ(internal::ReverseLastNBits(0x12345670u | 0b10u, 2), 0b01u);
  EXPECT_EQ(internal::ReverseLastNBits(0xabcdef00u | 0b11u, 2), 0b11u);

  // n = 3: 0..7 bit-reversed in 3 bits; high bits are ignored.
  EXPECT_EQ(internal::ReverseLastNBits(0x76543210u | 0b001u, 3), 0b100u);
  EXPECT_EQ(internal::ReverseLastNBits(0xf0e1d2c0u | 0b010u, 3), 0b010u);
  EXPECT_EQ(internal::ReverseLastNBits(0x89abcde0u | 0b011u, 3), 0b110u);
  EXPECT_EQ(internal::ReverseLastNBits(0x31415920u | 0b100u, 3), 0b001u);
  EXPECT_EQ(internal::ReverseLastNBits(0x27182818u | 0b001u, 3), 0b100u);

  // Reversing the lowest n bits twice recovers the original lowest n bits
  // (with upper bits cleared), and single bit b maps to bit n - 1 - b.
  for (uint32_t n = 1; n <= 32; ++n) {
    const uint32_t mask = n == 32 ? 0xffffffffu : ((uint32_t{1} << n) - 1);
    const uint32_t high_garbage = 0xa5c3f19eu & ~mask;
    for (uint32_t b = 0; b < n; ++b) {
      EXPECT_EQ(
          internal::ReverseLastNBits(high_garbage | (uint32_t{1} << b), n),
          uint32_t{1} << (n - 1 - b));
    }
    const uint32_t sample = 0x12345678u;
    EXPECT_EQ(internal::ReverseLastNBits(
                  high_garbage | internal::ReverseLastNBits(sample, n), n),
              sample & mask);
  }
}

}  // namespace
}  // namespace xla::cpu
