/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/cpu/collectives/in_process_communicator.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/collectives/cpu_collectives.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/rendezvous.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {
namespace {

static absl::Duration kWarnStuckTimeout = absl::Seconds(5);
static absl::Duration kTerminateTimeout = absl::Seconds(10);

// In-process collective operation participants.
//
// In-process collective operations implemented as a multi-step process
// coordinated by a rendezvous:
//
// - All participants arrive to a rendezvous barrier and exchange their local
//   participant information.
//
// - Rendezvous leader collects data from all participants and returns an
//   `OpParticipants` struct to all collective participants.
//
// - Each participant computes its own portion of the collective result in
//   parallel. Each partitipant waits for the completion of all other
//   participants before proceeding.
//
template <typename Participant>
class OpParticipants {
 public:
  explicit OpParticipants(std::vector<Participant> participants)
      : counter_(participants.size()), participants_(std::move(participants)) {}

  // Invokes user-provided collective operation function with participants data
  // sorted by rank and user-provided arguments. Returns when all participants
  // complete their portion of the collective operation.
  template <typename Fn, typename... Args>
  absl::Status Invoke(Fn fn, Args... args) {
    // Invoke user-provided collective operation function.
    absl::Status status = fn(participants_, std::forward<Args>(args)...);

    int64_t cnt = counter_.fetch_sub(1, std::memory_order_acq_rel);
    DCHECK_GE(cnt, 1) << "Counter should never drop below 0";

    // Wait for the completion of all collective operation participants.
    if (cnt == 1) notification_.Notify();
    notification_.WaitForNotification();

    return status;
  }

 private:
  std::atomic<int64_t> counter_;
  absl::Notification notification_;
  std::vector<Participant> participants_;
};

// Comparator for sorting participants by rank.
template <typename Participant>
static bool ByRank(const Participant* a, const Participant* b) {
  return a->rank < b->rank;
}

// Collects participants for an in-process collective operation.
template <typename Participant>
std::vector<Participant> CollectParticipants(
    absl::Span<const Participant*> participants) {
  absl::c_sort(participants, ByRank<Participant>);

  std::vector<Participant> ret;
  ret.reserve(participants.size());
  for (const Participant* participant : participants) {
    ret.push_back(*participant);
  }

  return ret;
}

template <typename T>
T GetInitialValue(ReductionKind reduction_kind) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return static_cast<T>(0);
    case ReductionKind::PRODUCT:
      return static_cast<T>(1);
    case ReductionKind::MIN:
      return std::numeric_limits<T>::has_infinity
                 ? std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::max();
    case ReductionKind::MAX:
      return std::numeric_limits<T>::has_infinity
                 ? -std::numeric_limits<T>::infinity()
                 : std::numeric_limits<T>::lowest();
  }
}

// We cannot use static_assert(false), because the C++ standard (prior to
// CWG2518) does not allow the statement discarded by a constexpr if to
// be ill-formed for every possible specialization.
// See https://en.cppreference.com/w/cpp/language/if#Constexpr_if
template <ReductionKind>
constexpr bool always_false_v = false;

template <ReductionKind reduction_kind, typename T>
void ReduceHelper(absl::Span<T> acc, absl::Span<T const* const> inputs) {
  // TODO(penporn): make sure this gets vectorized.
  if constexpr (reduction_kind == ReductionKind::SUM) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] += inputs[j][i];
      }
    }
  } else if constexpr (reduction_kind == ReductionKind::PRODUCT) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] *= inputs[j][i];
      }
    }
  } else if constexpr (reduction_kind == ReductionKind::MIN) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] = std::min(acc[i], inputs[j][i]);
      }
    }
  } else if constexpr (reduction_kind == ReductionKind::MAX) {
    for (size_t j = 0; j < inputs.size(); ++j) {
      for (size_t i = 0; i < acc.size(); ++i) {
        acc[i] = std::max(acc[i], inputs[j][i]);
      }
    }
  } else {
    static_assert(always_false_v<reduction_kind>, "Unsupported reduction kind");
  }
}

template <PrimitiveType PT>
absl::Status ReduceScatter(ReductionKind reduction_kind,
                           absl::Span<const void* const> inputs, void* output,
                           int64_t num_elems) {
  using T = primitive_util::NativeTypeOf<PT>;
  T initial_value = GetInitialValue<T>(reduction_kind);

  absl::Span<T> out_chunk =
      absl::MakeSpan(reinterpret_cast<T*>(output), num_elems);
  for (int64_t i = 0; i < num_elems; ++i) {
    out_chunk[i] = initial_value;
  }

  absl::Span<T const* const> input_chunks(
      reinterpret_cast<T const* const*>(inputs.data()), inputs.size());
  switch (reduction_kind) {
    case ReductionKind::SUM:
      ReduceHelper<ReductionKind::SUM, T>(out_chunk, input_chunks);
      break;
    case ReductionKind::PRODUCT:
      ReduceHelper<ReductionKind::PRODUCT, T>(out_chunk, input_chunks);
      break;
    case ReductionKind::MIN:
      if constexpr (!is_complex_v<T>) {
        ReduceHelper<ReductionKind::MIN, T>(out_chunk, input_chunks);
      } else {
        return absl::InvalidArgumentError(
            "Min reductions not supported for complex types");
      }
      break;
    case ReductionKind::MAX:
      if constexpr (!is_complex_v<T>) {
        ReduceHelper<ReductionKind::MAX, T>(out_chunk, input_chunks);
      } else {
        return absl::InvalidArgumentError(
            "Max reductions not supported for complex types");
      }
      break;
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllReduce
//===----------------------------------------------------------------------===//

struct AllReduceParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

static absl::Status AllReduceOp(
    absl::Span<const AllReduceParticipant> participants, size_t rank,
    PrimitiveType primitive_type, size_t count, ReductionKind reduction_kind) {
  if (!primitive_util::IsArrayType(primitive_type)) {
    return Unimplemented(
        "Unexpected datatype: %s",
        primitive_util::LowercasePrimitiveTypeName(primitive_type));
  }

  // Each partiticipant will process a single chunk of the data and then copy
  // the result to all other participants.
  size_t chunk_size = tsl::MathUtil::CeilOfRatio(count, participants.size());

  // Compute the count of elements to process for the given participant rank.
  size_t chunk_count = std::min(chunk_size * (rank + 1), count) -
                       std::min(chunk_size * rank, count);
  if (chunk_count == 0) return absl::OkStatus();

  // Returns a pointer to the chunk of data for the given participant rank.
  auto chunk_ptr = [&](se::DeviceMemoryBase mem) -> void* {
    std::byte* ptr = static_cast<std::byte*>(mem.opaque());
    return ptr + rank * chunk_size * primitive_util::ByteWidth(primitive_type);
  };

  // Collect reduction inputs from all participants.
  std::vector<const void*> inputs(participants.size());
  for (auto& participant : participants) {
    inputs[participant.rank] = chunk_ptr(participant.src);
  }

  // Reduce all inputs into the destination buffer at rank 0.
  void* output = chunk_ptr(participants[0].dest);

  TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch(
      [&](const auto type_tag) {
        return ReduceScatter<type_tag>(reduction_kind, inputs, output,
                                       chunk_count);
      },
      primitive_type));

  // Copy all-reduced output to all other participants.
  for (size_t i = 1; i < participants.size(); ++i) {
    std::memcpy(chunk_ptr(participants[i].dest),
                chunk_ptr(participants[0].dest),
                chunk_count * primitive_util::ByteWidth(primitive_type));
  }

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// ReduceScatter
//===----------------------------------------------------------------------===//

struct ReduceScatterParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

static absl::Status ReduceScatterOp(
    absl::Span<const ReduceScatterParticipant> participants, size_t rank,
    PrimitiveType primitive_type, size_t count, ReductionKind reduction_kind) {
  if (!primitive_util::IsArrayType(primitive_type)) {
    return Unimplemented(
        "Unexpected datatype: %s",
        primitive_util::LowercasePrimitiveTypeName(primitive_type));
  }

  size_t num_participants = participants.size();
  size_t num_bytes = count * primitive_util::ByteWidth(primitive_type);

  size_t offset = rank * num_bytes;

  // Collect reduction inputs from all participants.
  std::vector<const void*> inputs(num_participants);
  for (size_t j = 0; j < num_participants; ++j) {
    std::byte* src = static_cast<std::byte*>(participants[j].src.opaque());
    inputs[j] = src + offset;
  }

  // Reduce all inputs into the destination buffer.
  void* output = participants[rank].dest.opaque();

  TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch(
      [&](const auto type_tag) {
        return ReduceScatter<type_tag>(reduction_kind, inputs, output, count);
      },
      primitive_type));

  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllGather
//===----------------------------------------------------------------------===//

struct AllGatherParticipant {
  size_t rank;
  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

static absl::Status AllGatherOp(
    absl::Span<const AllGatherParticipant> participants, size_t rank,
    size_t num_bytes) {
  for (size_t i = 0; i < participants.size(); ++i) {
    std::byte* dest = static_cast<std::byte*>(participants[rank].dest.opaque());
    std::memcpy(dest + i * num_bytes, participants[i].src.opaque(), num_bytes);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// AllToAll
//===----------------------------------------------------------------------===//

struct AllToAllParticipant {
  size_t rank;

  std::vector<se::DeviceMemoryBase> src;
  std::vector<se::DeviceMemoryBase> dest;
};

static absl::Status AllToAllOp(
    absl::Span<const AllToAllParticipant> participants, size_t rank,
    size_t num_bytes) {
  for (size_t i = 0; i < participants.size(); ++i) {
    std::memcpy(participants[i].dest[rank].opaque(),
                participants[rank].src[i].opaque(), num_bytes);
  }
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CollectivePermute
//===----------------------------------------------------------------------===//

struct CollectivePermuteParticipant {
  size_t rank;
  std::optional<RankId> src_rank;

  se::DeviceMemoryBase src;
  se::DeviceMemoryBase dest;
};

static absl::Status CollectivePermuteOp(
    absl::Span<const CollectivePermuteParticipant> participants, size_t rank,
    size_t num_bytes) {
  const CollectivePermuteParticipant& participant = participants[rank];
  void* dest = participant.dest.opaque();

  if (participant.src_rank) {
    size_t src_rank = participant.src_rank->value();
    std::memcpy(dest, participants.at(src_rank).src.opaque(), num_bytes);
  } else {
    std::memset(dest, 0, num_bytes);
  }

  return absl::OkStatus();
}

}  // namespace

//===----------------------------------------------------------------------===//

InProcessCommunicator::InProcessCommunicator(size_t rank, size_t num_ranks)
    : rank_(rank), num_ranks_(num_ranks) {}

tsl::AsyncValueRef<InProcessCommunicator::Event>
InProcessCommunicator::AllReduce(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 ReductionKind reduction_kind,
                                 const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all reduce ", key.ToString());
  AllReduceParticipant partiticipant{rank_, send_buffer, recv_buffer};

  TF_ASSIGN_OR_RETURN(
      auto op, Rendezvous<OpParticipants<AllReduceParticipant>>(
                   name, key, partiticipant, key.num_local_participants,
                   CollectParticipants<AllReduceParticipant>, kWarnStuckTimeout,
                   kTerminateTimeout));

  TF_RETURN_IF_ERROR(
      op->Invoke(AllReduceOp, rank_, dtype, count, reduction_kind));

  return OkEvent();
}

tsl::AsyncValueRef<InProcessCommunicator::Event>
InProcessCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                     se::DeviceMemoryBase recv_buffer,
                                     PrimitiveType dtype, size_t count,
                                     ReductionKind reduction_kind,
                                     const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("reduce scatter ", key.ToString());
  ReduceScatterParticipant partiticipant{rank_, send_buffer, recv_buffer};

  TF_ASSIGN_OR_RETURN(auto op,
                      Rendezvous<OpParticipants<ReduceScatterParticipant>>(
                          name, key, partiticipant, key.num_local_participants,
                          CollectParticipants<ReduceScatterParticipant>,
                          kWarnStuckTimeout, kTerminateTimeout));

  TF_RETURN_IF_ERROR(
      op->Invoke(ReduceScatterOp, rank_, dtype, count, reduction_kind));

  return OkEvent();
}

tsl::AsyncValueRef<InProcessCommunicator::Event>
InProcessCommunicator::CollectivePermute(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         std::optional<RankId> source_rank,
                                         absl::Span<const RankId> target_ranks,
                                         const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("collective permute ", key.ToString());
  CollectivePermuteParticipant partiticipant{rank_, source_rank, send_buffer,
                                             recv_buffer};

  TF_ASSIGN_OR_RETURN(auto op,
                      Rendezvous<OpParticipants<CollectivePermuteParticipant>>(
                          name, key, partiticipant, key.num_local_participants,
                          CollectParticipants<CollectivePermuteParticipant>,
                          kWarnStuckTimeout, kTerminateTimeout));

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);

  TF_RETURN_IF_ERROR(op->Invoke(CollectivePermuteOp, rank_, num_bytes));

  return OkEvent();
}

tsl::AsyncValueRef<InProcessCommunicator::Event>
InProcessCommunicator::AllToAll(
    absl::InlinedVector<se::DeviceMemoryBase, 4> send_buffers,
    absl::InlinedVector<se::DeviceMemoryBase, 4> recv_buffers,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all to all ", key.ToString());
  AllToAllParticipant partiticipant{rank_,
                                    {send_buffers.begin(), send_buffers.end()},
                                    {recv_buffers.begin(), recv_buffers.end()}};

  TF_ASSIGN_OR_RETURN(
      auto op, Rendezvous<OpParticipants<AllToAllParticipant>>(
                   name, key, partiticipant, key.num_local_participants,
                   CollectParticipants<AllToAllParticipant>, kWarnStuckTimeout,
                   kTerminateTimeout));

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);

  TF_RETURN_IF_ERROR(op->Invoke(AllToAllOp, rank_, num_bytes));

  return OkEvent();
}

tsl::AsyncValueRef<InProcessCommunicator::Event>
InProcessCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                 se::DeviceMemoryBase recv_buffer,
                                 PrimitiveType dtype, size_t count,
                                 const Executor& executor) {
  TF_ASSIGN_OR_RETURN(auto cpu_executor, CpuCollectives::TryCast(&executor));
  const RendezvousKey& key = cpu_executor->rendezvous_key();

  std::string name = absl::StrCat("all gather ", key.ToString());
  AllGatherParticipant partiticipant{rank_, send_buffer, recv_buffer};

  TF_ASSIGN_OR_RETURN(
      auto op, Rendezvous<OpParticipants<AllGatherParticipant>>(
                   name, key, partiticipant, key.num_local_participants,
                   CollectParticipants<AllGatherParticipant>, kWarnStuckTimeout,
                   kTerminateTimeout));

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);

  TF_RETURN_IF_ERROR(op->Invoke(AllGatherOp, rank_, num_bytes));

  return OkEvent();
}

}  // namespace xla::cpu
