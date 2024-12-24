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

#include "xla/service/cpu/in_process_collectives.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/refcounting_hash_map.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace cpu {
namespace runtime {
namespace {

void FormatGlobalId(std::string* out, const GlobalDeviceId& device) {
  absl::StrAppend(out, device.value());
}

struct AllReduceParticipantData : ParticipantData {
  explicit AllReduceParticipantData(const RendezvousKey& rendezvous_key_p,
                                    int rank)
      : ParticipantData(rendezvous_key_p, rank) {}

  int64_t element_count;
  const void* source_data;
  void* destination_data;
  PrimitiveType primitive_type;

  ReductionKind reduction_kind;

  std::string ToString() const override {
    return absl::StrFormat(
        "AllReduceParticipantData{rank=%d, element_count=%d, type=%s, "
        "rendezvous_key=%s}",
        local_rank, element_count, PrimitiveType_Name(primitive_type),
        rendezvous_key.ToString());
  }
};

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

class CpuAllReduceRendezvous
    : public Rendezvous<AllReduceParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllReduceRendezvous(const RendezvousKey& k)
      : Rendezvous<AllReduceParticipantData, std::nullptr_t>(k) {}

 protected:
  absl::StatusOr<std::nullptr_t> RunCollectiveOp(
      const AllReduceParticipantData& me) override {
    VLOG(3) << me.ToString();
    int64_t world_size = participants_.size();
    // Divide the buffer up into equal(ish) chunks. Rank r computes the r-th
    // chunk of the output.
    int64_t chunk_elems = CeilOfRatio(me.element_count, world_size);

    int64_t start_elem = me.local_rank * chunk_elems;
    int64_t end_elem = std::min(start_elem + chunk_elems, me.element_count);
    chunk_elems = std::max(int64_t{0}, end_elem - start_elem);
    if (chunk_elems == 0) {
      return nullptr;
    }

    auto bytes_per_elem = primitive_util::ByteWidth(me.primitive_type);
    int64_t chunk_offset = start_elem * bytes_per_elem;
    int64_t chunk_bytes = chunk_elems * bytes_per_elem;
    void* reduce_output =
        reinterpret_cast<char*>(me.destination_data) + chunk_offset;

    std::vector<const void*> inputs;
    inputs.reserve(world_size);
    for (const auto& p : participants_) {
      inputs.push_back(reinterpret_cast<const char*>(p->source_data) +
                       chunk_offset);
    }

    if (primitive_util::IsArrayType(me.primitive_type)) {
      TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch<absl::Status>(
          [&](const auto constant_type) {
            return ReduceScatter<constant_type>(me.reduction_kind, inputs,
                                                reduce_output, chunk_elems);
          },
          me.primitive_type));
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Unexpected datatype: ",
          primitive_util::LowercasePrimitiveTypeName(me.primitive_type)));
    }

    // All-gather the reduced chunks.
    for (const auto& p : participants_) {
      if (p->local_rank != me.local_rank) {
        std::memcpy(reinterpret_cast<char*>(p->destination_data) + chunk_offset,
                    reduce_output, chunk_bytes);
      }
    }
    return nullptr;
  }
};

struct CollectivePermuteParticipantData : ParticipantData {
  CollectivePermuteParticipantData(const RendezvousKey& rendezvous_key_p,
                                   int rank)
      : ParticipantData(rendezvous_key_p, rank) {}
  const void* source_buffer;
  void* destination_buffer;
  size_t num_bytes;

  // From which rank is this participant receiving its data? Optional; if
  // absent fill with zeros.
  std::optional<int> source_rank;

  std::string ToString() const override {
    return absl::StrFormat(
        "CollectivePermuteParticipantData{rank=%d, "
        "source_buffer=%p, destination_buffer=%p, num_bytes=%d, "
        "source_replica_id=%d, "
        "devices=[%s]}",
        local_rank, source_buffer, destination_buffer, num_bytes,
        source_rank.value_or(-1),
        absl::StrJoin(rendezvous_key.global_devices, ", ", FormatGlobalId));
  }
};

class CpuCollectivePermuteRendezvous
    : public Rendezvous<CollectivePermuteParticipantData, std::nullptr_t> {
 public:
  explicit CpuCollectivePermuteRendezvous(const RendezvousKey& k)
      : Rendezvous<CollectivePermuteParticipantData, std::nullptr_t>(k) {}

 protected:
  CollectivesInterface* collectives_;

  absl::StatusOr<std::nullptr_t> RunCollectiveOp(
      const CollectivePermuteParticipantData& p) override {
    VLOG(3) << p.ToString();
    if (p.source_rank) {
      std::memcpy(p.destination_buffer,
                  participants_[*p.source_rank]->source_buffer, p.num_bytes);
    } else {
      std::memset(p.destination_buffer, 0, p.num_bytes);
    }
    return nullptr;
  }
};

struct AllToAllParticipantData : ParticipantData {
  AllToAllParticipantData(const RendezvousKey& rendezvous_key_p, int rank)
      : ParticipantData(rendezvous_key_p, rank) {}

  std::vector<const void*> source_buffers;
  std::vector<void*> destination_buffers;
  size_t chunk_size;

  std::string ToString() const override {
    auto addr_formatter = [](std::string* out, const void* mem) {
      absl::StrAppend(out, absl::StrFormat("%p", mem));
    };
    return absl::StrFormat(
        "AllToAllParticipantData{rank=%d, "
        "devices=[%s], source_buffers=[%s], "
        "destination_buffers=[%s], chunk_size=%d}",
        local_rank,
        absl::StrJoin(rendezvous_key.global_devices, ", ", FormatGlobalId),
        absl::StrJoin(source_buffers, ", ", addr_formatter),
        absl::StrJoin(destination_buffers, ", ", addr_formatter), chunk_size);
  }
};

class CpuAllToAllRendezvous
    : public Rendezvous<AllToAllParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllToAllRendezvous(const RendezvousKey& k)
      : Rendezvous<AllToAllParticipantData, std::nullptr_t>(k) {}

 protected:
  CollectivesInterface* collectives_;
  absl::StatusOr<std::nullptr_t> RunCollectiveOp(
      const AllToAllParticipantData& p) override {
    int world_size = p.rendezvous_key.global_devices.size();
    for (int i = 0; i < world_size; ++i) {
      std::memcpy(participants_[i]->destination_buffers[p.local_rank],
                  p.source_buffers[i], p.chunk_size);
    }
    return nullptr;
  }
};

struct AllGatherParticipantData : ParticipantData {
  AllGatherParticipantData(const RendezvousKey& rendezvous_key_p, int rank)
      : ParticipantData(rendezvous_key_p, rank) {}

  const void* source_buffer;
  void* destination_buffer;
  size_t chunk_size;

  std::string ToString() const override {
    return absl::StrFormat(
        "AllGatherParticipantData{rank=%d, "
        "devices=[%s], source_buffer=%p, "
        "destination_buffer=%p, chunk_size=%d}",
        local_rank,
        absl::StrJoin(rendezvous_key.global_devices, ", ", FormatGlobalId),
        source_buffer, destination_buffer, chunk_size);
  }
};

class CpuAllGatherRendezvous
    : public Rendezvous<AllGatherParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllGatherRendezvous(const RendezvousKey& k)
      : Rendezvous<AllGatherParticipantData, std::nullptr_t>(k) {}

 protected:
  CollectivesInterface* collectives_;
  absl::StatusOr<std::nullptr_t> RunCollectiveOp(
      const AllGatherParticipantData& p) override {
    int world_size = p.rendezvous_key.global_devices.size();
    char* out = static_cast<char*>(p.destination_buffer);
    for (int i = 0; i < world_size; ++i, out += p.chunk_size) {
      std::memcpy(out, participants_[i]->source_buffer, p.chunk_size);
    }
    return nullptr;
  }
};

struct ReduceScatterParticipantData : ParticipantData {
  ReduceScatterParticipantData(const RendezvousKey& rendezvous_key_p, int rank)
      : ParticipantData(rendezvous_key_p, rank) {}

  ReductionKind reduction_kind;
  PrimitiveType element_type;
  const void* source_buffer;
  void* destination_buffer;
  size_t chunk_elems;

  std::string ToString() const override {
    return absl::StrFormat(
        "ReduceScatterParticipantData{rank=%d, "
        "devices=[%s], source_buffer=%p, "
        "destination_buffer=%p, chunk_elems=%d}",
        local_rank,
        absl::StrJoin(rendezvous_key.global_devices, ", ", FormatGlobalId),
        source_buffer, destination_buffer, chunk_elems);
  }
};

class CpuReduceScatterRendezvous
    : public Rendezvous<ReduceScatterParticipantData, std::nullptr_t> {
 public:
  explicit CpuReduceScatterRendezvous(const RendezvousKey& k)
      : Rendezvous<ReduceScatterParticipantData, std::nullptr_t>(k) {}

 protected:
  CollectivesInterface* collectives_;
  absl::StatusOr<std::nullptr_t> RunCollectiveOp(
      const ReduceScatterParticipantData& me) override {
    auto bytes_per_elem = primitive_util::ByteWidth(me.element_type);
    int64_t chunk_offset = me.local_rank * me.chunk_elems * bytes_per_elem;

    std::vector<const void*> inputs;
    inputs.reserve(participants_.size());
    for (const auto& p : participants_) {
      inputs.push_back(reinterpret_cast<const char*>(p->source_buffer) +
                       chunk_offset);
    }

    if (primitive_util::IsArrayType(me.element_type)) {
      TF_RETURN_IF_ERROR(primitive_util::ArrayTypeSwitch<absl::Status>(
          [&](const auto constant_type) {
            return ReduceScatter<constant_type>(me.reduction_kind, inputs,
                                                me.destination_buffer,
                                                me.chunk_elems);
          },
          me.element_type));
    } else {
      return absl::UnimplementedError(absl::StrCat(
          "Unexpected datatype: ",
          primitive_util::LowercasePrimitiveTypeName(me.element_type)));
    }
    return nullptr;
  }
};

}  // namespace

struct InProcessCollectivesState {
  RefcountingHashMap<RendezvousKey, CpuAllReduceRendezvous>
      all_reduce_rendezvous_map;
  RefcountingHashMap<RendezvousKey, CpuCollectivePermuteRendezvous>
      collective_permute_rendezvous_map;
  RefcountingHashMap<RendezvousKey, CpuAllToAllRendezvous>
      all_to_all_rendezvous_map;
  RefcountingHashMap<RendezvousKey, CpuAllGatherRendezvous>
      all_gather_rendezvous_map;
  RefcountingHashMap<RendezvousKey, CpuReduceScatterRendezvous>
      reduce_scatter_rendezvous_map;
};

InProcessCollectivesCommunicator::InProcessCollectivesCommunicator(
    InProcessCollectivesState* state, int rank, int size)
    : state_(state), rank_(rank) {}
InProcessCollectivesCommunicator::~InProcessCollectivesCommunicator() = default;

absl::Status InProcessCollectivesCommunicator::AllReduce(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t num_elements,
    const void* const input_buffer, void* const output_buffer,
    absl::Duration timeout) {
  AllReduceParticipantData participant(key, rank_);
  participant.element_count = num_elements;
  participant.primitive_type = element_type;
  participant.source_data = input_buffer;
  participant.destination_data = output_buffer;
  participant.reduction_kind = reduction_kind;

  auto make_cpu_rendezvous = [](const RendezvousKey& k) {
    return std::make_unique<CpuAllReduceRendezvous>(k);
  };

  return CpuAllReduceRendezvous::SubmitParticipant(
             [&] {
               return state_->all_reduce_rendezvous_map.GetOrCreateIfAbsent(
                   key, make_cpu_rendezvous);
             },
             participant)
      .status();
}

absl::Status InProcessCollectivesCommunicator::CollectivePermute(
    const RendezvousKey& key, size_t num_bytes, std::optional<int> source_rank,
    absl::Span<int const> target_ranks, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  CollectivePermuteParticipantData participant(key, rank_);
  participant.source_buffer = input_buffer;
  participant.destination_buffer = output_buffer;
  participant.num_bytes = num_bytes;
  participant.source_rank = source_rank;
  auto make_cpu_rendezvous = [](const RendezvousKey& k) {
    return std::make_unique<CpuCollectivePermuteRendezvous>(k);
  };
  return CpuCollectivePermuteRendezvous::SubmitParticipant(
             [&] {
               return state_->collective_permute_rendezvous_map
                   .GetOrCreateIfAbsent(key, make_cpu_rendezvous);
             },
             participant)
      .status();
}

absl::Status InProcessCollectivesCommunicator::AllToAll(
    const RendezvousKey& key, size_t chunk_bytes,
    absl::Span<const void* const> input_buffers,
    absl::Span<void* const> output_buffers, absl::Duration timeout) {
  AllToAllParticipantData participant(key, rank_);
  TF_RET_CHECK(input_buffers.size() == output_buffers.size());
  participant.chunk_size = chunk_bytes;
  participant.source_buffers.reserve(input_buffers.size());
  participant.destination_buffers.reserve(output_buffers.size());
  for (const void* input_buffer : input_buffers) {
    participant.source_buffers.push_back(input_buffer);
  }
  for (void* output_buffer : output_buffers) {
    participant.destination_buffers.push_back(output_buffer);
  }
  auto make_cpu_rendezvous = [](const RendezvousKey& k) {
    return std::make_unique<CpuAllToAllRendezvous>(k);
  };
  return CpuAllToAllRendezvous::SubmitParticipant(
             [&] {
               return state_->all_to_all_rendezvous_map.GetOrCreateIfAbsent(
                   key, make_cpu_rendezvous);
             },
             participant)
      .status();
}

absl::Status InProcessCollectivesCommunicator::AllGather(
    const RendezvousKey& key, size_t chunk_bytes, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  AllGatherParticipantData participant(key, rank_);
  participant.chunk_size = chunk_bytes;
  participant.source_buffer = input_buffer;
  participant.destination_buffer = output_buffer;
  auto make_cpu_rendezvous = [](const RendezvousKey& k) {
    return std::make_unique<CpuAllGatherRendezvous>(k);
  };
  return CpuAllGatherRendezvous::SubmitParticipant(
             [&] {
               return state_->all_gather_rendezvous_map.GetOrCreateIfAbsent(
                   key, make_cpu_rendezvous);
             },
             participant)
      .status();
}

absl::Status InProcessCollectivesCommunicator::ReduceScatter(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  ReduceScatterParticipantData participant(key, rank_);
  participant.element_type = element_type;
  participant.reduction_kind = reduction_kind;
  participant.chunk_elems = chunk_elems;
  participant.source_buffer = input_buffer;
  participant.destination_buffer = output_buffer;
  auto make_cpu_rendezvous = [](const RendezvousKey& k) {
    return std::make_unique<CpuReduceScatterRendezvous>(k);
  };
  return CpuReduceScatterRendezvous::SubmitParticipant(
             [&] {
               return state_->reduce_scatter_rendezvous_map.GetOrCreateIfAbsent(
                   key, make_cpu_rendezvous);
             },
             participant)
      .status();
}
InProcessCollectives::InProcessCollectives()
    : state_(std::make_unique<InProcessCollectivesState>()) {}
InProcessCollectives::~InProcessCollectives() = default;

absl::StatusOr<std::shared_ptr<CollectivesCommunicator>>
InProcessCollectives::GetCommunicator(absl::Span<GlobalDeviceId const> devices,
                                      int rank) {
  // We don't care about devices here: we share rendezvous state globally.
  return std::make_shared<InProcessCollectivesCommunicator>(state_.get(), rank,
                                                            devices.size());
}

}  // namespace runtime
}  // namespace cpu
}  // namespace xla
