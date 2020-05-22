/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"

#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/refcounting_hash_map.h"
#include "tensorflow/compiler/xla/service/collective_ops_utils.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/dynamic_annotations.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/stream_executor.h"

namespace se = ::stream_executor;

namespace xla {
namespace cpu {
namespace runtime {

XfeedManager* GetXfeedManager(int device_ordinal) {
  static auto* managers = new absl::flat_hash_map<int, XfeedManager*>();
  static absl::Mutex* mutex = new absl::Mutex();

  absl::MutexLock lock(mutex);
  auto it = managers->find(device_ordinal);
  if (it == managers->end()) {
    it = managers->emplace(device_ordinal, new XfeedManager()).first;
  }
  return it->second;
}

extern const char* const kEigenMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenMatMulF16";
extern const char* const kEigenMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenMatMulF32";
extern const char* const kEigenMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenMatMulF64";
extern const char* const kEigenMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenMatMulS32";
extern const char* const kMKLConvF32SymbolName = "__xla_cpu_runtime_MKLConvF32";
extern const char* const kMKLMatMulF32SymbolName =
    "__xla_cpu_runtime_MKLMatMulF32";
extern const char* const kMKLMatMulF64SymbolName =
    "__xla_cpu_runtime_MKLMatMulF64";
extern const char* const kMKLSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_MKLSingleThreadedMatMulF32";
extern const char* const kMKLSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_MKLSingleThreadedMatMulF64";
extern const char* const kEigenConvF16SymbolName =
    "__xla_cpu_runtime_EigenConvF16";
extern const char* const kEigenConvF32SymbolName =
    "__xla_cpu_runtime_EigenConvF32";
extern const char* const kEigenFftSymbolName = "__xla_cpu_runtime_EigenFft";
extern const char* const kEigenSingleThreadedFftSymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedFft";
extern const char* const kEigenSingleThreadedMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF16";
extern const char* const kEigenSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF32";
extern const char* const kEigenSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF64";
extern const char* const kEigenSingleThreadedMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulS32";
extern const char* const kEigenSingleThreadedConvF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConvF16";
extern const char* const kEigenSingleThreadedConvF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConvF32";
extern const char* const kAcquireInfeedBufferForDequeueSymbolName =
    "__xla_cpu_runtime_AcquireInfeedBufferForDequeue";
extern const char* const kReleaseInfeedBufferAfterDequeueSymbolName =
    "__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue";
extern const char* const kAcquireOutfeedBufferForPopulationSymbolName =
    "__xla_cpu_runtime_AcquireOutfeedBufferForPopulation";
extern const char* const kReleaseOutfeedBufferAfterPopulationSymbolName =
    "__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation";
extern const char* const kParallelForkJoinSymbolName =
    "__xla_cpu_runtime_ParallelForkJoin";
extern const char* const kKeyValueSortSymbolName =
    "__xla_cpu_runtime_KeyValueSort";
extern const char* const kTracingStartSymbolName =
    "__xla_cpu_runtime_TracingStart";
extern const char* const kTracingEndSymbolName = "__xla_cpu_runtime_TracingEnd";
extern const char* const kXlaCpuRuntimeSymbolNamePrefix = "__xla_cpu_runtime_";
extern const char* const kAllReduceSymbolName = "__xla_cpu_runtime_AllReduce";
extern const char* const kCollectivePermuteSymbolName =
    "__xla_cpu_runtime_CollectivePermute";
extern const char* const kReplicaIdSymbolName = "__xla_cpu_runtime_ReplicaId";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

namespace {

// Inverses the encoding of a Shape protobuf into an LLVM global variable.
xla::StatusOr<xla::Shape> DecodeSelfDescribingShapeConstant(
    const void* shape_ptr, xla::int32 size_bytes) {
  xla::ShapeProto shape_proto;
  if (!shape_proto.ParseFromArray(shape_ptr, size_bytes)) {
    return tensorflow::errors::Internal("Failed parsing the shape proto");
  }
  xla::Shape shape(shape_proto);
  auto status = xla::ShapeUtil::ValidateShape(shape);
  if (!status.ok()) {
    return status;
  }
  return std::move(shape);
}

tensorflow::string ShapeString(const void* shape_ptr, xla::int32 shape_length) {
  xla::StatusOr<xla::Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  if (shape.ok()) {
    return xla::ShapeUtil::HumanStringWithLayout(shape.ValueOrDie());
  }
  return "<invalid shape>";
}

}  // namespace

extern "C" {

TF_ATTRIBUTE_NO_SANITIZE_MEMORY xla::int64 __xla_cpu_runtime_TracingStart(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    const char* name) {
  VLOG(3) << "TracingStart " << name;
  return tensorflow::profiler::TraceMe::ActivityStart(name);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_TracingEnd(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    xla::int64 id) {
  VLOG(3) << "TracingEnd " << id;
  tensorflow::profiler::TraceMe::ActivityEnd(id);
}

}  // extern "C"

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void*
__xla_cpu_runtime_AcquireInfeedBufferForDequeue(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    const void* shape, xla::int32 shape_length) {
  int device_ordinal =
      run_options ? run_options->stream()->parent()->device_ordinal() : 0;

  VLOG(2) << "AcquireInfeedBufferForDequeue: "
          << ShapeString(shape, shape_length) << " on stream executor "
          << device_ordinal;

  xla::cpu::runtime::XfeedManager* xfeed =
      xla::cpu::runtime::GetXfeedManager(device_ordinal);
  // Wait until there's a buffer to dequeue.
  xla::cpu::runtime::XfeedBuffer* buffer =
      xfeed->infeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length)
      << "XLA program infeed request buffer size " << buffer_length
      << " did not match the runtime's infed buffer length " << buffer->length()
      << "; program reports desired shape: "
      << ShapeString(shape, shape_length);
  return buffer->data();
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_ReleaseInfeedBufferAfterDequeue(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    void* buffer_ptr, const void* shape_ptr, xla::int32 shape_length) {
  int device_ordinal =
      run_options ? run_options->stream()->parent()->device_ordinal() : 0;

  VLOG(2) << "ReleaseInfeedBufferAfterDeque: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  xla::cpu::runtime::XfeedManager* xfeed =
      xla::cpu::runtime::GetXfeedManager(device_ordinal);
  xla::StatusOr<xla::Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  xfeed->infeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr,
                                        std::move(shape));
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void*
__xla_cpu_runtime_AcquireOutfeedBufferForPopulation(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    const void* shape_ptr, xla::int32 shape_length) {
  int device_ordinal =
      run_options ? run_options->stream()->parent()->device_ordinal() : 0;

  VLOG(2) << "AcquireOutfeedBufferForPopulation: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  xla::cpu::runtime::XfeedManager* xfeed =
      xla::cpu::runtime::GetXfeedManager(device_ordinal);
  // Wait until there's a buffer to dequeue.
  xla::cpu::runtime::XfeedBuffer* buffer =
      xfeed->outfeed()->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), buffer_length)
      << "XLA program outfeed request buffer size " << buffer_length
      << " did not match the runtime's outfeed buffer length "
      << buffer->length() << "; program reports outfed shape: "
      << ShapeString(shape_ptr, shape_length);
  return buffer->data();
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void
__xla_cpu_runtime_ReleaseOutfeedBufferAfterPopulation(
    const xla::ExecutableRunOptions* run_options, xla::int32 buffer_length,
    void* buffer_ptr, const void* shape_ptr, xla::int32 shape_length) {
  int device_ordinal =
      run_options ? run_options->stream()->parent()->device_ordinal() : 0;

  VLOG(2) << "ReleaseOutfeedBufferAfterPopulation: "
          << ShapeString(shape_ptr, shape_length) << " on stream executor "
          << device_ordinal;

  xla::cpu::runtime::XfeedManager* xfeed =
      xla::cpu::runtime::GetXfeedManager(device_ordinal);
  xla::StatusOr<xla::Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  xfeed->outfeed()->ReleaseCurrentBuffer(buffer_length, buffer_ptr,
                                         std::move(shape));
}

namespace {

class CpuCollectivePermuteRendezvous
    : public xla::Rendezvous<xla::CollectivePermuteParticipantData,
                             std::nullptr_t> {
 public:
  explicit CpuCollectivePermuteRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<xla::CollectivePermuteParticipantData, std::nullptr_t>(
            k) {}

 protected:
  xla::StatusOr<ParticipantImplOutput> SubmitParticipantImpl(
      const xla::CollectivePermuteParticipantData& participant) override {
    bool primary = InitializationBarrier();

    // Perform all copies from the primary thread.
    if (primary) {
      tensorflow::mutex_lock lock(mu_);

      std::map<int, int> replica_idx_to_participant_idx;
      for (int p_idx = 0; p_idx < participants_.size(); p_idx++) {
        replica_idx_to_participant_idx[participants_[p_idx].replica_id] = p_idx;
      }

      for (auto& p : participants_) {
        for (int dest_replica : p.replica_ids_to_copy_to) {
          auto& dest_p = participants_[xla::FindOrDie(
              replica_idx_to_participant_idx, dest_replica)];
          std::memcpy(dest_p.destination_data.opaque(), p.source_data.opaque(),
                      p.byte_size);

          // Each replica may be copied into only once.
          replica_idx_to_participant_idx.erase(dest_replica);
        }
      }

      // Zero out untouched participants.
      for (auto& replica_p : replica_idx_to_participant_idx) {
        auto& p = participants_[replica_p.second];
        std::memset(p.destination_data.opaque(), 0, p.byte_size);
      }
    }
    return ParticipantImplOutput{primary, /*custom_output=*/nullptr};
  }
};

class CpuAllReduceRendezvous
    : public xla::Rendezvous<xla::AllReduceParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllReduceRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<xla::AllReduceParticipantData, std::nullptr_t>(k) {}

 protected:
  xla::StatusOr<ParticipantImplOutput> SubmitParticipantImpl(
      const xla::AllReduceParticipantData& participant) override {
    xla::PrimitiveType datatype = participant.buffers.front().primitive_type;
    bool primary = InitializationBarrier();

    if (primary) {
      switch (datatype) {
        case xla::S8:
          DoAllReduce<xla::S8>(participant);
          break;
        case xla::PRED:
        case xla::U8:
          DoAllReduce<xla::U8>(participant);
          break;
        case xla::S32:
          DoAllReduce<xla::S32>(participant);
          break;
        case xla::U32:
          DoAllReduce<xla::U32>(participant);
          break;
        case xla::S64:
          DoAllReduce<xla::S64>(participant);
          break;
        case xla::U64:
          DoAllReduce<xla::U64>(participant);
          break;
        case xla::F16:
          DoAllReduce<xla::F16>(participant);
          break;
        case xla::F32:
          DoAllReduce<xla::F32>(participant);
          break;
        case xla::F64:
          DoAllReduce<xla::F64>(participant);
          break;
        default:
          LOG(FATAL) << "Unexpected datatype;";
      }
    }
    return ParticipantImplOutput{primary, /*custom_output=*/nullptr};
  }

 private:
  template <xla::PrimitiveType PT>
  void DoAllReduce(xla::AllReduceParticipantData participant) {
    using T = typename xla::primitive_util::PrimitiveTypeToNative<PT>::type;
    tensorflow::mutex_lock lock(mu_);
    CHECK(!participants_.empty());
    xla::ReductionKind reduction_kind = participant.reduction_kind;
    for (const auto& p : participants_) {
      CHECK(p.reduction_kind == reduction_kind);
    }
    int num_participants = participants_.size();

    // participant_idx -> buffer_idx -> buffer.
    std::vector<std::vector<absl::Span<T>>> input_buffers;
    std::vector<std::vector<absl::Span<T>>> output_buffers;
    input_buffers.reserve(num_participants);
    output_buffers.reserve(num_participants);
    const xla::AllReduceParticipantData& first_participant =
        participants_.front();

    int buffers_per_participant = first_participant.buffers.size();
    for (xla::AllReduceParticipantData& p : participants_) {
      CHECK_EQ(p.buffers.size(), buffers_per_participant);

      input_buffers.emplace_back();
      output_buffers.emplace_back();
      std::vector<absl::Span<T>>& participant_input_buffers =
          input_buffers.back();
      std::vector<absl::Span<T>>& participant_output_buffers =
          output_buffers.back();
      participant_input_buffers.reserve(p.buffers.size());
      participant_output_buffers.reserve(p.buffers.size());

      for (int buffer_idx = 0; buffer_idx < buffers_per_participant;
           buffer_idx++) {
        auto& participant_buffer = p.buffers[buffer_idx];
        participant_input_buffers.emplace_back(
            static_cast<T*>(participant_buffer.source_data.opaque()),
            participant_buffer.element_count);
        participant_output_buffers.emplace_back(
            static_cast<T*>(participant_buffer.destination_data.opaque()),
            participant_buffer.element_count);
        CHECK_EQ(participant_buffer.element_count,
                 first_participant.buffers[buffer_idx].element_count);
      }
    }

    for (int buffer_idx = 0; buffer_idx < buffers_per_participant;
         buffer_idx++) {
      int element_count = first_participant.buffers[buffer_idx].element_count;
      for (int idx = 0; idx < element_count; idx++) {
        T out = GetInitialValue<T>(reduction_kind);
        for (int participant_idx = 0; participant_idx < participants_.size();
             participant_idx++) {
          out = PerformReductionStep<T>(
              reduction_kind, out,
              input_buffers[participant_idx][buffer_idx][idx]);
        }
        for (int participant_idx = 0; participant_idx < participants_.size();
             participant_idx++) {
          output_buffers[participant_idx][buffer_idx][idx] = out;
        }
      }
    }
  }

  template <typename T>
  T GetInitialValue(xla::ReductionKind reduction_kind) {
    switch (reduction_kind) {
      case xla::ReductionKind::SUM:
        return static_cast<T>(0);
      case xla::ReductionKind::PRODUCT:
        return static_cast<T>(1);
      case xla::ReductionKind::MIN:
        return std::numeric_limits<T>::max();
      case xla::ReductionKind::MAX:
        return std::numeric_limits<T>::min();
    }
  }

  template <typename T>
  T PerformReductionStep(xla::ReductionKind reduction_kind, T a, T b) {
    switch (reduction_kind) {
      case xla::ReductionKind::SUM:
        return a + b;
      case xla::ReductionKind::PRODUCT:
        return a * b;
      case xla::ReductionKind::MIN:
        return std::min(a, b);
      case xla::ReductionKind::MAX:
        return std::max(a, b);
    }
  }
};

xla::RefcountingHashMap<xla::RendezvousKey, CpuAllReduceRendezvous>&
GlobalAllReduceRendezvousMap() {
  static auto& m =
      *new xla::RefcountingHashMap<xla::RendezvousKey, CpuAllReduceRendezvous>;
  return m;
}

xla::RefcountingHashMap<xla::RendezvousKey, CpuCollectivePermuteRendezvous>&
GlobalCollectivePermuteRendezvousMap() {
  static auto& m = *new xla::RefcountingHashMap<xla::RendezvousKey,
                                                CpuCollectivePermuteRendezvous>;
  return m;
}

int GetDeviceOrdinal(const xla::ExecutableRunOptions* run_options) {
  if (run_options->stream()) {
    return run_options->stream()->parent()->device_ordinal();
  } else {
    return run_options->device_ordinal();
  }
}

xla::RendezvousKey GetRendezvousKey(
    const xla::ExecutableRunOptions* run_options,
    std::vector<xla::ReplicaGroup> group, xla::int32 channel_id_present,
    xla::int64 op_id) {
  const xla::DeviceAssignment& device_assignment =
      *run_options->device_assignment();
  xla::int32 replica_count = device_assignment.replica_count();
  int device_ordinal = GetDeviceOrdinal(run_options);
  CHECK_EQ(device_assignment.computation_count(), 1);
  std::vector<xla::int64> participating_replicas =
      xla::GetParticipatingReplicas(xla::GlobalDeviceId(device_ordinal), group,
                                    replica_count,
                                    *run_options->device_assignment())
          .ValueOrDie();
  xla::RendezvousKey::CollectiveOpKind op_kind =
      channel_id_present ? xla::RendezvousKey::kCrossModule
                         : xla::RendezvousKey::kCrossReplica;
  std::vector<xla::GlobalDeviceId> participating_devices;
  participating_devices.reserve(participating_replicas.size());
  for (xla::int64 replica : participating_replicas) {
    participating_devices.push_back(
        xla::GlobalDeviceId(device_assignment(replica, 0)));
  }
  return xla::RendezvousKey{
      run_options->run_id(), std::move(participating_devices),
      static_cast<int>(participating_replicas.size()), op_kind, op_id};
}

}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_AllReduce(
    const xla::ExecutableRunOptions* run_options,
    const void* replica_groups_str, xla::int32 replica_groups_str_size,
    xla::int32 channel_id_present, xla::int64 op_id, xla::int32 reduction_kind,
    const void* shape_ptr, xla::int32 shape_length, xla::int32 num_buffers,
    void** input_buffers, void** output_buffers) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  absl::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<xla::ReplicaGroup> group =
      xla::ParseReplicaGroupsOnly(replica_groups_serialized).ValueOrDie();
  xla::RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, group, channel_id_present, op_id);
  auto shape_str = ShapeString(shape_ptr, shape_length);
  VLOG(2) << "All-reduce input/output shape : " << shape_str;

  xla::Shape shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length).ValueOrDie();

  CHECK((num_buffers > 1 && shape.IsTuple()) ||
        (num_buffers == 1 && xla::LayoutUtil::IsDenseArray(shape)));

  xla::AllReduceParticipantData participant(rendezvous_key);
  participant.device_ordinal = device_ordinal;
  participant.stream = run_options->stream();
  participant.reduction_kind = static_cast<xla::ReductionKind>(reduction_kind);
  for (int i = 0; i < num_buffers; i++) {
    xla::Shape subshape = num_buffers == 1 ? shape : shape.tuple_shapes(i);
    xla::AllReduceParticipantData::Buffer buffer;
    buffer.element_count = xla::ShapeUtil::ElementsIn(subshape);
    buffer.primitive_type = subshape.element_type();
    buffer.source_data = se::DeviceMemoryBase(
        input_buffers[i], xla::ShapeUtil::ByteSizeOf(subshape));
    buffer.destination_data = se::DeviceMemoryBase(
        output_buffers[i], xla::ShapeUtil::ByteSizeOf(subshape));
    participant.buffers.push_back(buffer);
  }

  auto make_cpu_rendezvous = [](const xla::RendezvousKey& k) {
    return absl::make_unique<CpuAllReduceRendezvous>(k);
  };

  TF_CHECK_OK(CpuAllReduceRendezvous::SubmitParticipant(
                  [&] {
                    return GlobalAllReduceRendezvousMap().GetOrCreateIfAbsent(
                        rendezvous_key, make_cpu_rendezvous);
                  },
                  participant)
                  .status());
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ReplicaId(
    const xla::ExecutableRunOptions* run_options, void* output_buffer) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  xla::int32 replica_id = run_options->device_assignment()
                              ->ReplicaIdForDeviceOrdinal(device_ordinal)
                              .ValueOrDie();
  std::memcpy(output_buffer, &replica_id, 4);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_CollectivePermute(
    const xla::ExecutableRunOptions* run_options, xla::int32 channel_id_present,
    xla::int64 op_id, xla::int32 byte_size, void* input_buffer,
    void* output_buffer, const void* source_target_pairs,
    xla::int32 source_target_pairs_size) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  absl::string_view source_target_pairs_serialized(
      static_cast<const char*>(source_target_pairs), source_target_pairs_size);
  auto pairs = absl::StrSplit(source_target_pairs_serialized, ',');
  xla::int32 replica_id = run_options->device_assignment()
                              ->ReplicaIdForDeviceOrdinal(device_ordinal)
                              .ValueOrDie();
  std::vector<int> copy_to;
  for (auto& p : pairs) {
    std::vector<std::string> mapping = absl::StrSplit(p, '=');
    CHECK_EQ(mapping.size(), 2);
    int from = std::stoi(mapping[0]);
    int to = std::stoi(mapping[1]);
    if (from == replica_id) {
      copy_to.push_back(to);
    }
  }
  xla::RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, {}, channel_id_present, op_id);

  xla::CollectivePermuteParticipantData participant(rendezvous_key);
  participant.replica_id = replica_id;
  participant.device_ordinal = device_ordinal;
  participant.stream = run_options->stream();
  participant.source_data = se::DeviceMemoryBase(input_buffer, byte_size);
  participant.destination_data = se::DeviceMemoryBase(output_buffer, byte_size);
  participant.replica_ids_to_copy_to = copy_to;
  participant.byte_size = byte_size;

  auto make_cpu_rendezvous = [](const xla::RendezvousKey& k) {
    return absl::make_unique<CpuCollectivePermuteRendezvous>(k);
  };
  TF_CHECK_OK(
      CpuCollectivePermuteRendezvous::SubmitParticipant(
          [&] {
            return GlobalCollectivePermuteRendezvousMap().GetOrCreateIfAbsent(
                rendezvous_key, make_cpu_rendezvous);
          },
          participant)
          .status());
}
