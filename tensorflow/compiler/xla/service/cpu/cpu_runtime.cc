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

#include <cstdarg>
#include <cstddef>
#include <cstring>
#include <functional>
#include <limits>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
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
extern const char* const kEigenMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenMatMulC64";
extern const char* const kEigenMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenMatMulC128";
extern const char* const kEigenMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenMatMulS32";
extern const char* const kMKLConv2DF32SymbolName =
    "__xla_cpu_runtime_MKLConv2DF32";
extern const char* const kMKLMatMulF32SymbolName =
    "__xla_cpu_runtime_MKLMatMulF32";
extern const char* const kMKLMatMulF64SymbolName =
    "__xla_cpu_runtime_MKLMatMulF64";
extern const char* const kMKLSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_MKLSingleThreadedMatMulF32";
extern const char* const kMKLSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_MKLSingleThreadedMatMulF64";
extern const char* const kEigenConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenConv2DF16";
extern const char* const kEigenConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenConv2DF32";
extern const char* const kEigenConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenConv3DF16";
extern const char* const kEigenConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenConv3DF32";
extern const char* const kEigenFftSymbolName = "__xla_cpu_runtime_EigenFft";
extern const char* const kEigenSingleThreadedFftSymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedFft";
extern const char* const kEigenSingleThreadedMatMulF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF16";
extern const char* const kEigenSingleThreadedMatMulF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF32";
extern const char* const kEigenSingleThreadedMatMulF64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulF64";
extern const char* const kEigenSingleThreadedMatMulC64SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC64";
extern const char* const kEigenSingleThreadedMatMulC128SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulC128";
extern const char* const kEigenSingleThreadedMatMulS32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedMatMulS32";
extern const char* const kEigenSingleThreadedConv2DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF16";
extern const char* const kEigenSingleThreadedConv2DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv2DF32";
extern const char* const kEigenSingleThreadedConv3DF16SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF16";
extern const char* const kEigenSingleThreadedConv3DF32SymbolName =
    "__xla_cpu_runtime_EigenSingleThreadedConv3DF32";
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
extern const char* const kPrintfToStderrSymbolName =
    "__xla_cpu_runtime_PrintfToStderr";
extern const char* const kStatusIsSuccessSymbolName =
    "__xla_cpu_runtime_StatusIsSuccess";
extern const char* const kKeyValueSortSymbolName =
    "__xla_cpu_runtime_KeyValueSort";
extern const char* const kTopKF32SymbolName = "__xla_cpu_runtime_TopKF32";
extern const char* const kTracingStartSymbolName =
    "__xla_cpu_runtime_TracingStart";
extern const char* const kTracingEndSymbolName = "__xla_cpu_runtime_TracingEnd";
extern const char* const kXlaCpuRuntimeSymbolNamePrefix = "__xla_cpu_runtime_";
extern const char* const kAllReduceSymbolName = "__xla_cpu_runtime_AllReduce";
extern const char* const kAllToAllSymbolName = "__xla_cpu_runtime_AllToAll";
extern const char* const kCollectivePermuteSymbolName =
    "__xla_cpu_runtime_CollectivePermute";
extern const char* const kReplicaIdSymbolName = "__xla_cpu_runtime_ReplicaId";

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

namespace {

struct CollectivePermuteParticipantData : xla::ParticipantData {
  CollectivePermuteParticipantData(const xla::RendezvousKey& rendezvous_key_p,
                                   int64_t device_ordinal_p,
                                   se::Stream* stream_p)
      : ParticipantData(rendezvous_key_p),
        device_ordinal(device_ordinal_p),
        stream(stream_p) {}

  int64_t device_ordinal;
  se::Stream* stream;
  int replica_id;
  se::DeviceMemoryBase source_data;
  se::DeviceMemoryBase destination_data;
  int64_t byte_size;
  std::vector<int> replica_ids_to_copy_to;

  std::string ToString() const override {
    return absl::StrFormat(
        "CollectivePermuteParticipantData{replica_id=%d, "
        "source_data=%p, destination_data=%p, byte_size=%d, "
        "replica_ids_to_copy_to=[%s], device_ordinal=%d, stream=%p}",
        replica_id, source_data.opaque(), destination_data.opaque(), byte_size,
        absl::StrJoin(replica_ids_to_copy_to, ", "), device_ordinal, stream);
  }
};

struct AllToAllParticipantData : xla::ParticipantData {
  AllToAllParticipantData(const xla::RendezvousKey& rendezvous_key_p,
                          int64_t device_ordinal_p, se::Stream* stream_p)
      : ParticipantData(rendezvous_key_p),
        device_ordinal(device_ordinal_p),
        stream(stream_p) {}

  int64_t device_ordinal;
  se::Stream* stream;
  std::vector<se::DeviceMemoryBase> source_buffers;
  std::vector<se::DeviceMemoryBase> destination_buffers;
  int replica_id;

  // Replica ids participating in AllToAll, concatenation happens in the order
  // of appearence.
  std::vector<int> replica_ids_to_copy_to;

  std::string ToString() const override {
    auto addr_formatter = [](std::string* out,
                             const se::DeviceMemoryBase& mem) {
      absl::StrAppend(out, absl::StrFormat("%p", mem.opaque()));
    };
    return absl::StrFormat(
        "AllToAllParticipantData{replica_id=%d, "
        "replica_ids_to_copy_to=[%s], source_buffers=[%s], "
        "destination_buffers=[%s], device_ordinal=%d, stream=%p}",
        replica_id, absl::StrJoin(replica_ids_to_copy_to, ", "),
        absl::StrJoin(source_buffers, ", ", addr_formatter),
        absl::StrJoin(destination_buffers, ", ", addr_formatter),
        device_ordinal, stream);
  }
};

// Inverses the encoding of a Shape protobuf into an LLVM global variable.
xla::StatusOr<xla::Shape> DecodeSelfDescribingShapeConstant(
    const void* shape_ptr, int32_t size_bytes) {
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

tensorflow::string ShapeString(const void* shape_ptr, int32_t shape_length) {
  xla::StatusOr<xla::Shape> shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length);
  if (shape.ok()) {
    return xla::ShapeUtil::HumanStringWithLayout(shape.ValueOrDie());
  }
  return "<invalid shape>";
}

// TODO(zhangqiaorjc): Prefer to make callers set and use device_ordinal
// directly since callers may not have a Stream*.
int GetDeviceOrdinal(const xla::ExecutableRunOptions* run_options) {
  if (!run_options) {
    return 0;
  } else if (run_options->device_ordinal() != -1) {
    return run_options->device_ordinal();
  }
  return run_options->stream()->parent()->device_ordinal();
}

}  // namespace

extern "C" {

TF_ATTRIBUTE_NO_SANITIZE_MEMORY int __xla_cpu_runtime_PrintfToStderr(
    const char* format, ...) {
  VLOG(3) << "__xla_cpu_runtime_PrintfToStderr " << format;
  va_list args;
  va_start(args, format);
  int result = vfprintf(stderr, format, args);
  va_end(args);
  return result;
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY int64_t __xla_cpu_runtime_TracingStart(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr,
    const char* name) {
  VLOG(3) << "TracingStart " << name;
  return tensorflow::profiler::TraceMe::ActivityStart(name);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_TracingEnd(
    const void* /* xla::ExecutableRunOptions* */ run_options_ptr, int64_t id) {
  VLOG(3) << "TracingEnd " << id;
  tensorflow::profiler::TraceMe::ActivityEnd(id);
}

}  // extern "C"

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void*
__xla_cpu_runtime_AcquireInfeedBufferForDequeue(
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    const void* shape, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

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
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

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
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

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
    const xla::ExecutableRunOptions* run_options, int32_t buffer_length,
    void* buffer_ptr, const void* shape_ptr, int32_t shape_length) {
  int device_ordinal = GetDeviceOrdinal(run_options);

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

class CpuAllToAllRendezvous
    : public xla::Rendezvous<AllToAllParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllToAllRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<AllToAllParticipantData, std::nullptr_t>(k) {}

 protected:
  xla::StatusOr<std::nullptr_t> RunCollectiveOp(
      const AllToAllParticipantData& /*participant*/) override {
    bool is_primary = InitializationBarrier();

    if (is_primary) {
      tensorflow::mutex_lock lock(mu_);

      CHECK(!participants_.empty());
      CHECK(!participants_[0].source_buffers.empty());
      int expected_buffer_size = participants_[0].source_buffers[0].size();

      // Replica id -> position in participants_.
      absl::flat_hash_map<int, int> replica_id_map;

      for (int pos = 0; pos < participants_.size(); pos++) {
        const AllToAllParticipantData& p = participants_[pos];
        CHECK_EQ(p.source_buffers.size(), p.destination_buffers.size());
        CHECK_EQ(p.source_buffers.size(), participants_.size());
        for (int i = 0; i < p.source_buffers.size(); i++) {
          CHECK_EQ(p.destination_buffers[i].size(), expected_buffer_size);
          CHECK_EQ(p.source_buffers[i].size(), expected_buffer_size);
        }
        replica_id_map[p.replica_id] = pos;
      }

      const std::vector<int>& replica_ids_to_copy_to =
          participants_[0].replica_ids_to_copy_to;

      // Replica id -> rank
      absl::flat_hash_map<int, int> replica_ranks;
      for (int rank = 0; rank < replica_ids_to_copy_to.size(); ++rank) {
        int replica_id = replica_ids_to_copy_to[rank];
        replica_ranks[replica_id] = rank;
      }

      for (const AllToAllParticipantData& sender : participants_) {
        VLOG(3) << "Processing AllToAll participant: " << sender.ToString();

        int rank = xla::FindOrDie(replica_ranks, sender.replica_id);

        for (int i = 0; i < participants_.size(); ++i) {
          int replica_id = replica_ids_to_copy_to[i];
          int participant_num = xla::FindOrDie(replica_id_map, replica_id);
          AllToAllParticipantData& receiver = participants_[participant_num];

          std::memcpy(receiver.destination_buffers[rank].opaque(),
                      sender.source_buffers[i].opaque(), expected_buffer_size);
        }
      }
    }
    return nullptr;
  }
};

class CpuCollectivePermuteRendezvous
    : public xla::Rendezvous<CollectivePermuteParticipantData, std::nullptr_t> {
 public:
  explicit CpuCollectivePermuteRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<CollectivePermuteParticipantData, std::nullptr_t>(k) {}

 protected:
  xla::StatusOr<std::nullptr_t> RunCollectiveOp(
      const CollectivePermuteParticipantData& /*participant*/) override {
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
    return nullptr;
  }
};

class CpuAllReduceRendezvous
    : public xla::Rendezvous<xla::AllReduceParticipantData, std::nullptr_t> {
 public:
  explicit CpuAllReduceRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<xla::AllReduceParticipantData, std::nullptr_t>(k) {}

 protected:
  xla::StatusOr<std::nullptr_t> RunCollectiveOp(
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
    return nullptr;
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

xla::RefcountingHashMap<xla::RendezvousKey, CpuAllToAllRendezvous>&
GlobalAllToAllRendezvousMap() {
  static auto& m =
      *new xla::RefcountingHashMap<xla::RendezvousKey, CpuAllToAllRendezvous>;
  return m;
}

xla::RendezvousKey GetRendezvousKey(
    const xla::ExecutableRunOptions* run_options,
    std::vector<xla::ReplicaGroup> group, int32_t channel_id_present,
    int64_t op_id) {
  const xla::DeviceAssignment& device_assignment =
      *run_options->device_assignment();
  int device_ordinal = GetDeviceOrdinal(run_options);
  xla::RendezvousKey::CollectiveOpKind op_kind =
      channel_id_present ? xla::RendezvousKey::kCrossModule
                         : xla::RendezvousKey::kCrossReplica;
  std::vector<xla::GlobalDeviceId> participating_devices =
      xla::GetParticipatingDevices(xla::GlobalDeviceId(device_ordinal),
                                   device_assignment, group,
                                   xla::CollectiveOpGroupMode::kCrossReplica)
          .ValueOrDie();
  int num_local_participants = participating_devices.size();
  return xla::RendezvousKey{run_options->run_id(),
                            std::move(participating_devices),
                            num_local_participants, op_kind, op_id};
}

}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_AllToAll(
    const xla::ExecutableRunOptions* run_options, int32_t channel_id_present,
    int64_t op_id, const void* replica_groups_str,
    int32_t replica_groups_str_size, int32_t num_buffers, int64_t buffer_size,
    void** source_buffers, void** destination_buffers) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  int32_t replica_id =
      run_options->device_assignment()
          ->ReplicaIdForDevice(xla::GlobalDeviceId(device_ordinal))
          .ValueOrDie();
  absl::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);
  std::vector<xla::ReplicaGroup> group =
      xla::ParseReplicaGroupsOnly(replica_groups_serialized).ValueOrDie();
  xla::RendezvousKey rendezvous_key =
      GetRendezvousKey(run_options, group, channel_id_present, op_id);

  AllToAllParticipantData participant(rendezvous_key, device_ordinal,
                                      run_options->stream());
  participant.replica_id = replica_id;
  participant.replica_ids_to_copy_to =
      xla::GetParticipatingIDs(
          replica_id, run_options->device_assignment()->replica_count(), group)
          .ValueOrDie();
  for (int i = 0; i < num_buffers; i++) {
    participant.source_buffers.emplace_back(source_buffers[i], buffer_size);
    participant.destination_buffers.emplace_back(destination_buffers[i],
                                                 buffer_size);
  }
  auto make_cpu_rendezvous = [](const xla::RendezvousKey& k) {
    return absl::make_unique<CpuAllToAllRendezvous>(k);
  };
  TF_CHECK_OK(CpuAllToAllRendezvous::SubmitParticipant(
                  [&] {
                    return GlobalAllToAllRendezvousMap().GetOrCreateIfAbsent(
                        rendezvous_key, make_cpu_rendezvous);
                  },
                  participant)
                  .status());
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_AllReduce(
    const xla::ExecutableRunOptions* run_options,
    const void* replica_groups_str, int32_t replica_groups_str_size,
    int32_t channel_id_present, int64_t op_id, int32_t reduction_kind,
    const void* shape_ptr, int32_t shape_length, int32_t num_buffers,
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

  xla::AllReduceParticipantData participant(rendezvous_key, device_ordinal,
                                            run_options->stream());
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
  int32_t replica_id =
      run_options->device_assignment()
          ->ReplicaIdForDevice(xla::GlobalDeviceId(device_ordinal))
          .ValueOrDie();
  std::memcpy(output_buffer, &replica_id, 4);
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_CollectivePermute(
    const xla::ExecutableRunOptions* run_options, int32_t channel_id_present,
    int64_t op_id, int32_t byte_size, void* input_buffer, void* output_buffer,
    const void* source_target_pairs, int32_t source_target_pairs_size) {
  int device_ordinal = GetDeviceOrdinal(run_options);
  absl::string_view source_target_pairs_serialized(
      static_cast<const char*>(source_target_pairs), source_target_pairs_size);
  auto pairs = absl::StrSplit(source_target_pairs_serialized, ',');
  int32_t replica_id =
      run_options->device_assignment()
          ->ReplicaIdForDevice(xla::GlobalDeviceId(device_ordinal))
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

  CollectivePermuteParticipantData participant(rendezvous_key, device_ordinal,
                                               run_options->stream());
  participant.replica_id = replica_id;
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
