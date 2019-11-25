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

class CpuAllReduceRendezvous : public xla::Rendezvous<std::nullptr_t> {
 public:
  explicit CpuAllReduceRendezvous(const xla::RendezvousKey& k)
      : xla::Rendezvous<std::nullptr_t>(k) {}

 protected:
  xla::StatusOr<std::pair<std::nullptr_t, bool>> SubmitParticipantImpl(
      xla::AllReduceParticipantData participant) override {
    xla::PrimitiveType datatype = participant.primitive_type;
    bool primary = [&] {
      tensorflow::mutex_lock lock(mu_);
      if (!initialized_) {
        initialized_ = true;
        return true;
      }
      return false;
    }();

    if (primary) {
      switch (datatype) {
        case xla::S8:
          DoAllReduce<xla::S8>(participant);
          break;
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

    // First element is a dummy value.
    return std::make_pair(nullptr, primary);
  }

 private:
  template <xla::PrimitiveType PT>
  void DoAllReduce(xla::AllReduceParticipantData participant) {
    using T = typename xla::primitive_util::PrimitiveTypeToNative<PT>::type;
    tensorflow::mutex_lock lock(mu_);
    CHECK(!participants_.empty());
    xla::int64 element_count = participant.element_count;
    xla::ReductionKind reduction_kind = participant.reduction_kind;
    for (const auto& p : participants_) {
      CHECK_EQ(p.element_count, element_count);
      CHECK(p.reduction_kind == reduction_kind);
    }

    std::vector<absl::Span<T>> input_buffers;
    std::vector<absl::Span<T>> output_buffers;
    input_buffers.reserve(participants_.size());
    output_buffers.reserve(participants_.size());

    for (auto& p : participants_) {
      input_buffers.emplace_back(static_cast<T*>(p.source_data.opaque()),
                                 element_count);
      output_buffers.emplace_back(static_cast<T*>(p.destination_data.opaque()),
                                  element_count);
    }

    auto compute = [reduction_kind](T a, T b) -> T {
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
    };

    for (int idx = 0; idx < element_count; idx++) {
      T out = [&]() -> T {
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
      }();

      for (auto& input : input_buffers) {
        out = compute(out, input[idx]);
      }
      for (auto& output : output_buffers) {
        output[idx] = out;
      }
    }
  }
};

xla::RefcountingHashMap<xla::RendezvousKey, CpuAllReduceRendezvous>&
GlobalRendezvousMap() {
  static auto& m =
      *new xla::RefcountingHashMap<xla::RendezvousKey, CpuAllReduceRendezvous>(
          [](const xla::RendezvousKey& k) {
            return absl::make_unique<CpuAllReduceRendezvous>(k);
          });
  return m;
}

}  // namespace

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_AllReduce(
    const xla::ExecutableRunOptions* run_options,
    const void* replica_groups_str, xla::int32 replica_groups_str_size,
    xla::int32 channel_id_present, xla::int64 op_id, xla::int32 reduction_kind,
    const void* shape_ptr, xla::int32 shape_length, void* input_buffer,
    void* output_buffer) {
  absl::string_view replica_groups_serialized(
      static_cast<const char*>(replica_groups_str), replica_groups_str_size);

  // FIXME(cheshire): avoid repetition w/__xla_cpu_runtime_ReplicaId.
  int device_ordinal = [&] {
    if (run_options->stream()) {
      return run_options->stream()->parent()->device_ordinal();
    } else {
      return run_options->device_ordinal();
    }
  }();

  std::vector<xla::ReplicaGroup> group =
      xla::ParseReplicaGroupsOnly(replica_groups_serialized).ValueOrDie();
  xla::int32 replica_count = run_options->device_assignment()->replica_count();
  std::vector<xla::int64> participating_replicas_vec =
      xla::GetParticipatingReplicas(device_ordinal, group, replica_count,
                                    *run_options->device_assignment())
          .ValueOrDie();

  xla::RendezvousKey::CollectiveOpKind op_kind =
      channel_id_present ? xla::RendezvousKey::kCrossModule
                         : xla::RendezvousKey::kCrossReplica;
  xla::RendezvousKey rendezvous_key(run_options->run_id(),
                                    participating_replicas_vec, op_kind, op_id);

  std::shared_ptr<CpuAllReduceRendezvous> rendezvous =
      GlobalRendezvousMap()[rendezvous_key];

  auto shape_str = ShapeString(shape_ptr, shape_length);
  VLOG(2) << "All-reduce input/output shape : " << shape_str;

  xla::Shape shape =
      DecodeSelfDescribingShapeConstant(shape_ptr, shape_length).ValueOrDie();

  xla::AllReduceParticipantData participant(rendezvous_key);

  CHECK_LE(shape.dimensions_size(), 1);
  participant.element_count = xla::ShapeUtil::ElementsIn(shape);
  participant.device_ordinal = device_ordinal;
  participant.primitive_type = shape.element_type();
  participant.stream = run_options->stream();

  se::DeviceMemoryBase input(input_buffer, xla::ShapeUtil::ByteSizeOf(shape));
  se::DeviceMemoryBase output(output_buffer, xla::ShapeUtil::ByteSizeOf(shape));
  participant.source_data = input;
  participant.destination_data = output;
  participant.reduction_kind = static_cast<xla::ReductionKind>(reduction_kind);

  auto p = rendezvous->SubmitParticipant(participant).ValueOrDie();
  std::shared_ptr<tensorflow::BlockingCounter> blocking_counter = p.second;
  blocking_counter->DecrementCount();
  xla::WaitAndLogIfStuck(blocking_counter.get(), [&] {
    return absl::StrFormat(
        "participant waiting for all threads to drop their reference to the "
        "rendezvous: %s",
        rendezvous_key.ToString());
  });

  rendezvous.reset();
}

TF_ATTRIBUTE_NO_SANITIZE_MEMORY void __xla_cpu_runtime_ReplicaId(
    const xla::ExecutableRunOptions* run_options, void* output_buffer) {
  int device_ordinal = [&]() {
    if (run_options->stream()) {
      return run_options->stream()->parent()->device_ordinal();
    } else {
      return run_options->device_ordinal();
    }
  }();

  xla::int32 replica_id = run_options->device_assignment()
                              ->ReplicaIdForDeviceOrdinal(device_ordinal)
                              .ValueOrDie();
  std::memcpy(output_buffer, &replica_id, 4);
}
