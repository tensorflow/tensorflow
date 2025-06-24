/* Copyright 2025 The OpenXLA Authors.
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

#include "xla/backends/gpu/collectives/nvshmem_communicator.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/gpus/cuda/include/cuda_bf16.h"
#include "third_party/gpus/cuda/include/cuda_fp16.h"
#include "third_party/nvshmem/nvshmemx.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nvshmem_collectives.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// NVSHMEM Utility Functions
//==-----------------------------------------------------------------------===//

size_t ToRealCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

//==-----------------------------------------------------------------------===//
// NVSHMEM Templated APIs
//==-----------------------------------------------------------------------===//

#define CALL_NVSHMEM_COLL(coll, TYPENAME, TYPE, OP, team, source_ptr,         \
                          dest_ptr, stream)                                   \
  do {                                                                        \
    if (nvshmemx_##TYPENAME##_##OP##_##coll##_on_stream(                      \
            team, (TYPE*)dest_ptr, (const TYPE*)source_ptr, count, stream) != \
        0) {                                                                  \
      return absl::InternalError("Nvshmem collective failed");                \
    }                                                                         \
  } while (0)

#define NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                     \
    coll, TYPENAME, TYPE, team, source_ptr, dest_ptr, count, stream,    \
    reduction_kind)                                                     \
  switch (reduction_kind) {                                             \
    case ReductionKind::SUM:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::MIN:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::MAX:                                            \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,  \
                        dest_ptr, stream);                              \
      break;                                                            \
    case ReductionKind::PRODUCT:                                        \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr, \
                        dest_ptr, stream);                              \
      break;                                                            \
    default:                                                            \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");    \
  }

#define NVSHMEM_REDUCTION_DATATYPE(coll, TYPENAME, TYPE, team, source_ptr, \
                                   dest_ptr, num_elements, gpu_stream,     \
                                   reduction_kind)                         \
  switch (reduction_kind) {                                                \
    case ReductionKind::SUM:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, sum, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::MIN:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, min, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::MAX:                                               \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, max, team, source_ptr,     \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    case ReductionKind::PRODUCT:                                           \
      CALL_NVSHMEM_COLL(reduce, TYPENAME, TYPE, prod, team, source_ptr,    \
                        dest_ptr, gpu_stream);                             \
      break;                                                               \
    default:                                                               \
      return absl::InternalError("Invalid NVSHMEM reduction kind.");       \
  }

#define CALL_NVSHMEM_REDUCTION_DATATYPE(TYPENAME, TYPE, team, gpu_stream,     \
                                        reduction_kind, dest_ptr, source_ptr, \
                                        count)                                \
  NVSHMEM_REDUCTION_DATATYPE(reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD,      \
                             (TYPE*)source_ptr, (TYPE*)dest_ptr, count,       \
                             gpu_stream, reduction_kind);
#define CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(TYPENAME, TYPE, team,        \
                                                gpu_stream, reduction_kind,  \
                                                dest_ptr, source_ptr, count) \
  NVSHMEM_BITWISE_REDUCTION_BITWISE_DATATYPE(                                \
      reduce, TYPENAME, TYPE, NVSHMEM_TEAM_WORLD, (TYPE*)source_ptr,         \
      (TYPE*)dest_ptr, count, gpu_stream, reduction_kind);

#define CALL_NVSHMEM_P2P(op, TYPENAME, TYPE, pe, source_ptr, dest_ptr,    \
                         num_elements, stream)                            \
  nvshmemx_##TYPENAME##_##op##_nbi_on_stream(                             \
      (TYPE*)dest_ptr, (const TYPE*)source_ptr, num_elements, pe.value(), \
      se::gpu::AsGpuStreamValue(stream))

//==-----------------------------------------------------------------------===//
// NVSHMEM Communicator
//==-----------------------------------------------------------------------===//

NvshmemCommunicator::NvshmemCommunicator(NvshmemCollectives* collectives)
    : collectives_(collectives) {
  VLOG(1) << "Created " << *this;
}

absl::Status NvshmemCommunicator::Abort() {
  VLOG(1) << "Abort NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  aborted_ = true;
  // Call nvshmem_global_exit with a non-zero return code
  // to abort the program.
  nvshmem_global_exit(1);
  return absl::OkStatus();
}

absl::Status NvshmemCommunicator::Barrier(
    const Communicator::Executor& executor) {
  VLOG(1) << "Barrier NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return FailedPrecondition("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  auto gpu_stream = se::gpu::AsGpuStreamValue(stream);

  if (nvshmemx_barrier_on_stream(NVSHMEMX_TEAM_NODE, gpu_stream) != 0) {
    return absl::InternalError("Nvshmem team barrier failed.");
  }
  return absl::OkStatus();
}
absl::StatusOr<size_t> NvshmemCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  int32_t count = 0;
  count = nvshmem_team_n_pes(NVSHMEMX_TEAM_NODE);
  if (count < 0) {
    return absl::InvalidArgumentError(
        "NvshmemCommunicator::NumRanks invalid team.");
  }
  return count;
}

absl::StatusOr<size_t> NvshmemCommunicator::CurrentRank() {
  VLOG(5) << "Get current rank in NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  int32_t rank = 0;
  rank = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  if (rank < 0) {
    return absl::InvalidArgumentError(
        "NvshmemCommunicator::NumRanks invalid team.");
  }
  return rank;
}

tsl::AsyncValueRef<NvshmemCommunicator::Event> NvshmemCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  void* dest_ptr = send_buffer.opaque();
  void* source_ptr = recv_buffer.opaque();
  count = ToRealCount(dtype, count);
  VLOG(3) << absl::StreamFormat(
      "Launch NVSHMEM AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=node; "
      "team=%d;"
      "stream=%p",
      nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), NVSHMEMX_TEAM_NODE, stream);

  switch (dtype) {
    case PrimitiveType::F64: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          double, double, NVSHMEMX_TEAM_NODE, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::F16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          half, __half, NVSHMEMX_TEAM_NODE, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::F32: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          float, float, NVSHMEMX_TEAM_NODE, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::BF16: {
      CALL_NVSHMEM_REDUCTION_DATATYPE(
          bfloat16, __nv_bfloat16, NVSHMEMX_TEAM_NODE,
          se::gpu::AsGpuStreamValue(stream), reduction_kind, dest_ptr,
          source_ptr, count);
      break;
    }
    case PrimitiveType::S32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          int32, int32_t, NVSHMEMX_TEAM_NODE, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::S64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          int64, int64_t, NVSHMEMX_TEAM_NODE, se::gpu::AsGpuStreamValue(stream),
          reduction_kind, dest_ptr, source_ptr, count);
      break;
    }
    case PrimitiveType::U32: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint32, uint32_t, NVSHMEMX_TEAM_NODE,
          se::gpu::AsGpuStreamValue(stream), reduction_kind, dest_ptr,
          source_ptr, count);
      break;
    }
    case PrimitiveType::U64: {
      CALL_NVSHMEM_BITWISE_REDUCTION_DATATYPE(
          uint64, uint64_t, NVSHMEMX_TEAM_NODE,
          se::gpu::AsGpuStreamValue(stream), reduction_kind, dest_ptr,
          source_ptr, count);
      break;
    }
    default:
      return absl::InternalError("Invalid Nvshmem reduction type.");
  }
  return OkEvent();
}

std::string NvshmemCommunicator::ToString() const {
  return absl::StrFormat("NvshmemCommunicator(nvshmem_team_t=%d)",
                         NVSHMEMX_TEAM_NODE);
}

absl::StatusOr<se::Stream*> NvshmemCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

// Helper function to get the size of a primitive type in bytes
size_t GetPrimitiveTypeSize(PrimitiveType type) {
  switch (type) {
    case PrimitiveType::F16:
    case PrimitiveType::BF16:
    case PrimitiveType::S16:
    case PrimitiveType::U16:
      return 2;
    case PrimitiveType::F32:
    case PrimitiveType::S32:
    case PrimitiveType::U32:
      return 4;
    case PrimitiveType::F64:
    case PrimitiveType::S64:
    case PrimitiveType::U64:
      return 8;
    default:
      LOG(FATAL) << "Unsupported primitive type: " << static_cast<int>(type);
      return 0;
  }
}

// Performs point-to-point communication between two ranks using NVSHMEM.
// This is a helper function used by both Send and Recv operations to handle
// the actual data transfer between peers.
absl::Status NvshmemCommunicator::P2P(absl::string_view op_name,
                                      PrimitiveType type,
                                      se::DeviceMemoryBase recv_buffer,
                                      se::DeviceMemoryBase send_buffer,
                                      size_t count, RankId peer,
                                      const Executor& executor) {
  if (!op_name.empty() && op_name != "put" && op_name != "get") {
    return absl::InternalError(
        absl::StrFormat("Unsupported NVSHMEM operation: %s", op_name));
  }

  void* source_ptr = send_buffer.opaque();
  void* dest_ptr = recv_buffer.opaque();

  // Register the source buffer since it's allocated in device memory (not with
  // nvshmem_malloc). This is required for NVSHMEM to access the buffer during
  // P2P operations.
  TF_RETURN_IF_ERROR(
      RegisterBuffer(source_ptr, count * GetPrimitiveTypeSize(type)));

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  switch (type) {
    case PrimitiveType::F64:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, double, double, peer, source_ptr, dest_ptr, count,
                         stream);
      } else {
        CALL_NVSHMEM_P2P(get, double, double, peer, source_ptr, dest_ptr, count,
                         stream);
      }
      break;
    case PrimitiveType::F32:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, float, float, peer, source_ptr, dest_ptr, count,
                         stream);
      } else {
        CALL_NVSHMEM_P2P(get, float, float, peer, source_ptr, dest_ptr, count,
                         stream);
      }
      break;
    case PrimitiveType::F16:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, half, __half, peer, source_ptr, dest_ptr, count,
                         stream);
      } else {
        CALL_NVSHMEM_P2P(get, half, __half, peer, source_ptr, dest_ptr, count,
                         stream);
      }
      break;
    case PrimitiveType::BF16:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, bfloat16, __nv_bfloat16, peer, source_ptr,
                         dest_ptr, count, stream);
      } else {
        CALL_NVSHMEM_P2P(get, bfloat16, __nv_bfloat16, peer, source_ptr,
                         dest_ptr, count, stream);
      }
      break;
    case PrimitiveType::S32:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, int32, int32_t, peer, source_ptr, dest_ptr, count,
                         stream);
      } else {
        CALL_NVSHMEM_P2P(get, int32, int32_t, peer, source_ptr, dest_ptr, count,
                         stream);
      }
      break;
    case PrimitiveType::S64:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, int64, int64_t, peer, source_ptr, dest_ptr, count,
                         stream);
      } else {
        CALL_NVSHMEM_P2P(get, int64, int64_t, peer, source_ptr, dest_ptr, count,
                         stream);
      }
      break;
    case PrimitiveType::U32:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, uint32, uint32_t, peer, source_ptr, dest_ptr,
                         count, stream);
      } else {
        CALL_NVSHMEM_P2P(get, uint32, uint32_t, peer, source_ptr, dest_ptr,
                         count, stream);
      }
      break;
    case PrimitiveType::U64:
      if (op_name == "put") {
        CALL_NVSHMEM_P2P(put, uint64, uint64_t, peer, source_ptr, dest_ptr,
                         count, stream);
      } else {
        CALL_NVSHMEM_P2P(get, uint64, uint64_t, peer, source_ptr, dest_ptr,
                         count, stream);
      }
      break;
    default:
      return absl::InternalError(
          absl::StrFormat("Invalid NVSHMEM %s type.", op_name));
  }
  return absl::OkStatus();
}

absl::Status NvshmemCommunicator::RegisterBuffer(void* addr, size_t length) {
  VLOG(3) << absl::StreamFormat("Registering NVSHMEM buffer: %p, length: %zu",
                                addr, length);

  if (nvshmemx_buffer_register(addr, length) != 0) {
    LOG(ERROR) << absl::StrFormat(
        "Failed to register NVSHMEM buffer at %p with length %zu", addr,
        length);
    return absl::InternalError("Failed to register NVSHMEM buffer");
  }

  return absl::OkStatus();
}

tsl::AsyncValueRef<NvshmemCommunicator::Event> NvshmemCommunicator::Send(
    se::DeviceMemoryBase recv_buffer, se::DeviceMemoryBase send_buffer,
    PrimitiveType dtype, size_t count, RankId peer, const Executor& executor) {
  VLOG(1) << "Send NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return absl::FailedPreconditionError("NvshmemCollectives not initialized.");
  }

  count = ToRealCount(dtype, count);
  TF_RETURN_IF_ERROR(
      P2P("put", dtype, recv_buffer, send_buffer, count, peer, executor));
  return tsl::MakeAvailableAsyncValueRef<Event>();
}

tsl::AsyncValueRef<NvshmemCommunicator::Event> NvshmemCommunicator::Recv(
    se::DeviceMemoryBase recv_buffer, se::DeviceMemoryBase send_buffer,
    PrimitiveType dtype, size_t count, RankId peer, const Executor& executor) {
  VLOG(1) << "Recv NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return absl::FailedPreconditionError("NvshmemCollectives not initialized.");
  }

  count = ToRealCount(dtype, count);
  TF_RETURN_IF_ERROR(
      P2P("get", dtype, recv_buffer, send_buffer, count, peer, executor));
  return tsl::MakeAvailableAsyncValueRef<Event>();
}

absl::Status NvshmemCommunicator::Quiet(const Executor& executor) {
  VLOG(1) << "Quiet NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return absl::FailedPreconditionError("NvshmemCollectives not initialized.");
  }

  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));
  nvshmemx_quiet_on_stream(se::gpu::AsGpuStreamValue(stream));
  return absl::OkStatus();
}

absl::Status NvshmemCommunicator::Fence() {
  VLOG(1) << "Fence NVSHMEM communicator: " << ToString();
  if (aborted_) {
    return absl::FailedPreconditionError("NvshmemCommunicator aborted");
  }
  if (!collectives_->IsInitialized()) {
    return FailedPrecondition("NvshmemCollectives not initialized.");
  }

  nvshmem_fence();
  return absl::OkStatus();
}

}  // namespace xla::gpu
