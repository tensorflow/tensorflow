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

#include "xla/backends/gpu/collectives/nccl_communicator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/backends/gpu/collectives/nccl_errors.h"
#include "xla/core/collectives/communicator.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#if (TF_ROCM_VERSION >= 50200)
#include "rocm/include/rccl/rccl.h"
#else
#include "rocm/include/rccl.h"
#endif  // TF_ROCM_VERSION >= 50200
#else
#include "third_party/nccl/nccl.h"
#endif  // TENSORFLOW_USE_ROCM

namespace xla::gpu {

//==-----------------------------------------------------------------------===//
// Conversions between XLA and NCCL data types
//==-----------------------------------------------------------------------===//

static size_t ToNcclCount(PrimitiveType dtype, size_t count) {
  return primitive_util::IsComplexType(dtype) ? count * 2 : count;
}

static absl::StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType dtype,
                                                     bool is_reduction_op) {
  switch (dtype) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
    case F8E5M2FNUZ:
    case F8E4M3FNUZ:
      return ncclInt8;
    case PRED:
    case U8:
      return ncclUint8;
    case S32:
      return ncclInt32;
    case U32:
      return ncclUint32;
    case S64:
      return ncclInt64;
    case U64:
      return ncclUint64;
    case F16:
      return ncclFloat16;
    case F32:
    case C64:
      return ncclFloat32;
    case F64:
    case C128:
      return ncclFloat64;
    case S16:
    case U16:
      // For reductions we expect 16 bit integer types to be promoted to 32-bit.
      if (is_reduction_op) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Unsupported data type for reduction operation: %s",
                            primitive_util::LowercasePrimitiveTypeName(dtype)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
    case BF16:
      return ncclBfloat16;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %s",
                          primitive_util::LowercasePrimitiveTypeName(dtype)));
  }
}

static ncclRedOp_t ToNcclReduction(ReductionKind kind) {
  switch (kind) {
    case ReductionKind::SUM:
      return ncclSum;
    case ReductionKind::PRODUCT:
      return ncclProd;
    case ReductionKind::MIN:
      return ncclMin;
    case ReductionKind::MAX:
      return ncclMax;
  }
}

//==-----------------------------------------------------------------------===//
// NCCL Communicator
//==-----------------------------------------------------------------------===//

namespace {
// An RAII handle for user buffers registered with an NCCL communicator.
class NcclRegisteredBufferHandle : public Communicator::RegisteredBufferHandle {
 public:
  NcclRegisteredBufferHandle(ncclComm_t comm_, void* handle);
  ~NcclRegisteredBufferHandle() override;

  absl::Status Unregister() final;

 private:
  ncclComm_t comm_;
  void* handle_;
};
}  // namespace

NcclRegisteredBufferHandle::NcclRegisteredBufferHandle(ncclComm_t comm,
                                                       void* handle)
    : comm_(comm), handle_(handle) {}

NcclRegisteredBufferHandle::~NcclRegisteredBufferHandle() {
  if (auto status = Unregister(); !status.ok()) {
    LOG(ERROR) << status.message();
  }
}

absl::Status NcclRegisteredBufferHandle::Unregister() {
  VLOG(3) << absl::StreamFormat(
      "Deregister buffer for NCCL communicator; handle=%p; comm=%p", handle_,
      comm_);
#if (NCCL_VERSION_CODE >= 21901)
  return XLA_NCCL_STATUS(ncclCommDeregister(comm_, handle_));
#else
  return Unimplemented("NCCL version does not support ncclCommDeregister");
#endif  // NCCL_VERSION_CODE >= 21901
}

NcclCommunicator::NcclCommunicator(ncclComm_t comm) : comm_(comm) {
  VLOG(1) << "Created " << *this;
}

NcclCommunicator::~NcclCommunicator() {
  VLOG(1) << "Destroy " << *this;
  XLA_NCCL_LOG_IF_ERROR(ncclCommDestroy(comm_));
}

absl::Status NcclCommunicator::Abort() {
  VLOG(1) << "Abort NCCL communicator: " << ToString();
  return XLA_NCCL_STATUS(ncclCommAbort(comm_));
}

absl::Status NcclCommunicator::HealthCheck() const {
  VLOG(5) << "Get last async error for NCCL communicator: " << ToString();

  ncclResult_t async_err;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommGetAsyncError(comm_, &async_err));
  if (async_err == ncclSuccess) return absl::OkStatus();

  return Internal("%s. Last NCCL error (maybe unrelated): %s",
                  ncclGetLastError(comm_), ncclGetErrorString(async_err));
}

absl::StatusOr<size_t> NcclCommunicator::NumRanks() const {
  VLOG(5) << "Get the number of ranks in NCCL communicator: " << ToString();
  int32_t count;
  XLA_NCCL_RETURN_IF_ERROR(ncclCommCount(comm_, &count));
  return count;
}

absl::StatusOr<std::unique_ptr<Communicator::RegisteredBufferHandle>>
NcclCommunicator::RegisterBuffer(stream_executor::DeviceMemoryBase buffer) {
  VLOG(3) << absl::StreamFormat(
      "Register buffer for NCCL communicator; buffer=%p; size=%d; comm=%p",
      buffer.opaque(), buffer.size(), comm_);
#if (NCCL_VERSION_CODE >= 21901)
  void* handle = nullptr;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclCommRegister(comm_, buffer.opaque(), buffer.size(), &handle));
  return std::make_unique<NcclRegisteredBufferHandle>(comm_, handle);
#else
  return Unimplemented("NCCL version does not support ncclCommRegister");
#endif  // NCCL_VERSION_CODE >= 21901
}

absl::Status NcclCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Communicator::Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllReduce operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclAllReduce(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::Broadcast(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         size_t root,
                                         const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Broadcast operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; root=%d; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, root, comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclBroadcast(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, root, comm_, se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::ReduceScatter(se::DeviceMemoryBase send_buffer,
                                             se::DeviceMemoryBase recv_buffer,
                                             PrimitiveType dtype, size_t count,
                                             ReductionKind reduction_kind,
                                             const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL ReduceScatter operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; reduction_kind=%s; comm=%p; "
      "stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, ReductionKindToString(reduction_kind), comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclReduceScatter(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, ToNcclReduction(reduction_kind), comm_,
      se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::AllGather(se::DeviceMemoryBase send_buffer,
                                         se::DeviceMemoryBase recv_buffer,
                                         PrimitiveType dtype, size_t count,
                                         const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL AllGather operation on device #%d; send_buffer=%p; "
      "recv_buffer=%p; dtype=%s; count=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      recv_buffer.opaque(), primitive_util::LowercasePrimitiveTypeName(dtype),
      count, comm_, stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclAllGather(
      send_buffer.opaque(), recv_buffer.opaque(), ToNcclCount(dtype, count),
      nccl_dtype, comm_, se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::Send(se::DeviceMemoryBase send_buffer,
                                    PrimitiveType dtype, size_t count,
                                    int32_t peer, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Send operation on device #%d; send_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), send_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer, comm_,
      stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclSend(send_buffer.opaque(),
                                  ToNcclCount(dtype, count), nccl_dtype, peer,
                                  comm_, se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::SendPtrToPeer(void* ptr, int32_t peer,
                                             const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL RecvPtrFromPeer operation on device #%d;  "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), peer, comm_, stream);
  return XLA_NCCL_STATUS(ncclSend(ptr, 1, ncclUint64, peer, comm_,
                                  se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::Recv(se::DeviceMemoryBase recv_buffer,
                                    PrimitiveType dtype, size_t count,
                                    int32_t peer, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL Recv operation on device #%d; recv_buffer=%p; dtype=%s; "
      "count=%d; peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), recv_buffer.opaque(),
      primitive_util::LowercasePrimitiveTypeName(dtype), count, peer, comm_,
      stream);

  TF_ASSIGN_OR_RETURN(ncclDataType_t nccl_dtype, ToNcclDataType(dtype, false));

  return XLA_NCCL_STATUS(ncclRecv(recv_buffer.opaque(),
                                  ToNcclCount(dtype, count), nccl_dtype, peer,
                                  comm_, se::gpu::AsGpuStreamValue(stream)));
}

absl::Status NcclCommunicator::RecvPtrFromPeer(void* ptr, int32_t peer,
                                               const Executor& executor) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream, ToStream(executor));

  VLOG(3) << absl::StreamFormat(
      "Launch NCCL RecvPtrFromPeer operation on device #%d; "
      "peer=%d; comm=%p; stream=%p",
      stream->parent()->device_ordinal(), peer, comm_, stream);

  return XLA_NCCL_STATUS(ncclRecv(ptr, 1, ncclUint64, peer, comm_,
                                  se::gpu::AsGpuStreamValue(stream)));
}

std::string NcclCommunicator::ToString() const {
  return absl::StrFormat("NccCommunicator(ncclComm_t=%p)", comm_);
}

absl::StatusOr<se::Stream*> NcclCommunicator::ToStream(
    const Executor& executor) {
  if (auto* gpu_executor =
          tsl::down_cast<const GpuCollectives::Executor*>(&executor)) {
    return gpu_executor->stream();
  }
  return InvalidArgument("Communicator executor is not a GPU executor");
}

}  // namespace xla::gpu
