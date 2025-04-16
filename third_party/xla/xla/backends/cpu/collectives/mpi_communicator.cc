/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/backends/cpu/collectives/mpi_communicator.h"

#include <cstddef>
#include <cstring>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "mpi.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<MPI_Datatype> PrimitiveTypeToMpiType(
    PrimitiveType element_type) {
  switch (element_type) {
    case S8:
      return MPI_INT8_T;
    case U8:
    case PRED:
      return MPI_UINT8_T;
    case S16:
      return MPI_INT16_T;
    case U16:
      return MPI_UINT16_T;
    case S32:
      return MPI_INT32_T;
    case U32:
      return MPI_UINT32_T;
    case S64:
      return MPI_INT64_T;
    case U64:
      return MPI_UINT64_T;
    case F32:
      return MPI_FLOAT;
    case F64:
      return MPI_DOUBLE;
    case C64:
      return MPI_C_COMPLEX;
    case C128:
      return MPI_C_DOUBLE_COMPLEX;
    default:
      // For implementing the reduction of unsupported types
      // see e.g. https://stackoverflow.com/a/29643391
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported primitive type for reduction: ",
          primitive_util::LowercasePrimitiveTypeName(element_type)));
  }
}

bool MpiTypeIsComplex(MPI_Datatype type) {
  return type == MPI_C_COMPLEX || type == MPI_C_DOUBLE_COMPLEX;
}

absl::StatusOr<MPI_Op> ReductionKindToMpiOp(ReductionKind reduction_kind,
                                            MPI_Datatype type) {
  switch (reduction_kind) {
    case ReductionKind::SUM:
      return MPI_SUM;
    case ReductionKind::PRODUCT:
      return MPI_PROD;
    case ReductionKind::MIN:
      if (!MpiTypeIsComplex(type)) {
        return MPI_MIN;
      } else {
        return absl::InvalidArgumentError(
            "MIN reduction not supported for complex types");
      }
    case ReductionKind::MAX:
      if (!MpiTypeIsComplex(type)) {
        return MPI_MAX;
      } else {
        return absl::InvalidArgumentError(
            "MAX reduction not supported for complex types");
      }
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unknown reduction kind: ", reduction_kind));
  }
}

static absl::Status MpiErrorToAbslStatus(int error) {
  if (error != MPI_SUCCESS) {
    char error_str[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(error, error_str, &len);
    return absl::UnknownError(absl::StrCat("MPI error: ", error_str));
  }
  return absl::OkStatus();
}

MpiCommunicator::MpiCommunicator(int color, int key) {
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm_);
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);
}

MpiCommunicator::~MpiCommunicator() { MPI_Comm_free(&comm_); };

tsl::AsyncValueRef<MpiCommunicator::Event> MpiCommunicator::AllReduce(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(dtype));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(MPI_Allreduce(
      send_buffer.opaque(), recv_buffer.opaque(), count, type, op, comm_)));
  return OkEvent();
}

tsl::AsyncValueRef<MpiCommunicator::Event> MpiCommunicator::CollectivePermute(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, std::optional<RankId> source_rank,
    absl::Span<const RankId> target_ranks, const Executor& executor) {
  int tag = 0;  // TODO come up with better tags.

  const int rank = mpi_rank_;

  std::vector<MPI_Request> requests;

  size_t num_bytes = count * primitive_util::ByteWidth(dtype);

  if (source_rank) {
    if (source_rank->value() == rank) {
      std::memcpy(recv_buffer.opaque(), send_buffer.opaque(), num_bytes);
    } else {
      VLOG(1) << "recv at " << rank << " from " << source_rank->value();
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Irecv(recv_buffer.opaque(), num_bytes, MPI_BYTE,
                    source_rank->value(), tag, comm_, &requests.back())));
    }
  } else {
    std::memset(recv_buffer.opaque(), 0, num_bytes);
  }

  for (RankId target : target_ranks) {
    if (target != rank) {
      VLOG(1) << "send from " << rank << " to " << target.value();
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(send_buffer.opaque(), num_bytes, MPI_BYTE, target.value(),
                    tag, comm_, &requests.back())));
    }
  }

  for (auto& request : requests) {
    TF_RETURN_IF_ERROR(
        MpiErrorToAbslStatus(MPI_Wait(&request, MPI_STATUS_IGNORE)));
  }

  return OkEvent();
}

tsl::AsyncValueRef<MpiCommunicator::Event> MpiCommunicator::AllToAll(
    absl::Span<const se::DeviceMemoryBase> send_buffers,
    absl::Span<const se::DeviceMemoryBase> recv_buffers, PrimitiveType dtype,
    size_t count, const Executor& executor) {
  // We can't use MPI_Alltoall directly because it assumes that the inputs and
  // outputs are contiguous. Therefore here we implement it using MPI_Sendrecv.

  int tag = 0;  // TODO use better tags.
  const int rank = mpi_rank_;
  const int size = mpi_size_;
  TF_RET_CHECK(size == send_buffers.size());
  TF_RET_CHECK(size == recv_buffers.size());

  size_t chunk_bytes = count * primitive_util::ByteWidth(dtype);

  std::vector<void*> input_buffers;
  std::vector<void*> output_buffers;

  for (int i = 0; i < size; i++) {
    input_buffers.push_back(send_buffers[i].opaque());
    output_buffers.push_back(recv_buffers[i].opaque());
  }

  std::memcpy(output_buffers[rank], input_buffers[rank], chunk_bytes);

  for (int i = 1; i < size; i++) {
    int send_rank = (rank + i) % size;
    int recv_rank = (rank + size - i) % size;
    TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
        MPI_Sendrecv(input_buffers[send_rank], chunk_bytes, MPI_BYTE, send_rank,
                     tag, output_buffers[recv_rank], chunk_bytes, MPI_BYTE,
                     recv_rank, tag, comm_, MPI_STATUS_IGNORE)));
  }

  return OkEvent();
}

tsl::AsyncValueRef<MpiCommunicator::Event> MpiCommunicator::AllGather(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, const Executor& executor) {
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(dtype));
  TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
      MPI_Allgather(send_buffer.opaque(), count, type, recv_buffer.opaque(),
                    count, type, comm_)));

  return OkEvent();
}

tsl::AsyncValueRef<MpiCommunicator::Event> MpiCommunicator::ReduceScatter(
    se::DeviceMemoryBase send_buffer, se::DeviceMemoryBase recv_buffer,
    PrimitiveType dtype, size_t count, ReductionKind reduction_kind,
    const Executor& executor) {
  const int size = mpi_size_;
  std::vector<int> recvcounts(size, count);
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(dtype));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
      MPI_Reduce_scatter(send_buffer.opaque(), recv_buffer.opaque(),
                         recvcounts.data(), type, op, comm_)));

  return OkEvent();
}

}  // namespace xla::cpu
