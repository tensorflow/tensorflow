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

#include "xla/pjrt/cpu/mpi_collectives.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "mpi.h"  // NOLINT
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/cpu/collectives_interface.h"
#include "xla/service/global_device_id.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

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

MpiCollectivesCommunicator::MpiCollectivesCommunicator(int color, int key) {
  MPI_Comm_split(MPI_COMM_WORLD, color, key, &comm_);
  MPI_Comm_rank(comm_, &mpi_rank_);
  MPI_Comm_size(comm_, &mpi_size_);
}

MpiCollectivesCommunicator::~MpiCollectivesCommunicator() {
  MPI_Comm_free(&comm_);
};

absl::Status MpiCollectivesCommunicator::AllReduce(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t num_elements, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(element_type));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  return MpiErrorToAbslStatus(MPI_Allreduce(input_buffer, output_buffer,
                                            num_elements, type, op, comm_));
}

absl::Status MpiCollectivesCommunicator::CollectivePermute(
    const RendezvousKey& key, size_t num_bytes, std::optional<int> source_rank,
    absl::Span<int const> target_ranks, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  int tag = 0;  // TODO come up with better tags.

  const int rank = mpi_rank_;

  std::vector<MPI_Request> requests;

  if (source_rank) {
    if (*source_rank == rank) {
      std::memcpy(output_buffer, input_buffer, num_bytes);
    } else {
      VLOG(1) << "recv at " << rank << " from " << *source_rank;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Irecv(output_buffer, num_bytes, MPI_BYTE, *source_rank, tag,
                    comm_, &requests.back())));
    }
  } else {
    std::memset(output_buffer, 0, num_bytes);
  }

  for (int target : target_ranks) {
    if (target != rank) {
      VLOG(1) << "send from " << rank << " to " << target;
      requests.emplace_back();
      TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
          MPI_Isend(input_buffer, num_bytes, MPI_BYTE, target, tag, comm_,
                    &requests.back())));
    }
  }

  for (auto& request : requests) {
    TF_RETURN_IF_ERROR(
        MpiErrorToAbslStatus(MPI_Wait(&request, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllToAll(
    const RendezvousKey& key, size_t chunk_bytes,
    absl::Span<const void* const> input_buffers,
    absl::Span<void* const> output_buffers, absl::Duration timeout) {
  // We can't use MPI_Alltoall directly because it assumes that the inputs and
  // outputs are contiguous. Therefore here we implement it using MPI_Sendrecv.

  int tag = 0;  // TODO use better tags.
  const int rank = mpi_rank_;
  const int size = mpi_size_;
  TF_RET_CHECK(size == input_buffers.size());
  TF_RET_CHECK(size == output_buffers.size());

  std::memcpy(output_buffers[rank], input_buffers[rank], chunk_bytes);

  for (int i = 1; i < size; i++) {
    int send_rank = (rank + i) % size;
    int recv_rank = (rank + size - i) % size;
    TF_RETURN_IF_ERROR(MpiErrorToAbslStatus(
        MPI_Sendrecv(input_buffers[send_rank], chunk_bytes, MPI_BYTE, send_rank,
                     tag, output_buffers[recv_rank], chunk_bytes, MPI_BYTE,
                     recv_rank, tag, comm_, MPI_STATUS_IGNORE)));
  }

  return absl::OkStatus();
}

absl::Status MpiCollectivesCommunicator::AllGather(const RendezvousKey& key,
                                                   size_t chunk_bytes,
                                                   const void* input_buffer,
                                                   void* output_buffer,
                                                   absl::Duration timeout) {
  return MpiErrorToAbslStatus(MPI_Allgather(input_buffer, chunk_bytes, MPI_BYTE,
                                            output_buffer, chunk_bytes,
                                            MPI_BYTE, comm_));
}

absl::Status MpiCollectivesCommunicator::ReduceScatter(
    const RendezvousKey& key, ReductionKind reduction_kind,
    PrimitiveType element_type, size_t chunk_elems, const void* input_buffer,
    void* output_buffer, absl::Duration timeout) {
  const int size = mpi_size_;
  std::vector<int> recvcounts(size, chunk_elems);
  TF_ASSIGN_OR_RETURN(MPI_Datatype type, PrimitiveTypeToMpiType(element_type));
  TF_ASSIGN_OR_RETURN(MPI_Op op, ReductionKindToMpiOp(reduction_kind, type));
  return MpiErrorToAbslStatus(MPI_Reduce_scatter(
      input_buffer, output_buffer, recvcounts.data(), type, op, comm_));
}

void MpiCollectives::Init() {
  int provided;
  MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_world_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size_);
  VLOG(1) << "MPI rank=" << mpi_world_rank_ << " size=" << mpi_world_size_;
}

void MpiCollectives::Finalize() {
  contexts_.clear();
  MPI_Finalize();
}

absl::StatusOr<std::shared_ptr<CollectivesCommunicator>>
MpiCollectives::GetCommunicator(absl::Span<GlobalDeviceId const> global_devices,
                                int rank) {
  int flag;
  MPI_Is_thread_main(&flag);
  if (!flag) {
    return absl::UnknownError(
        absl::StrCat("MPI: Communicator requested from a thread that is not "
                     "the one MPI was initialized from. Multiple "
                     "threads/devices per process are not yet supported."));
  }

  auto& context = contexts_[std::make_tuple(
      std::vector<GlobalDeviceId>(global_devices.begin(), global_devices.end()),
      rank)];
  if (context) {
    return context;
  }

  int color;
  int key = 0;
  if (global_devices.size() > 0) {
    color = static_cast<int>(global_devices.at(0).value());
    key = rank;
  } else {
    color = MPI_UNDEFINED;
  }
  context = std::make_shared<MpiCollectivesCommunicator>(color, key);
  return context;
}

}  // namespace xla::cpu
