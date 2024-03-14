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

#include "xla/service/gpu/mock_nccl_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/gpus/cuda/include/vector_types.h"
#include "third_party/gpus/nccl/graph/topo.h"
#include "third_party/gpus/nccl/graph/xml.h"
#include "third_party/gpus/nccl/include/alloc.h"
#include "third_party/gpus/nccl/include/comm.h"
#include "third_party/gpus/nccl/include/graph.h"
#include "third_party/gpus/nccl/include/info.h"
#include "third_party/gpus/nccl/include/nccl_common.h"
#include "third_party/nccl/nccl.h"
#include "third_party/gpus/nccl/src/include/device.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/primitive_util.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/mock_nccl_topo_config.h"
#include "xla/service/gpu/mock_nccl_xml.h"
#include "xla/service/gpu/nccl_api.h"
#include "xla/service/gpu/nccl_clique.h"
#include "xla/service/gpu/nccl_clique_key.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/sleep_kernel.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/lockable.h"
#include "xla/service/rendezvous.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

//==-----------------------------------------------------------------------===//
// Macros to return or warn on NCCL errors.
//==-----------------------------------------------------------------------===//

static absl::Status ToStatus(ncclResult_t s, const char* file, int64_t line,
                             const char* expr) {
  if (s == ncclSuccess) return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "%s:%d: NCCL operation %s failed: %s."
      " Last NCCL warning(error) log entry (may be unrelated) '%s'.",
      file, line, expr, ncclGetErrorString(s), ncclGetLastError(nullptr)));
}

#define XLA_NCCL_STATUS(expr) \
  xla::gpu::ToStatus(expr, __FILE__, __LINE__, #expr)

#define XLA_NCCL_RETURN_IF_ERROR(expr)      \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      return s;                             \
    }                                       \
  } while (0)

#define XLA_NCCL_LOG_IF_ERROR(expr)         \
  do {                                      \
    absl::Status s = XLA_NCCL_STATUS(expr); \
    if (!s.ok()) {                          \
      LOG(ERROR) << s.ToString();           \
    }                                       \
  } while (0)

//==-----------------------------------------------------------------------===//

static absl::StatusOr<ncclDataType_t> ToNcclDataType(PrimitiveType element_type,
                                                     Thunk::Kind reduction_op) {
  switch (element_type) {
    case S8:
    case F8E5M2:
    case F8E4M3FN:
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
      // For all-reduce and reduce-scatter, we expect 16 bit integer types to be
      // promoted to 32-bit.
      if (reduction_op == Thunk::kNcclAllReduce ||
          reduction_op == Thunk::kNcclAllReduceStart ||
          reduction_op == Thunk::kNcclReduceScatter) {
        return tsl::errors::InvalidArgument(absl::StrFormat(
            "Unsupported data type: %s", PrimitiveType_Name(element_type)));
      }
      // For collectives that just move data around, we can use ncclFloat16 for
      // 16-bit integer data types.
      return ncclFloat16;
#if defined(__CUDA_BF16_TYPES_EXIST__) || TENSORFLOW_USE_ROCM
    case BF16:
      return ncclBfloat16;
#endif
    default:
      return tsl::errors::InvalidArgument(absl::StrFormat(
          "Unsupported data type: %s", PrimitiveType_Name(element_type)));
  }
}

static absl::StatusOr<std::pair<ncclDataType_t, int>>
ToNcclDataTypeAndCountMultiplier(PrimitiveType element_type,
                                 Thunk::Kind reduction_op) {
  TF_ASSIGN_OR_RETURN(ncclDataType_t dtype,
                      ToNcclDataType(element_type, reduction_op));
  bool is_complex = primitive_util::IsComplexType(element_type);
  return std::make_pair(dtype, is_complex ? 2 : 1);
}

using ncclInfo_t = ncclInfo*;

absl::StatusOr<int> GetNcclDataTypeSize(ncclDataType_t dtype) {
  switch (dtype) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclInt32:
    case ncclUint32:
      return 4;
    case ncclInt64:
    case ncclUint64:
      return 8;
    case ncclFloat16:
      return 2;
    case ncclFloat32:
      return 4;
    case ncclFloat64:
      return 8;
#if defined(__CUDA_BF16_TYPES_EXIST__) || TENSORFLOW_USE_ROCM
    case ncclBfloat16:
      return 2;
#endif
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported nccl data type: %d", dtype));
  }
}

absl::StatusOr<ncclFunc_t> ToNcclFunctionType(Thunk::Kind reduce_op) {
  switch (reduce_op) {
    case Thunk::kNcclAllReduce:
      return ncclFuncAllReduce;
    case Thunk::kNcclAllGather:
      return ncclFuncAllGather;
    case Thunk::kNcclReduceScatter:
      return ncclFuncReduceScatter;
    case Thunk::kNcclSend:
      return ncclFuncSend;
    case Thunk::kNcclRecv:
      return ncclFuncRecv;
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported nccl function type: %d", reduce_op));
  }
}

absl::Status LaunchSleepKernel(se::StreamExecutor* executor,
                               se::gpu::GpuStreamHandle gpu_stream,
                               ncclInfo_t info, int64_t sleep_duration) {
  void* kernel = GetSleepKernel();
  int64_t clock_cycles =
      sleep_duration * executor->GetDeviceDescription().clock_rate_ghz();
  void* kernel_args[] = {&clock_cycles};
  dim3 gridDim = {1, 1, 1};
  dim3 blockDim = {512, 1, 1};
  cudaError_t launch_status =
      cudaLaunchKernel(kernel, gridDim, blockDim, kernel_args, 0, gpu_stream);
  if (launch_status != cudaSuccess) {
    return absl::InternalError(absl::StrCat("Failed to launch kernel: ",
                                            cudaGetErrorString(launch_status)));
  }
  return absl::OkStatus();
}

inline absl::Status MockNcclInfoSetDerived(ncclInfo_t info, int nRanks) {
  TF_ASSIGN_OR_RETURN(int dtype_size, GetNcclDataTypeSize(info->datatype));
  info->nBytes = info->count * dtype_size;
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncBroadcast) {
    info->count = info->nBytes;
    info->datatype = ncclInt8;
  }
  if (info->coll == ncclFuncAllGather || info->coll == ncclFuncReduceScatter)
    info->nBytes *= nRanks;  // count is per rank
  return absl::OkStatus();
}

// Return estimated sleep time in nano seconds for simulating the nccl
// collective calls
absl::StatusOr<int64_t> GetMockNcclSleepTime(size_t count,
                                             ncclDataType_t datatype,
                                             ncclComm_t comm,
                                             cudaStream_t stream,
                                             ncclInfo_t info) {
  info->count = count;
  info->datatype = datatype;
  info->nChannels = 1;
  info->algorithm = -1;
  info->protocol = -1;

  TF_RETURN_IF_ERROR(MockNcclInfoSetDerived(info, comm->nRanks));

  int numPipeOps = 1;  // number of pipelined ops. Used to adjust latency.
                       // Assume 1 for simplicity.
  float minTime = std::numeric_limits<float>::infinity();
  float time = 0.0f;
  if (info->coll == ncclFuncAllReduce) {
    XLA_NCCL_RETURN_IF_ERROR(ncclTopoGetAlgoTime(
        info, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, numPipeOps, &time));
    info->algorithm = NCCL_ALGO_RING;
    info->protocol = NCCL_PROTO_SIMPLE;
    minTime = time;
  } else {
    for (int p = 0; p < 3; p++) {
      XLA_NCCL_RETURN_IF_ERROR(
          ncclTopoGetAlgoTime(info, NCCL_ALGO_RING, p, numPipeOps, &time));
      if (time > 0 && time < minTime) {
        info->algorithm = NCCL_ALGO_RING;
        info->protocol = p;
        minTime = time;
      }
    }
  }
  return ceil(minTime * 1000);
}

// Create the mock nccl communicator assuming all hosts have the same hardwares.
// We first create a local nccl communicator for gpus within a single host; then
// together with the input clique, we re-run nccl algorithms to construct the
// target nccl topology graphs.
absl::StatusOr<NcclComm::Lock> LockMockNcclComm(
    const Thunk::CollectiveExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model) {
  GlobalDeviceId global_device_id = params.global_device_id;

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  if (IsGlobalNcclConfig() &&
      (participants.size() != params.device_assn->replica_count())) {
    return InvalidArgument(
        "Partial replica groups are not allowed when using NCCL_COMM_ID "
        "environment configuration.");
  }

  std::vector<GlobalDeviceId> local_devices;
  if (params.global_device_id_map) {
    local_devices.reserve(params.global_device_id_map->size());
    for (const auto& entry : *params.global_device_id_map) {
      local_devices.push_back(entry.second);
    }
  } else {
    local_devices = participants;
  }
  TF_ASSIGN_OR_RETURN(
      const NcclCliqueIdCallback* clique_id_callback,
      GetNcclCliqueIdCallback(params.nccl_clique_id_callback, true));

  size_t num_local_participants = GetNumLocalParticipants(
      participants, params.global_device_id_map ? &local_devices : nullptr);

  auto global_it = absl::c_find(participants, global_device_id);
  TF_RET_CHECK(global_it != participants.end());
  int global_rank = global_it - participants.begin();

  if (global_rank != 0) {
    return absl::CancelledError("Only mock nccl call for gpu rank 0");
  }

  return AcquireMockNcclComm(params.run_id, OpId(op_id),
                             std::move(participants), std::move(local_devices),
                             num_local_participants, *clique_id_callback,
                             global_rank, stream_id, false, topo_model);
}

absl::Status RunMockNcclCollectives(NcclApi* nccl_api,
                                    std::vector<DeviceBufferPair>& buffers,
                                    se::Stream& stream,
                                    NcclApi::NcclCommHandle comm,
                                    Thunk::Kind reduce_op) {
  ncclComm_t mock_comm = reinterpret_cast<ncclComm_t>(comm);

  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing the mock nccl collective call from device ordinal: "
          << device_ordinal;
  se::StreamExecutor* executor = stream.parent();
  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  ncclInfo info;
  TF_ASSIGN_OR_RETURN(info.coll, ToNcclFunctionType(reduce_op));
  info.comm = reinterpret_cast<ncclComm_t>(mock_comm);
  info.stream = gpu_stream;

  int64_t total_element_count = 0;
  ncclDataType_t previous_dtype = ncclNumTypes;
  int64_t sleep_duration = 0;
  for (size_t i = 0; i < buffers.size(); ++i) {
    DeviceBufferPair& buffer = buffers[i];
    PrimitiveType element_type = buffer.element_type;
    TF_ASSIGN_OR_RETURN(
        auto dtype_and_multiplier,
        ToNcclDataTypeAndCountMultiplier(element_type, reduce_op));
    ncclDataType_t dtype = dtype_and_multiplier.first;
    int64_t element_count = buffer.element_count * dtype_and_multiplier.second;
    if (reduce_op == Thunk::kNcclReduceScatter)
      element_count = element_count / mock_comm->nRanks;
    if (i == 0 || dtype == previous_dtype) {
      previous_dtype = dtype;
      total_element_count += element_count;
      continue;
    }

    TF_ASSIGN_OR_RETURN(sleep_duration, GetMockNcclSleepTime(
                                            total_element_count, previous_dtype,
                                            mock_comm, gpu_stream, &info));
    TF_RETURN_IF_ERROR(
        LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
    total_element_count = element_count;
    previous_dtype = dtype;
  }

  TF_ASSIGN_OR_RETURN(sleep_duration,
                      GetMockNcclSleepTime(total_element_count, previous_dtype,
                                           mock_comm, gpu_stream, &info));

  TF_RETURN_IF_ERROR(
      LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
  VLOG(3) << "Done performing the mock nccl collective call for ordinal: "
          << device_ordinal;
  return absl::OkStatus();
}

absl::Status RunMockNcclAllToAll(NcclApi* nccl_api, bool has_split_dimension,
                                 std::vector<DeviceBufferPair>& buffers,
                                 se::Stream& stream,
                                 NcclApi::NcclCommHandle comm) {
  ncclComm_t mock_comm = reinterpret_cast<ncclComm_t>(comm);

  se::StreamExecutor* executor = stream.parent();
  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  int num_participants = mock_comm->nRanks;

  ncclInfo info;
  info.comm = mock_comm;
  info.stream = gpu_stream;

  int64_t sleep_duration = 0;

  // AllToAll can operate in two modes. Either it specifies a split dimension,
  // in which case inputs are split and outputs concatenated in that dimension
  // (here, we only support dimension 0), or it takes a list of inputs
  // and produces a tuple of outputs.
  if (has_split_dimension) {
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      TF_RET_CHECK(element_count % num_participants == 0)
          << "Buffer was not an exact multiple of the number of participants.";
      size_t chunk_elements = element_count / num_participants;
      size_t chunk_bytes = chunk_elements * ShapeUtil::ByteSizeOfPrimitiveType(
                                                buffer.element_type);
      for (int rank = 0; rank < num_participants; ++rank) {
        VLOG(3) << absl::StreamFormat(
            "Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
            "comm=%p, stream=%p)",
            send_buffer + rank * chunk_bytes, chunk_elements, rank,
            static_cast<const void*>(mock_comm), gpu_stream);
        info.coll = ncclFuncSend;
        TF_ASSIGN_OR_RETURN(sleep_duration,
                            GetMockNcclSleepTime(chunk_elements, dtype,
                                                 mock_comm, gpu_stream, &info));
        TF_RETURN_IF_ERROR(
            LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));

        VLOG(3) << absl::StreamFormat(
            "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
            "comm=%p, stream=%p)",
            recv_buffer + rank * chunk_bytes, chunk_elements, rank,
            static_cast<const void*>(mock_comm), gpu_stream);

        info.coll = ncclFuncRecv;
        TF_ASSIGN_OR_RETURN(sleep_duration,
                            GetMockNcclSleepTime(chunk_elements, dtype,
                                                 mock_comm, gpu_stream, &info));
        TF_RETURN_IF_ERROR(
            LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
      }
    }
  } else {
    TF_RET_CHECK(buffers.size() == num_participants)
        << "Number of inputs didn't match the number of participants.";
    for (size_t i = 0; i < buffers.size(); ++i) {
      DeviceBufferPair& buffer = buffers[i];
      const uint8_t* send_buffer =
          static_cast<uint8_t*>(buffer.source_buffer.opaque());
      uint8_t* recv_buffer =
          static_cast<uint8_t*>(buffer.destination_buffer.opaque());

      TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                          ToNcclDataTypeAndCountMultiplier(
                              buffer.element_type, Thunk::kNcclAllToAll));
      ncclDataType_t dtype = dtype_and_multiplier.first;
      int64_t element_count =
          buffer.element_count * dtype_and_multiplier.second;

      VLOG(3) << absl::StreamFormat(
          "Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          send_buffer, element_count, i, static_cast<const void*>(mock_comm),
          gpu_stream);

      info.coll = ncclFuncSend;
      TF_ASSIGN_OR_RETURN(sleep_duration,
                          GetMockNcclSleepTime(element_count, dtype, mock_comm,
                                               gpu_stream, &info));
      TF_RETURN_IF_ERROR(
          LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));

      VLOG(3) << absl::StreamFormat(
          "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          recv_buffer, element_count, i, static_cast<const void*>(mock_comm),
          gpu_stream);

      info.coll = ncclFuncRecv;
      TF_ASSIGN_OR_RETURN(sleep_duration,
                          GetMockNcclSleepTime(element_count, dtype, mock_comm,
                                               gpu_stream, &info));
      TF_RETURN_IF_ERROR(
          LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
    }
  }

  VLOG(3) << "Done performing mock all-to-all ";
  return absl::OkStatus();
}

absl::Status RunMockCollectivePermute(
    NcclApi* nccl_api, NcclP2PConfig::SourceTargetMapEntry source_target,
    DeviceBufferPair& buffer, se::Stream& stream, NcclApi::NcclCommHandle comm,
    absl::string_view device_string, int64_t current_id) {
  ncclComm_t mock_comm = reinterpret_cast<ncclComm_t>(comm);

  se::StreamExecutor* executor = stream.parent();
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing collective permute from device ordinal: "
          << device_ordinal << "current_id " << current_id;

  const std::optional<int64_t> source_id = source_target.source;
  const std::optional<int64_t> target_id = source_target.target;

  se::DeviceMemoryBase src_addr = buffer.source_buffer;
  se::DeviceMemoryBase dest_addr = buffer.destination_buffer;

  VLOG(3) << absl::StreamFormat("%s : id = %d, source_id = %d, target_id = %d",
                                device_string, current_id,
                                source_id.value_or(-1), target_id.value_or(-1));

  TF_ASSIGN_OR_RETURN(auto dtype_and_multiplier,
                      ToNcclDataTypeAndCountMultiplier(
                          buffer.element_type, Thunk::kNcclCollectivePermute));
  ncclDataType_t dtype = dtype_and_multiplier.first;
  int64_t element_count = buffer.element_count * dtype_and_multiplier.second;

  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  ncclInfo info;
  info.comm = mock_comm;
  info.stream = gpu_stream;

  int64_t sleep_duration = 0;

  // Send source buffer to target peer if needed.
  if (target_id) {
    info.coll = ncclFuncSend;
    VLOG(3) << absl::StreamFormat(
        "%s : Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
        "comm=%p, stream=%p)",
        device_string, src_addr.opaque(), element_count, *target_id,
        static_cast<const void*>(mock_comm), gpu_stream);
    TF_ASSIGN_OR_RETURN(sleep_duration,
                        GetMockNcclSleepTime(element_count, dtype, mock_comm,
                                             gpu_stream, &info));
    TF_RETURN_IF_ERROR(
        LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
  }

  // Receive data from the source peer to the destination buffer.
  if (source_id) {
    info.coll = ncclFuncRecv;
    VLOG(3) << absl::StreamFormat(
        "%s : Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d comm=%p, "
        "stream=%p)",
        device_string, dest_addr.opaque(), element_count, *source_id,
        static_cast<const void*>(mock_comm), gpu_stream);
    TF_ASSIGN_OR_RETURN(sleep_duration,
                        GetMockNcclSleepTime(element_count, dtype, mock_comm,
                                             gpu_stream, &info));
    TF_RETURN_IF_ERROR(
        LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
  }

  VLOG(3) << "Done performing the mock nccl collective call for ordinal: "
          << device_ordinal;

  if (!source_id) {
    // If there is no source peer, i.e. no one send us any data, zero out dest
    // buffer.
    VLOG(3) << absl::StreamFormat(
        "%s : mock collective-Permute: Issuing MemZero", device_string);
    return stream.MemZero(&dest_addr, dest_addr.size());
  }
  return absl::OkStatus();
}

namespace {
void CheckNcclAsyncError(NcclComm& lockable_comm) {
  NcclApi::NcclCommHandle comm = *lockable_comm.Acquire();
  if (comm == nullptr) return;

  absl::Status status = NcclApi::Default()->CommGetAsyncError(comm);
  if (!status.ok()) LOG(ERROR) << status;
}

struct NcclCliqueState {
  NcclCliqueId clique_id;
  int64_t run_id = -1;

  // `mu` guards `communicators` and `status` during initialization.
  // Once `ready` has been notified, the communicators may be accessed without
  // synchronization.
  absl::Mutex mu;
  absl::Notification ready;
  absl::Status status;
  absl::flat_hash_map<int, std::unique_ptr<NcclComm>> communicators;
};

using NcclClique = Lockable<NcclCliqueState>;

struct NcclCliques {
  NcclClique& operator[](const NcclCliqueKey& key) {
    absl::MutexLock lock(&mu);
    return cliques[key];
  }

  absl::Mutex mu;
  absl::node_hash_map<NcclCliqueKey, NcclClique> cliques ABSL_GUARDED_BY(mu);
};

absl::StatusOr<ncclUniqueId> ToNcclUniqueId(const std::string& id_str) {
  static_assert(sizeof(ncclUniqueId) == NCCL_UNIQUE_ID_BYTES,
                "NCCL_UNIQUE_ID_BYTES");

  TF_RET_CHECK(id_str.size() == NCCL_UNIQUE_ID_BYTES);
  ncclUniqueId id;
  absl::c_copy(id_str, id.internal);
  return id;
}

absl::StatusOr<std::shared_ptr<NcclClique::Lock>> AcquireNcclClique(
    RunId run_id, OpId op_id, NcclCliqueKey clique_key,
    const NcclCliqueIdCallback& clique_id_callback,
    size_t num_local_participants, bool may_skip_rendezvous) {
  static auto& cliques = *new NcclCliques;

  VLOG(2) << "AcquireNcclClique Rendezvous key (clique_key:"
          << clique_key.ToString() << ", run" << run_id.ToString() << ", op"
          << op_id.value() << ")";

  auto rendezvous_key = std::make_tuple(run_id, op_id, std::move(clique_key));

  int64_t terminate_timeout = xla::GetDebugOptionsFromFlags()
                                  .xla_gpu_nccl_termination_timeout_seconds();

  return RendezvousSingle<absl::StatusOr<NcclClique::Lock>>(
      "acquire mock NCCL clique", rendezvous_key, num_local_participants,
      [&]() -> absl::StatusOr<NcclClique::Lock> {
        const NcclCliqueKey& clique_key = std::get<2>(rendezvous_key);
        NcclClique::Lock clique = cliques[clique_key].Acquire();
        clique->run_id = run_id.ToInt();
        return clique;
      },
      /*warn_stuck_timeout=*/absl::Seconds(10),
      (terminate_timeout >= 0) ? absl::Seconds(terminate_timeout)
                               : absl::InfiniteDuration());
}

absl::Status InitializeMockNcclCostModel(
    int nRanks, int rank, int num_local_participants,
    absl::Span<const std::pair<int, int>> local_ranks,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model,
    ncclComm_t* comm_ptr) {
  XLA_NCCL_RETURN_IF_ERROR(ncclCalloc(comm_ptr, 1));
  ncclComm_t comm = *comm_ptr;
  comm->nChannels = 1;
  comm->nRanks = nRanks;
  comm->rank = rank;
  absl::string_view xml_str;
  switch (topo_model) {
    case GpuExecutableRunOptions::MockNcclTopoModel::kGCPA3:
      comm->collNetSupport = false;
      comm->nvlsSupport = false;
      comm->minCompCap = comm->maxCompCap = stream_executor::
          CudaComputeCapability::CudaComputeCapabilities::HOPPER;
      XLA_NCCL_RETURN_IF_ERROR(ncclCalloc(&comm->peerInfo, nRanks + 1));
      xml_str = kGCPA3;
      break;
    case GpuExecutableRunOptions::MockNcclTopoModel::kNvidia:
      comm->collNetSupport = false;
      comm->nvlsSupport = false;
      comm->minCompCap = comm->maxCompCap = stream_executor::
          CudaComputeCapability::CudaComputeCapabilities::AMPERE;
      XLA_NCCL_RETURN_IF_ERROR(ncclCalloc(&comm->peerInfo, nRanks + 1));
      xml_str = kNvidia;
      break;
    default:
      return absl::InvalidArgumentError("Unknown MockNcclTopoModel");
  }

  auto xml = std::make_unique<ncclXml>();
  TF_RETURN_IF_ERROR(MockTopoGetXml(xml_str, xml.get()));
  TF_RETURN_IF_ERROR(MockNcclTopoUpdateXml(local_ranks, xml.get()));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoTrimXml(xml.get()));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoGetSystemFromXml(xml.get(), &comm->topo));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoComputePaths(comm->topo, nullptr));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoTrimSystem(comm->topo, comm));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoComputePaths(comm->topo, nullptr));
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoSearchInit(comm->topo));

  ncclTopoGraph ringGraph;
  ncclTopoGraph treeGraph;
  ncclTopoGraph collNetGraph;
  ncclTopoGraph nvlsGraph;
  ncclTopoGraph* graphs[] = {&treeGraph,    &ringGraph, &collNetGraph,
                             &collNetGraph, &nvlsGraph, &nvlsGraph};

  // Get rings and trees
  ringGraph.id = 0;
  ringGraph.pattern = NCCL_TOPO_PATTERN_RING;
  ringGraph.collNet = 0;
  ringGraph.minChannels = 1;
  ringGraph.maxChannels = MAXCHANNELS / 2;
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoCompute(comm->topo, &ringGraph));

  treeGraph.id = 1;
  treeGraph.pattern = NCCL_TOPO_PATTERN_BALANCED_TREE;
  treeGraph.collNet = 0;
  treeGraph.minChannels = ringGraph.nChannels;
  treeGraph.maxChannels = ringGraph.nChannels;
  XLA_NCCL_RETURN_IF_ERROR(ncclTopoCompute(comm->topo, &treeGraph));

  collNetGraph.id = 2;
  collNetGraph.pattern = NCCL_TOPO_PATTERN_TREE;
  collNetGraph.collNet = 1;
  collNetGraph.minChannels = collNetGraph.maxChannels = ringGraph.nChannels;
  if (comm->collNetSupport) {
    XLA_NCCL_RETURN_IF_ERROR(ncclTopoCompute(comm->topo, &collNetGraph));
  } else {
    collNetGraph.nChannels = 0;
  }

  nvlsGraph.id = 3;
  nvlsGraph.pattern = NCCL_TOPO_PATTERN_NVLS;
  nvlsGraph.collNet = 0;
  nvlsGraph.minChannels = 1;
  nvlsGraph.maxChannels = MAXCHANNELS;
  if (comm->nvlsSupport) {
    XLA_NCCL_RETURN_IF_ERROR(ncclTopoCompute(comm->topo, &nvlsGraph));
  } else {
    nvlsGraph.nChannels = 0;
  }

  comm->nNodes = nRanks / num_local_participants;
  XLA_NCCL_RETURN_IF_ERROR(
      ncclTopoTuneModel(comm, comm->minCompCap, comm->maxCompCap, graphs));
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<NcclComm::Lock> AcquireMockNcclComm(
    RunId run_id, OpId op_id, std::vector<GlobalDeviceId> participants,
    std::vector<GlobalDeviceId> local_devices, size_t num_local_participants,
    const NcclCliqueIdCallback& clique_id_callback, int rank, int64_t stream_id,
    bool enable_clique_optimization,
    GpuExecutableRunOptions::MockNcclTopoModel topo_model) {
  int nRanks = participants.size();
  std::vector<std::pair<int, int>> local_ranks;
  for (int i = 0; i < local_devices.size(); i++) {
    auto it = absl::c_find(participants, local_devices[i]);
    if (it != participants.end()) {
      local_ranks.push_back(std::make_pair(it - participants.begin(), i));
    }
  }

  // Ensure that this group of threads have exclusive access to the clique to
  // prevent threads from different groups locking communicators in the clique.
  NcclCliqueKey clique_key(std::move(participants), stream_id);
  TF_ASSIGN_OR_RETURN(
      auto clique,
      AcquireNcclClique(
          run_id, op_id, clique_key, clique_id_callback, 1,
          enable_clique_optimization ||
              stream_id == GetStreamId(true, AsyncStreamKind::kP2P0)));

  struct AllCommunicators {
    absl::Mutex mu;
    std::vector<NcclComm*> communicators ABSL_GUARDED_BY(mu);
  };
  static auto& all_communicators = *new AllCommunicators;

  // Launch a thread that periodically checks all NCCL communicators for
  // asynchronous errors. If an asynchronous error is observed, the communicator
  // is aborted and an error message logged.
  static auto check_async_error_thread = tsl::Env::Default()->StartThread(
      tsl::ThreadOptions(), "nccl_async_error_thread", [&] {
        while (true) {
          absl::SleepFor(absl::Seconds(30));
          absl::MutexLock lock(&all_communicators.mu);
          for (NcclComm* comm : all_communicators.communicators) {
            CheckNcclAsyncError(*comm);
          }
        }
      });
  (void)check_async_error_thread;  // Silence unused variable warning.

  NcclCliqueState& state = **clique;

  if (!state.ready.HasBeenNotified()) {
    ncclComm_t comm = nullptr;
    absl::Status status = InitializeMockNcclCostModel(
        nRanks, rank, num_local_participants, local_ranks, topo_model, &comm);
    size_t num_initialized = [&] {
      absl::MutexLock lock(&state.mu);
      state.status.Update(status);
      state.communicators[rank] = std::make_unique<NcclComm>(
          reinterpret_cast<NcclApi::NcclCommHandle>(comm));
      return state.communicators.size();
    }();

    // Wait for all communicators to initialize before allowing any progress.
    // Otherwise we may get deadlocks, because ncclCommInitRank may allocate,
    // which may block on the completion of device activity on a peer device,
    // which may depend on the completion of this collective if we do not have a
    // barrier to prevent it.
    if (num_initialized == 1) {
      state.ready.Notify();
    } else {
      TF_RETURN_IF_ERROR(status);
      state.ready.WaitForNotification();
    }

    absl::MutexLock lock(&all_communicators.mu);
    all_communicators.communicators.push_back(state.communicators[rank].get());
  }

  TF_RETURN_IF_ERROR(state.status);
  return state.communicators[rank]->Acquire();
}

}  // namespace gpu
}  // namespace xla
