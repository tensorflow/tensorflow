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

#include "xla/service/gpu/mock_nccl_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/driver_types.h"
#include "third_party/gpus/cuda/include/vector_types.h"
#include "third_party/gpus/nccl/graph/topo.h"
#include "third_party/gpus/nccl/include/comm.h"
#include "third_party/gpus/nccl/include/graph.h"
#include "third_party/gpus/nccl/include/info.h"
#include "third_party/gpus/nccl/include/nccl_common.h"
#include "third_party/nccl/nccl.h"
#include "xla/executable_run_options.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/global_device_id.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/service/gpu/mock_nccl_config.h"
#include "xla/service/gpu/mock_nccl_config.pb.h"
#include "xla/service/gpu/mock_nccl_sleep_kernel.h"
#include "xla/service/gpu/nccl_collective_thunk.h"
#include "xla/service/gpu/nccl_p2p_thunk_common.h"
#include "xla/service/gpu/nccl_utils.h"
#include "xla/service/gpu/thunk.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_types.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

struct MockNcclComm {
  ncclComm comm;
  ncclTopoSystem topo;
};

void DestroyMockNcclComm::operator()(MockNcclComm_t mock_comm) {
  delete mock_comm;
}

using ncclInfo_t = ncclInfo*;

StatusOr<int> GetNcclDataTypeSize(ncclDataType_t dtype) {
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

StatusOr<ncclFunc_t> ToNcclFunctionType(Thunk::Kind reduce_op) {
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

Status LaunchSleepKernel(se::StreamExecutor* executor,
                         se::gpu::GpuStreamHandle gpu_stream, ncclInfo_t info,
                         int64_t sleep_duration) {
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
StatusOr<int64_t> GetMockNcclSleepTime(size_t count, ncclDataType_t datatype,
                                       ncclComm_t comm, cudaStream_t stream,
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
    XLA_CUDA_RETURN_IF_ERROR(ncclTopoGetAlgoTime(
        info, NCCL_ALGO_RING, NCCL_PROTO_SIMPLE, numPipeOps, &time));
    info->algorithm = NCCL_ALGO_RING;
    info->protocol = NCCL_PROTO_SIMPLE;
    minTime = time;
  } else {
    for (int p = 0; p < 3; p++) {
      XLA_CUDA_RETURN_IF_ERROR(
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

static void ConvertMockNcclTopoGraphConfig(
    ncclTopoGraph** graphs, absl::Span<MockNcclTopoGraphConfig> configs) {
  CHECK_EQ(configs.size(), 6);
  for (int i = 0; i < configs.size(); ++i) {
    graphs[i]->id = i;
    graphs[i]->typeInter = configs[i].type_inter();
    graphs[i]->typeIntra = configs[i].type_intra();
    graphs[i]->bwInter = configs[i].bw_inter();
    graphs[i]->bwIntra = configs[i].bw_intra();
    graphs[i]->pattern = configs[i].pattern();
    graphs[i]->nChannels = configs[i].num_channels();
    graphs[i]->sameChannels = configs[i].same_channels();
    graphs[i]->latencyInter = configs[i].latency_inter();
  }
}

Status MockNcclCommInitRank(se::StreamExecutor* executor,
                            MockNcclComm_t mock_comm, int nranks, int nnodes,
                            int rank) {
  absl::InlinedVector<MockNcclTopoGraphConfig, 6> mock_graph_configs =
      GetNcclTopoGraphConfig();
  MockNcclTopoCpuNode mock_cpu_node = GetNcclTopoCpuNode();

  struct ncclTopoGraph ringGraph;
  struct ncclTopoGraph treeGraph;
  struct ncclTopoGraph collNetGraph;
  struct ncclTopoGraph nvlsGraph;
  struct ncclTopoGraph* graphs[] = {&treeGraph,    &ringGraph, &collNetGraph,
                                    &collNetGraph, &nvlsGraph, &nvlsGraph};

  ConvertMockNcclTopoGraphConfig(graphs, absl::MakeSpan(mock_graph_configs));
  const stream_executor::DeviceDescription& device_description =
      executor->GetDeviceDescription();
  mock_comm->comm.rank = rank;
  mock_comm->comm.nNodes = nnodes;
  mock_comm->comm.nRanks = nranks;
  mock_comm->comm.nChannels = 1;
  mock_comm->comm.collNetSupport = false;
  mock_comm->comm.nvlsSupport = false;
  mock_comm->comm.maxCompCap = mock_comm->comm.minCompCap =
      device_description.cuda_compute_capability().major;

  mock_comm->topo.nodes[CPU].nodes[0].cpu.arch = mock_cpu_node.arch();
  mock_comm->topo.nodes[CPU].nodes[0].cpu.vendor = mock_cpu_node.vendor();
  mock_comm->comm.topo = &mock_comm->topo;

  return XLA_CUDA_STATUS(ncclTopoTuneModel(&mock_comm->comm,
                                           mock_comm->comm.minCompCap,
                                           mock_comm->comm.maxCompCap, graphs));
}

StatusOr<MockNcclCommReference> InitializeMockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id,
    bool enable_clique_optimization) {
  TF_ASSIGN_OR_RETURN(GlobalDeviceId global_device_id,
                      params.GetGlobalDeviceId());

  TF_ASSIGN_OR_RETURN(
      std::vector<GlobalDeviceId> participants,
      GetParticipatingDevices(global_device_id, *params.device_assn,
                              replica_groups, group_mode));

  std::vector<GlobalDeviceId> local_devices;
  if (params.gpu_global_device_ids) {
    local_devices.reserve(params.gpu_global_device_ids->size());
    for (const auto& entry : *params.gpu_global_device_ids) {
      local_devices.push_back(entry.second);
    }
  }
  size_t num_local_participants = GetNumLocalParticipants(
      participants, params.gpu_global_device_ids ? &local_devices : nullptr);

  se::StreamExecutor* executor = params.stream_executor;
  se::gpu::ScopedActivateExecutorContext scoped_context(executor);

  int nranks = participants.size();
  int nnodes = nranks / num_local_participants;

  auto comm = MockNcclCommReference(new MockNcclComm());
  TF_RETURN_IF_ERROR(
      MockNcclCommInitRank(executor, comm.get(), nranks, nnodes, /*rank=*/0));
  return comm;
}

Status RunMockNcclCollectives(std::vector<DeviceBufferPair>& buffers,
                              se::Stream& stream, MockNcclComm_t mock_comm,
                              Thunk::Kind reduce_op) {
  int device_ordinal = stream.parent()->device_ordinal();
  VLOG(3) << "Performing the mock nccl collective call from device ordinal: "
          << device_ordinal;
  se::StreamExecutor* executor = stream.parent();
  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  ncclComm_t comm = &mock_comm->comm;
  ncclInfo info;
  TF_ASSIGN_OR_RETURN(info.coll, ToNcclFunctionType(reduce_op));
  info.comm = comm;
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
      element_count = element_count / comm->nRanks;
    if (i == 0 || dtype == previous_dtype) {
      previous_dtype = dtype;
      total_element_count += element_count;
      continue;
    }

    TF_ASSIGN_OR_RETURN(sleep_duration, GetMockNcclSleepTime(
                                            total_element_count, previous_dtype,
                                            comm, gpu_stream, &info));
    TF_RETURN_IF_ERROR(
        LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
    total_element_count = element_count;
    previous_dtype = dtype;
  }

  TF_ASSIGN_OR_RETURN(sleep_duration,
                      GetMockNcclSleepTime(total_element_count, previous_dtype,
                                           comm, gpu_stream, &info));

  TF_RETURN_IF_ERROR(
      LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
  VLOG(3) << "Done performing the mock nccl collective call for ordinal: "
          << device_ordinal;
  return absl::OkStatus();
}

Status RunMockNcclAllToAll(bool has_split_dimension,
                           std::vector<DeviceBufferPair>& buffers,
                           se::Stream& stream, MockNcclComm_t mock_comm) {
  se::StreamExecutor* executor = stream.parent();
  se::gpu::GpuStreamHandle gpu_stream = se::gpu::AsGpuStreamValue(&stream);
  ncclComm_t comm = &mock_comm->comm;
  int num_participants = comm->nRanks;

  ncclInfo info;
  info.comm = comm;
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
            static_cast<const void*>(comm), gpu_stream);
        info.coll = ncclFuncSend;
        TF_ASSIGN_OR_RETURN(sleep_duration,
                            GetMockNcclSleepTime(chunk_elements, dtype, comm,
                                                 gpu_stream, &info));
        TF_RETURN_IF_ERROR(
            LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));

        VLOG(3) << absl::StreamFormat(
            "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
            "comm=%p, stream=%p)",
            recv_buffer + rank * chunk_bytes, chunk_elements, rank,
            static_cast<const void*>(comm), gpu_stream);

        info.coll = ncclFuncRecv;
        TF_ASSIGN_OR_RETURN(sleep_duration,
                            GetMockNcclSleepTime(chunk_elements, dtype, comm,
                                                 gpu_stream, &info));
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
          send_buffer, element_count, i, static_cast<const void*>(comm),
          gpu_stream);

      info.coll = ncclFuncSend;
      TF_ASSIGN_OR_RETURN(
          sleep_duration,
          GetMockNcclSleepTime(element_count, dtype, comm, gpu_stream, &info));
      TF_RETURN_IF_ERROR(
          LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));

      VLOG(3) << absl::StreamFormat(
          "Calling mock ncclRecv(recvbuff=%p, count=%d, peer=%d "
          "comm=%p, stream=%p)",
          recv_buffer, element_count, i, static_cast<const void*>(comm),
          gpu_stream);

      info.coll = ncclFuncRecv;
      TF_ASSIGN_OR_RETURN(
          sleep_duration,
          GetMockNcclSleepTime(element_count, dtype, comm, gpu_stream, &info));
      TF_RETURN_IF_ERROR(
          LaunchSleepKernel(executor, gpu_stream, &info, sleep_duration));
    }
  }

  VLOG(3) << "Done performing mock all-to-all ";
  return OkStatus();
}

Status RunMockCollectivePermute(
    NcclP2PConfig::SourceTargetMapEntry source_target, DeviceBufferPair& buffer,
    se::Stream& stream, MockNcclComm_t mock_comm,
    absl::string_view device_string, int64_t current_id) {
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
  ncclComm_t comm = &mock_comm->comm;
  ncclInfo info;
  info.comm = comm;
  info.stream = gpu_stream;

  int64_t sleep_duration = 0;

  // Send source buffer to target peer if needed.
  if (target_id) {
    info.coll = ncclFuncSend;
    VLOG(3) << absl::StreamFormat(
        "%s : Calling mock ncclSend(sendbuff=%p, count=%d, peer=%d "
        "comm=%p, stream=%p)",
        device_string, src_addr.opaque(), element_count, *target_id,
        static_cast<const void*>(comm), gpu_stream);
    TF_ASSIGN_OR_RETURN(
        sleep_duration,
        GetMockNcclSleepTime(element_count, dtype, comm, gpu_stream, &info));
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
        static_cast<const void*>(comm), gpu_stream);
    TF_ASSIGN_OR_RETURN(
        sleep_duration,
        GetMockNcclSleepTime(element_count, dtype, comm, gpu_stream, &info));
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
    stream.ThenMemZero(&dest_addr, dest_addr.size());
  }
  return OkStatus();
}

}  // namespace gpu
}  // namespace xla
