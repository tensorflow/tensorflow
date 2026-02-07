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

#ifndef XLA_BACKENDS_GPU_COST_MODEL_SOL_GPU_COST_MODEL_H_
#define XLA_BACKENDS_GPU_COST_MODEL_SOL_GPU_COST_MODEL_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Speed-of-Light (SoL) analytical cost model for NCCL collectives.
class SolGPUCostModel {
 public:
  // Tunable system configuration, see
  // xla_gpu_analytical_latency_estimator_options
  struct Config {
    absl::Duration nccl_op_launch_time;
    // it's GBytes/s, not Gbit/s (ex: 40Gb/s = 5GB/s)
    // GBytes per second = 10^9 bytes per second
    double nic_speed_gbps;
    absl::Duration chunk_prep_time;
    absl::Duration rtt;
    int64_t gpus_per_node;
    int64_t chunk_size_bytes;
    // Partition size (devices per fast-interconnect domain). 0 means unset.
    int64_t partition_size;
  };

  enum CollectiveAlgorithmType {
    RING = 0,
    TREE,
  };

  enum class CollectiveType {
    kAllGather,
    kAllReduce,
    kAllToAll,
    kReduceScatter,
    kSendRecv,
  };

  explicit SolGPUCostModel(const Config& sys_config);

  // Extract the SoL-related configuration from XLA flags.
  static SolGPUCostModel::Config GetConfig(
      const HloModule* module, const se::DeviceDescription& device_info);

  // Returns the latency of a NCCL ring collective.
  //
  // `buff_size_bytes`: the size of the message to be transferred.
  // `num_nodes`: the number of nodes participating in the ring.
  // `coll_type`: the type of the collective (eg AllGather).
  // `mask`: the mask of the collective (AllWorld 0x0 vs RailAligned 0x7).
  absl::StatusOr<absl::Duration> RingLatency(int64_t buff_size_bytes,
                                             int num_nodes,
                                             const CollectiveType& coll_type,
                                             int num_communicators) const;

  // Returns the latency of an AllToAll collective across multiple nodes.
  //
  // `buff_size_bytes`: the size of the message to be transferred.
  // `num_nodes`: the number of nodes participating in the all-to-all.
  // `num_communicators`: the number of communicators participating in the
  // all-to-all.
  absl::StatusOr<absl::Duration> AllToAllLatency(int64_t buff_size_bytes,
                                                 int num_nodes,
                                                 int num_communicators) const;

 private:
  // Helper functions to estimate the latency subcomponents
  absl::Duration ChunkPrepLatency(int64_t per_gpu_msg_size_bytes) const;

  absl::Duration TransferDuration(int64_t per_gpu_msg_size_bytes) const;
  // NumGpusPerComm returns  GPUs number participating in a given NCCL
  // collective operation.
  absl::StatusOr<int> NumGpusPerComm(int num_nodes,
                                     const CollectiveType& coll_type,
                                     int num_communicators) const;

  // SoL-related configuration for NCCL cost modelling passed by user as flags.
  Config xla_flag_config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_COST_MODEL_SOL_GPU_COST_MODEL_H_
