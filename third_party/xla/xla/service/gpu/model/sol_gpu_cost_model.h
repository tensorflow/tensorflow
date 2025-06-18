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

#ifndef XLA_SERVICE_GPU_MODEL_SOL_GPU_COST_MODEL_H_
#define XLA_SERVICE_GPU_MODEL_SOL_GPU_COST_MODEL_H_

#include <cstdint>

#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {
namespace gpu {

// Speed-of-Light (SoL) analytical cost model for NCCL collectives.
class SolGPUCostModel {
 public:
  static constexpr absl::string_view kSplitMaskWorldLevel = "0x0";

  static constexpr absl::string_view kSplitMaskNonRailAligned = "0x7";

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
  };
  enum CollectiveAlgorithmType {
    RING = 0,
    TREE,
  };
  enum class CollectiveType {
    kAllReduce,
    kAllGather,
    kReduceScatter,
    kSendRecv,
  };
  explicit SolGPUCostModel(const Config& sys_config);

  // Extract the SoL-related configuration from XLA flags.
  static SolGPUCostModel::Config GetConfig(const HloModule* module);

  // Returns the latency of a NCCL ring collective.
  //
  // `buff_size_bytes`: the size of the message to be transferred.
  // `num_nodes`: the number of nodes participating in the ring.
  // `coll_type`: the type of the collective (eg AllGather).
  // `mask`: the mask of the collective (AllWorld 0x0 vs RailAligned 0x7).
  absl::Duration RingLatency(
      int64_t buff_size_bytes, int num_nodes, const CollectiveType& coll_type,
      absl::string_view mask = kSplitMaskWorldLevel) const;

 private:
  // Helper functions to estimate the latency subcomponents
  absl::Duration ChunkPrepLatency(int64_t per_gpu_msg_size_bytes) const;

  absl::Duration TransferDuration(int64_t per_gpu_msg_size_bytes) const;
  // NumGpusPerComm returns  GPUs number participating in a given NCCL
  // collective operation.
  int NumGpusPerComm(int num_nodes, const CollectiveType& coll_type,
                     absl::string_view mask) const;

  // SoL-related configuration for NCCL cost modelling passed by user as flags.
  Config xla_flag_config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SOL_GPU_COST_MODEL_H_
