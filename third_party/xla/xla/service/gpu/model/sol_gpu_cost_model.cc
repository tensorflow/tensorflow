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

#include "xla/service/gpu/model/sol_gpu_cost_model.h"

#include <cmath>
#include <cstdint>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/strings/numbers.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/collective_utils.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr double kHeaderOverhead = 0.025;

constexpr char kUnknownKey[] = "<unknown>";

static auto& device_to_cfg =
    *(new absl::flat_hash_map<std::string, SolGPUCostModel::Config>({
        {
            "NVIDIA H100 80GB HBM3",
            {
                /*nccl_op_launch_time=*/absl::Microseconds(
                    100.0f * kDefaultNcclCostModelCoeff),
                /*nic_speed_gbps=*/
                55.56f * kDefaultNcclCostModelCoeff,
                /*chunk_prep_time=*/
                absl::Microseconds(13.34f * kDefaultNcclCostModelCoeff),
                /*rtt=*/
                absl::Microseconds(68.89f * kDefaultNcclCostModelCoeff),
                /*gpus_per_node=*/8,
                /*chunk_size_bytes=*/kDefaultNcclCostModelChunkSizeBytes,
            },
        },
        {
            kUnknownKey,
            {
                /*nccl_op_launch_time=*/absl::Microseconds(
                    100.0f * kDefaultNcclCostModelCoeff),
                /*nic_speed_gbps=*/
                55.56f * kDefaultNcclCostModelCoeff,
                /*chunk_prep_time=*/
                absl::Microseconds(13.34f * kDefaultNcclCostModelCoeff),
                /*rtt=*/
                absl::Microseconds(68.89f * kDefaultNcclCostModelCoeff),
                /*gpus_per_node=*/8,
                /*chunk_size_bytes=*/kDefaultNcclCostModelChunkSizeBytes,
            },
        },
    }));

// Returns the number of communicators in the mask.
// For example, if the mask is 0x0, this function returns 1. If the mask is 0x7,
// this function returns 8.
int NumCommunicators(const absl::string_view mask) {
  // Assuming the mask is a hexadecimal number
  uint64_t mask_value = std::stoul(std::string(mask), nullptr, 16);
  int bit_count = absl::popcount(mask_value);  // Count set bits
  return static_cast<int>(std::pow(2, bit_count));
}

// Returns the number of rounds for the given collective type.
int NumRounds(const SolGPUCostModel::CollectiveType& coll_type) {
  // AllReduce requires ReduceScatter and AllGather, so it has 2 rounds.
  return coll_type == SolGPUCostModel::CollectiveType::kAllReduce ? 2 : 1;
}

SolGPUCostModel::Config GetPlatformConfig(
    const se::DeviceDescription& device_info) {
  std::string key = device_info.name();
  if (!device_to_cfg.contains(key)) {
    return device_to_cfg[kUnknownKey];
  }
  return device_to_cfg[key];
}

}  // namespace

/*static*/ SolGPUCostModel::Config SolGPUCostModel::GetConfig(
    const HloModule* module, const se::DeviceDescription& device_info) {
  SolGPUCostModel::Config config = GetPlatformConfig(device_info);
  const auto& extra_options =
      module->config()
          .debug_options()
          .xla_gpu_analytical_latency_estimator_options();
  for (const auto& [option_name, option_value] : extra_options) {
    int64_t value;
    double value_d;
    VLOG(2) << "[SoL] extra option: " << option_name << " is " << option_value;
    if (option_name == kSolNcclOpLaunchUs &&
        absl::SimpleAtoi(option_value, &value) && value > 0) {
      config.nccl_op_launch_time = absl::Microseconds(value);
    } else if (option_name == kSolNicSpeedGbps &&
               absl::SimpleAtod(option_value, &value_d) && value_d > 0.0) {
      config.nic_speed_gbps = value_d;
    } else if (option_name == kSolChunkPrepUs &&
               absl::SimpleAtoi(option_value, &value) && value > 0) {
      config.chunk_prep_time = absl::Microseconds(value);
    } else if (option_name == kSolRttUs &&
               absl::SimpleAtoi(option_value, &value) && value > 0) {
      config.rtt = absl::Microseconds(value);
    } else if (option_name == kSolGpusPerNode &&
               absl::SimpleAtoi(option_value, &value) && value > 0) {
      config.gpus_per_node = value;
    } else if (option_name == kSolChunkSizeBytes &&
               absl::SimpleAtoi(option_value, &value) && value > 0) {
      config.chunk_size_bytes = value;
    }
  }
  return config;
}

SolGPUCostModel::SolGPUCostModel(const Config& sys_config)
    : xla_flag_config_(sys_config) {
  VLOG(2) << "[SoL] NIC speed: " << xla_flag_config_.nic_speed_gbps;
  VLOG(2) << "[SoL] RTT: " << xla_flag_config_.rtt;
  VLOG(2) << "[SoL] Chunk preparation time: "
          << xla_flag_config_.chunk_prep_time;
  VLOG(2) << "[SoL] NCCL op launch time: "
          << xla_flag_config_.nccl_op_launch_time;
  VLOG(2) << "[SoL] GPUs per node: " << xla_flag_config_.gpus_per_node;
}

// This is a insignificant term, and we are making it consistent
// with the existing formula.
absl::Duration SolGPUCostModel::ChunkPrepLatency(
    const int64_t per_gpu_msg_size_bytes) const {
  return std::ceil(static_cast<double>(per_gpu_msg_size_bytes) /
                   xla_flag_config_.chunk_size_bytes) *
         xla_flag_config_.chunk_prep_time;
}

absl::Duration SolGPUCostModel::TransferDuration(
    const int64_t per_gpu_msg_size_bytes) const {
  // x1e6 to convert secs to microseconds;
  // x10^9 to convert Gbytes/sec to bytes/sec
  const long double ret =
      (1e6 * static_cast<long double>(per_gpu_msg_size_bytes)) /
      (1e9 * xla_flag_config_.nic_speed_gbps);
  return absl::Microseconds(ret * (1 + kHeaderOverhead));
}

absl::Duration SolGPUCostModel::RingLatency(
    const int64_t buff_size_bytes, const int num_nodes,
    const CollectiveType& coll_type, const absl::string_view mask) const {
  const int num_gpus = NumGpusPerComm(num_nodes, coll_type, mask);

  int64_t per_gpu_msg_size_bytes;
  if (coll_type == CollectiveType::kSendRecv) {
    per_gpu_msg_size_bytes = buff_size_bytes;
  } else {
    per_gpu_msg_size_bytes = buff_size_bytes / num_gpus;
  }

  // This is the number of GPUs per communicator per node. We assume that each
  // GPU has a NIC, and this is also the number of NICs per communicator per
  // node.
  // Note that this happens to be correct value (i.e. 1) for SendRecv.
  int num_gpus_per_node = num_gpus / num_nodes;

  // In each channel, consider one GPU next to the Ethernet link. Below is the
  // sum of 3 time costs for each piece of data of size
  // `per_gpu_msg_size_bytes`
  //
  // 1. transfer duration defined by the NIC bandwidth,
  // 2. chunk preparation latency, and
  // 3. RTT
  //
  // then followed by two factors:
  //
  // 1. Multiply by `num_gpus - 1`, as `num_gpus - 1` pieces of data will be
  //    sent over the link in AllGather.
  // 2. Divide by `num_gpus_per_node` as there are `num_gpus_per_node` NICs
  // and
  //    GPUs in each node for parallelism.
  //
  // Better estimates of terms like this will come in future versions
  // of the SoL model.
  absl::Duration ret = TransferDuration(per_gpu_msg_size_bytes) +
                       ChunkPrepLatency(per_gpu_msg_size_bytes) +
                       xla_flag_config_.rtt;
  ret *= (num_gpus - 1.0) / static_cast<long double>(num_gpus_per_node);
  // Multiply by the number of rounds, which is different for AllReduce.
  ret = ret * NumRounds(coll_type);

  // Time to initiate the collective.
  return ret + xla_flag_config_.nccl_op_launch_time;
}

// Helper functions
int SolGPUCostModel::NumGpusPerComm(int num_nodes,
                                    const CollectiveType& coll_type,
                                    const absl::string_view mask) const {
  if (coll_type == CollectiveType::kSendRecv) {
    return 2;
  }
  int num_comms = NumCommunicators(mask);
  CHECK_EQ(xla_flag_config_.gpus_per_node % num_comms, 0)
      << "GPU_PER_NODE must be divisible by the number of communicators. "
         "GPU_PER_NODE: "
      << xla_flag_config_.gpus_per_node
      << " Number of communicators: " << num_comms
      << ". Adjust the number of GPUs per node with the flag "
         "gpus_per_node in xla_gpu_analytical_latency_estimator_options.";
  return num_nodes * xla_flag_config_.gpus_per_node / num_comms;
}

}  // namespace gpu
}  // namespace xla
