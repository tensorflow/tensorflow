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

#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::gpu_dot_fusion_cost_model {

namespace detail {

namespace {
using ::xla::primitive_util::BitWidth;

struct BandwidthEntry {
  int64_t dma_size_bytes;
  float bandwidth_gbps;
};

struct MemoryBandwidthSpec {
  double l2_cache_bandwidth_bytes_per_sec;
  absl::InlinedVector<BandwidthEntry, 32> hbm_bandwidth_table_gbps;
};

MemoryBandwidthSpec GetMemoryBandwidthSpec(
    const se::DeviceDescription& device_info) {
  // Empirical HBM bandwidth scaling table (dma_size_bytes -> fraction of peak
  // memory bandwidth), obtained from microbenchmarks on H100 SXM (peak
  // bandwidth: 3352.32 GB/s).
  // TODO(karupayun): Add explicit microbenchmarked tables for Blackwell
  // (B200/GB200) and other architectures here as they are measured.
  static constexpr std::array<BandwidthEntry, 18> kScaledHbmBandwidthTable = {
      {{8192, 0.00042359f},
       {16384, 0.00090385f},
       {32768, 0.00179577f},
       {65536, 0.00351100f},
       {131072, 0.00706376f},
       {262144, 0.01412455f},
       {524288, 0.02761073f},
       {1048576, 0.05341375f},
       {2097152, 0.10343583f},
       {4194304, 0.19072761f},
       {8388608, 0.31917596f},
       {16777216, 0.47249365f},
       {33554432, 0.58906066f},
       {67108864, 0.69897562f},
       {134217728, 0.78541428f},
       {268435456, 0.82530600f},
       {536870912, 0.88562244f},
       {1073741824, 0.93248850f}}};

  constexpr double kBytesPerGigabyte = 1 << 30;
  double device_memory_bandwidth_gbps =
      static_cast<double>(device_info.memory_bandwidth()) / kBytesPerGigabyte;

  absl::InlinedVector<BandwidthEntry, 32> device_hbm_bandwidth_table(
      kScaledHbmBandwidthTable.begin(), kScaledHbmBandwidthTable.end());
  for (auto& entry : device_hbm_bandwidth_table) {
    entry.bandwidth_gbps *= device_memory_bandwidth_gbps;
  }

  // L2 cache bandwidth scales proportionally to memory bandwidth relative to
  // H100 SXM (measured at 6.65 TB/s on H100 SXM).
  // TODO(maniananth): L2 bandwidth has been hardcoded for H100 based on
  // microbenchmarking L2 bandwidth within a partition, but we should add this
  // to the device info and extend for more GPUs.
  constexpr double kH100L2CacheBandwidthBytesPerSec = 6.65 * 1e12;
  // Peak memory bandwidth of H100 SXM in GB/s (3352.32 GB/s) from
  // xla/backends/gpu/target_config/specs/h100_sxm.txtpb:L26.
  constexpr double kH100SxmPeakMemoryBandwidthGBps = 3352.32;
  double bandwidth_scale =
      device_memory_bandwidth_gbps / kH100SxmPeakMemoryBandwidthGBps;

  return MemoryBandwidthSpec{
      /*l2_cache_bandwidth_bytes_per_sec=*/kH100L2CacheBandwidthBytesPerSec *
          bandwidth_scale,
      /*hbm_bandwidth_table_gbps=*/device_hbm_bandwidth_table,
  };
}

int64_t CalculateNumThreadblocks(const DotProblemInfo& dot,
                                 const DotTileSize& dot_tile) {
  // TODO(maniananth): Add special handling for grouped matmuls here.
  int64_t num_tiles_along_b_dimension = CeilOfRatio<int64_t>(dot.b, dot_tile.b);
  int64_t num_tiles_along_m_dimension = CeilOfRatio<int64_t>(dot.m, dot_tile.m);
  int64_t num_tiles_along_n_dimension = CeilOfRatio<int64_t>(dot.n, dot_tile.n);
  int64_t num_threadblocks = num_tiles_along_b_dimension *
                             num_tiles_along_m_dimension *
                             num_tiles_along_n_dimension;

  return num_threadblocks;
}

int64_t CalculateNumWaves(int64_t threadblock_count,
                          const se::DeviceDescription& device_info) {
  int64_t core_count = device_info.core_count();
  return CeilOfRatio<int64_t>(threadblock_count, core_count);
}

int64_t CalculateTileFlops(const DotTileSize& dot_tile, int64_t problem_k) {
  return /*2 FLOPs per MAC*/ 2 * dot_tile.b * dot_tile.m * dot_tile.n *
         problem_k;
}

// Calculates the effective flops for a GPU DOT operation as a function of the
// tile size (excludes clock throttling). Not all tile sizes are equally able to
// extract utilization on the same generation GPUs even if the workload is
// compute bound. GEMM performance is sensitive to the tensor core
// instruction throughputs that the programming model exposes.
double GetEffectiveFlopsPerNsForTileSize(
    const int64_t tile_m, const se::DeviceDescription& device_info,
    xla::PrimitiveType element_type) {
  se::CudaComputeCapability cuda_compute_capability =
      device_info.cuda_compute_capability();

  // Peak flops per ns for device.
  int64_t peak_flops_per_ns =
      GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(device_info,
                                                           element_type);

  // Final flops derate factor.
  double flops_derate = 1.0;

  if (cuda_compute_capability.IsBlackwell()) {
    if (tile_m < 128) {
      // TODO(maniananth): Update this derate once we have more data from
      // actual measurements on Blackwell. For now, we are applying a 50%
      // derate to account for smaller M shapes.
      flops_derate = 0.5;
    }
  } else if (cuda_compute_capability.IsHopper()) {
    if (tile_m < 64) {
      // Having a tile size M < 64 will lead to not being able to use the H100
      // tensor core instructions (wgmma). Defaulting to wmma instructions from
      // A100 can result in a 63% derate in flops as benchmarked by HazyResearch
      // as part of ThunderKittens work.
      // (https://hazyresearch.stanford.edu/blog/2024-05-12-tk)
      flops_derate = 0.63;
    }
  } else if (cuda_compute_capability.IsAmpere()) {
    if (tile_m < 16) {
      // A100 tensor core instructions are effective at tile_m >= 16. We're
      // applying a 50% derate to account for this.
      flops_derate = 0.5;
    }
  }
  return peak_flops_per_ns * flops_derate;
}

int64_t CalculateL2Bytes(const DotProblemInfo& dot, const DotTileSize& out_tile,
                         int64_t threadblock_count) {
  // When tiling the GEMM problem on the outputs and mapping one tile per SM,
  // the problem of data replication (or extra loads of the same data) between
  // multiple SMs occurs. This leads to more data loads than what’s expected
  // algorithmically, and increases bandwidth needs on the L2 → SM paths.

  // Input data loaded by each tile is equal to (Tile_M + Tile_N) * problem_k
  // bytes (The threadblock iterates over the entire problem_k dimension).
  int64_t lhs_bytes = CeilOfRatio<int64_t>(
      out_tile.b * out_tile.m * dot.k * BitWidth(dot.lhs_element_type), 8);
  int64_t rhs_bytes = CeilOfRatio<int64_t>(
      out_tile.b * out_tile.n * dot.k * BitWidth(dot.rhs_element_type), 8);
  int64_t l2_data_per_tile = lhs_bytes + rhs_bytes;

  // Across all the tiles, data loads will be equal to: (l2_data_per_tile *
  // threadblock_count).

  // TODO(maniananth): Since H100, threadblocks within the same cluster will
  // avoid redundant loads by reading from L2 cache once and multicasting the
  // data to all threadblocks within the cluster. This is controlled
  // programmatically and most performant GEMM implementations will use this
  // feature. To model this, we scale the total data loads by the total number
  // of threadblocks in a cluster.

  // On A100 and older GPUs, we will not see this behavior and the total data
  // loads will be equal to (l2_data_per_tile * threadblock_count). Hence the
  // cluster shape can be set to (1x1).
  // TODO(maniananth): Account for Threadblock clusters here.
  int64_t total_l2_data = l2_data_per_tile * threadblock_count;
  return total_l2_data;
}

}  // namespace

DotProblemInfo::DotProblemInfo(const HloDotInstruction& dot) {
  const Shape& lhs_shape = dot.operand(0)->shape();
  const Shape& rhs_shape = dot.operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot.dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  // We support 4D and higher rank GEMMs to handle multi-dimensional batching
  // (such as having independent head and batch dimensions in multi-head
  // attention workloads) without requiring explicit reshape or flattening ops.
  b = 1;
  for (int64_t batch_dim_idx : dim_numbers.lhs_batch_dimensions()) {
    b *= lhs_shape.dimensions(batch_dim_idx);
  }
  m = lhs_shape.dimensions(lhs_non_contracting_dims[0]);
  n = rhs_shape.dimensions(rhs_non_contracting_dims[0]);
  k = lhs_shape.dimensions(dim_numbers.lhs_contracting_dimensions()[0]);

  lhs_element_type = dot.operand(0)->shape().element_type();
  rhs_element_type = dot.operand(1)->shape().element_type();
  output_element_type = dot.shape().element_type();
}

absl::StatusOr<ComputeAndFlops> CalculateComputeTimeWithTileAndWaveQuantization(
    const DotProblemInfo& dot, const DotTileSize& dot_tile,
    const se::DeviceDescription& device_info) {
  int64_t threadblock_count = CalculateNumThreadblocks(dot, dot_tile);
  int64_t wave_count = CalculateNumWaves(threadblock_count, device_info);
  int64_t flops_per_tile = CalculateTileFlops(dot_tile, dot.k);
  // The following is not the actual number of threadblocks launched, but due to
  // how wave quantization works, we get the effect of running extra
  // threadblocks when adding to roofline projections.
  int64_t cta_count_with_wave_quant = wave_count * device_info.core_count();
  int64_t total_flops_with_wave_quant =
      flops_per_tile * cta_count_with_wave_quant;
  double effective_flops_rate = GetEffectiveFlopsPerNsForTileSize(
      dot_tile.m, device_info, dot.lhs_element_type);

  ComputeAndFlops result;
  result.flops_with_wave_quant = total_flops_with_wave_quant;
  // TODO(maniananth): Add a cap for power throttling here.
  result.compute_time = absl::Nanoseconds(1.0f * total_flops_with_wave_quant /
                                          effective_flops_rate);

  return result;
}

absl::StatusOr<absl::Duration> CalculateL2Time(
    int64_t dot_k, int64_t tile_k, const se::DeviceDescription& device_info,
    int64_t l2_bytes_read, bool is_tma_allowed) {
  double l2_cache_bandwidth_bytes_per_sec =
      GetMemoryBandwidthSpec(device_info).l2_cache_bandwidth_bytes_per_sec;
  int64_t num_k_iters = CeilOfRatio<int64_t>(dot_k, tile_k);

  // Empirical overheads per K-dimension iteration.
  // The overhead is dictated by the memory instruction pathway rather than
  // strictly the hardware generation.
  // Tuned via grid search to minimize MAPE.
  constexpr double kTmaLoopOverheadSeconds =
      150 * 1e-9;  // Fast path (cp.async.bulk)
  constexpr double kLegacyLoopOverheadSeconds =
      400 * 1e-9;  // Slow path (cp.async)
  double k_loop_overhead =
      is_tma_allowed ? kTmaLoopOverheadSeconds : kLegacyLoopOverheadSeconds;

  double base_time_seconds =
      1.0 * l2_bytes_read / l2_cache_bandwidth_bytes_per_sec;
  return absl::Seconds(base_time_seconds + num_k_iters * k_loop_overhead);
}

// Returns the effective HBM bandwidth in bytes per second for a given dma_size.
// dma_size is the total amount of data transferred to/from HBM in bytes.
float GetEffectiveHbmBandwidth(int64_t dma_size,
                               const se::DeviceDescription& device_info) {
  constexpr float kBytesPerGigabyte = 1 << 30;
  MemoryBandwidthSpec spec = GetMemoryBandwidthSpec(device_info);
  const absl::InlinedVector<BandwidthEntry, 32>& hbm_bandwidth_table_gbps =
      spec.hbm_bandwidth_table_gbps;

  if (dma_size <= hbm_bandwidth_table_gbps.front().dma_size_bytes) {
    return hbm_bandwidth_table_gbps.front().bandwidth_gbps * kBytesPerGigabyte;
  }
  if (dma_size >= hbm_bandwidth_table_gbps.back().dma_size_bytes) {
    return hbm_bandwidth_table_gbps.back().bandwidth_gbps * kBytesPerGigabyte;
  }

  auto it2 = std::lower_bound(
      hbm_bandwidth_table_gbps.begin(), hbm_bandwidth_table_gbps.end(),
      dma_size,
      [](const BandwidthEntry& a, int64_t b) { return a.dma_size_bytes < b; });
  auto it1 = it2 - 1;

  // Linear interpolation between the two entries in the lookup table. std::lerp
  // is not used as it is only available since C++20.
  float a = it1->bandwidth_gbps;
  float b = it2->bandwidth_gbps;
  float t = (dma_size - it1->dma_size_bytes) /
            static_cast<float>(it2->dma_size_bytes - it1->dma_size_bytes);
  return (a + t * (b - a)) * kBytesPerGigabyte;
}

HbmEstimates CalculateHbmTime(const DotProblemInfo& dot,
                              const se::DeviceDescription& device_info) {
  // Calculate the number of bytes for input reads and output writes to HBM.
  int64_t lhs_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.m * dot.k * BitWidth(dot.lhs_element_type), 8);
  int64_t rhs_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.k * dot.n * BitWidth(dot.rhs_element_type), 8);
  int64_t output_tile_bytes = CeilOfRatio<int64_t>(
      dot.b * dot.m * dot.n * BitWidth(dot.output_element_type), 8);

  // Main loop loads the input matrices from HBM using SW pipelining and updates
  // accumulators stored in register files (within the SM/compute unit). The
  // epilogue loop writes the output matrices from register files to HBM. Main
  // loop and epilogue loop are executed sequentially.
  int64_t main_loop_bytes = lhs_tile_bytes + rhs_tile_bytes;
  int64_t epilogue_bytes = output_tile_bytes;

  HbmEstimates result;
  result.bytes_read = main_loop_bytes;
  result.bytes_written = epilogue_bytes;

  // Calculate the effective HBM bandwidth for the input and output bytes using
  // the derate lookup table.
  float dram_bandwidth =
      GetEffectiveHbmBandwidth(main_loop_bytes + epilogue_bytes, device_info);

  // Calculate the HBM time using the effective bandwidth for each transfer
  // size. In the current implementation, we are assuming that the main loop and
  // epilogue loop have the same effective DRAM bandwidth. This could change in
  // the future, if we choose to model it based on their respective transfer
  // sizes.
  result.read_time = absl::Seconds(1.0f * (main_loop_bytes) / dram_bandwidth);
  result.write_time = absl::Seconds(1.0f * (epilogue_bytes) / dram_bandwidth);

  return result;
}

absl::Duration CalculatePipelinedLoopTime(int64_t num_stages,
                                          int64_t k_loop_iterations,
                                          absl::Duration compute_time,
                                          const HbmEstimates& hbm_timing) {
  if (num_stages <= 1 || k_loop_iterations <= 1) {
    // Serial execution: Memory and compute are not overlapped.
    return hbm_timing.total_time() + compute_time +
           k_loop_iterations * kLoopLatencyTax;
  }
  // Pipelined execution: Calculate the compute and memory per loop iteration.
  const absl::Duration iter_compute_time = compute_time / k_loop_iterations;
  const absl::Duration iter_raw_mem_time =
      hbm_timing.read_time / k_loop_iterations;
  const absl::Duration iter_mem_time = iter_raw_mem_time + kLoopLatencyTax;

  // In a perfect pipeline with infinite stages, the latency tax should
  // disappear completely.
  const absl::Duration theoretical_iter_time =
      std::max(iter_raw_mem_time, iter_compute_time);
  const absl::Duration iter_time_including_latency =
      std::max(iter_mem_time, iter_compute_time);
  // TODO(b/529318599): Perfect overlap between compute and memory is not
  // always possible in practice. Here we should consider a deeper formula
  // that takes into account num_warps, num_stages and possibly other
  // parameters. I will investigate this further in a follow-up, but this
  // formula works well in practice and is a good starting point.
  const absl::Duration iter_time = std::max(
      theoretical_iter_time, iter_time_including_latency / (num_stages - 1));

  // During the first num_stages-1 iterations, only memory operations are
  // executed.
  const int64_t prologue_loops = std::min(num_stages - 1, k_loop_iterations);
  const absl::Duration prologue_time = prologue_loops * iter_mem_time;

  // During the overlap iterations, both compute and memory operations are
  // executed.
  const int64_t overlap_loops = k_loop_iterations - prologue_loops;
  const absl::Duration overlap_time = overlap_loops * iter_time;

  // During the last num_stages-1 iterations, only compute operations are
  // executed.
  const absl::Duration epilogue_time = prologue_loops * iter_compute_time;

  return prologue_time + overlap_time + epilogue_time + hbm_timing.write_time;
}

SmOccupancy CalculateSmOccupancy(int64_t shared_memory_per_block_bytes,
                                 int64_t num_warps,
                                 const se::DeviceDescription& device_info) {
  const int64_t hardware_max_shmem = device_info.shared_memory_per_core();
  const int64_t hardware_max_threads = device_info.threads_per_core_limit();
  const int64_t max_blocks_by_shmem =
      shared_memory_per_block_bytes > 0
          ? hardware_max_shmem / shared_memory_per_block_bytes
          : hardware_max_threads;
  const int64_t max_blocks_by_threads =
      hardware_max_threads / (num_warps * device_info.threads_per_warp());

  int64_t active_blocks_per_sm = std::max<int64_t>(
      1, std::min(max_blocks_by_shmem, max_blocks_by_threads));

  // Clamp to the physical limit of blocks per SM, if the device provides it.
  if (device_info.max_blocks_per_multiprocessor() > 0) {
    active_blocks_per_sm = std::min(
        active_blocks_per_sm, device_info.max_blocks_per_multiprocessor());
  }

  return SmOccupancy{active_blocks_per_sm, active_blocks_per_sm * num_warps};
}

int64_t CalculateHardwareLaunchWaves(int64_t threadblock_count,
                                     int64_t shared_memory_per_block_bytes,
                                     int64_t num_warps,
                                     const se::DeviceDescription& device_info) {
  const SmOccupancy occupancy = CalculateSmOccupancy(
      shared_memory_per_block_bytes, num_warps, device_info);

  const int64_t total_gpu_capacity =
      occupancy.active_blocks_per_sm * device_info.core_count();
  return CeilOfRatio<int64_t>(threadblock_count, total_gpu_capacity);
}

absl::Duration CalculatePipelinedLoopTimeWithLaunchWaves(
    int64_t num_stages, int64_t k_loop_iterations, int64_t threadblock_count,
    absl::Duration compute_time, const HbmEstimates& hbm_timing,
    int64_t shared_memory_per_block_bytes, int64_t num_warps,
    const se::DeviceDescription& device_info) {
  if (threadblock_count == 0) {
    return absl::ZeroDuration();
  }

  const int64_t launch_waves = CalculateHardwareLaunchWaves(
      threadblock_count, shared_memory_per_block_bytes, num_warps, device_info);

  // Evaluate the pipeline loop per-wave so the latency tax isn't diluted.
  // The total execution time is then the cost of a single wave multiplied by
  // the number of sequentially executed waves.
  const absl::Duration single_wave_compute = compute_time / launch_waves;

  HbmEstimates single_wave_hbm;
  single_wave_hbm.read_time = hbm_timing.read_time / launch_waves;
  single_wave_hbm.write_time = hbm_timing.write_time / launch_waves;

  return CalculatePipelinedLoopTime(num_stages, k_loop_iterations,
                                    single_wave_compute, single_wave_hbm) *
         launch_waves;
}

int64_t CalculateLoopIterBytes(const DotProblemInfo& dot,
                               const DotTileSize& dot_tile) {
  int64_t lhs_iter_bytes = CeilOfRatio<int64_t>(
      dot_tile.b * dot_tile.m * dot_tile.k * BitWidth(dot.lhs_element_type), 8);
  int64_t rhs_iter_bytes = CeilOfRatio<int64_t>(
      dot_tile.b * dot_tile.k * dot_tile.n * BitWidth(dot.rhs_element_type), 8);
  return lhs_iter_bytes + rhs_iter_bytes;
}

int64_t CalculateSharedMemoryPerBlockBytes(const DotProblemInfo& dot_info,
                                           const DotTileSize& dot_tile,
                                           int64_t num_stages) {
  const int64_t lhs_tile_bytes =
      dot_tile.m * dot_tile.k *
      primitive_util::BitWidth(dot_info.lhs_element_type) / 8;
  const int64_t rhs_tile_bytes =
      dot_tile.n * dot_tile.k *
      primitive_util::BitWidth(dot_info.rhs_element_type) / 8;

  return (lhs_tile_bytes + rhs_tile_bytes) * num_stages;
}

double CalculateComputeUtilization(const EstimateRunTimeData& estimates,
                                   const se::DeviceDescription& device_info,
                                   xla::PrimitiveType output_element_type) {
  const double total_estimated_sec = absl::ToDoubleSeconds(estimates.exec_time);
  constexpr double kNsPerSecond = 1e9;
  const double theoretical_flops_per_sec =
      GpuPerformanceModelBase::CalculatePeakMatrixOpsPerNs(
          device_info, output_element_type) *
      kNsPerSecond;

  if (total_estimated_sec == 0.0 || theoretical_flops_per_sec == 0.0) {
    VLOG(2) << "Returning 0.0 compute utilization: total_estimated_sec="
            << total_estimated_sec
            << ", theoretical_flops_per_sec=" << theoretical_flops_per_sec;
    return 0.0;
  }
  double utilization = (static_cast<double>(estimates.flops) /
                        (theoretical_flops_per_sec * total_estimated_sec));

  if (utilization > 1.0) {
    VLOG(2) << "Compute utilization exceeded 1.0 in dot fusion cost model: "
            << utilization;
  }
  return utilization;
}

double CalculateMemoryUtilization(const EstimateRunTimeData& estimates,
                                  const se::DeviceDescription& device_info) {
  const double total_estimated_sec = absl::ToDoubleSeconds(estimates.exec_time);
  const double dram_bytes =
      static_cast<double>(estimates.bytes_read + estimates.bytes_written);
  const double peak_memory_bandwidth = device_info.memory_bandwidth();

  if (total_estimated_sec == 0.0 || peak_memory_bandwidth == 0.0) {
    VLOG(2) << "Returning 0.0 memory utilization: total_estimated_sec="
            << total_estimated_sec
            << ", peak_memory_bandwidth=" << peak_memory_bandwidth;
    return 0.0;
  }
  double utilization =
      dram_bytes / (peak_memory_bandwidth * total_estimated_sec);

  if (utilization > 1.0) {
    VLOG(2) << "Memory utilization exceeded 1.0 in dot fusion cost model: "
            << utilization;
  }
  return utilization;
}

}  // namespace detail

absl::Status IsSupported(const HloDotInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  const Shape& rhs_shape = dot->operand(1)->shape();
  const DotDimensionNumbers& dim_numbers = dot->dot_dimension_numbers();

  DimensionVector lhs_non_contracting_dims = GetNonContractingDims(
      lhs_shape.dimensions().size(), dim_numbers.lhs_contracting_dimensions(),
      dim_numbers.lhs_batch_dimensions());
  DimensionVector rhs_non_contracting_dims = GetNonContractingDims(
      rhs_shape.dimensions().size(), dim_numbers.rhs_contracting_dimensions(),
      dim_numbers.rhs_batch_dimensions());

  if (lhs_non_contracting_dims.size() > 1 ||
      rhs_non_contracting_dims.size() > 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Multiple non-contracting dimensions are not supported, got LHS: [",
        absl::StrJoin(lhs_non_contracting_dims, ","), "], RHS: [",
        absl::StrJoin(rhs_non_contracting_dims, ","), "]"));
  }
  if (dim_numbers.lhs_contracting_dimensions_size() != 1 ||
      dim_numbers.rhs_contracting_dimensions_size() != 1) {
    return absl::UnimplementedError(absl::StrCat(
        "Exactly one contracting dimension is supported, got LHS: [",
        absl::StrJoin(dim_numbers.lhs_contracting_dimensions(), ","),
        "], RHS: [",
        absl::StrJoin(dim_numbers.rhs_contracting_dimensions(), ","), "]"));
  }

  // TODO(b/501002656): Support downstream transposes by fixing dimension
  // mapping.
  std::vector<const HloInstruction*> stack;
  absl::flat_hash_set<const HloInstruction*> visited;
  stack.push_back(dot);
  visited.insert(dot);
  while (!stack.empty()) {
    const HloInstruction* current = stack.back();
    stack.pop_back();
    if (current != dot && current->opcode() == HloOpcode::kTranspose) {
      return absl::UnimplementedError(
          "Dot with a downstream transpose is not supported.");
    }
    for (const HloInstruction* user : current->users()) {
      if (visited.insert(user).second) {
        stack.push_back(user);
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<int64_t> ExtractBlockK(const HloDotInstruction* dot) {
  if (!dot->has_backend_config()) {
    return absl::FailedPreconditionError(
        "Dot instruction must have a backend config with tiling sizes.");
  }
  ABSL_ASSIGN_OR_RETURN(auto tile_config, dot->backend_config<xla::gpu::Tile>());
  TF_RET_CHECK(tile_config.sizes_size() > 0)
      << "Tile backend config must have sizes.";
  return tile_config.sizes(0);
}

absl::StatusOr<EstimateRunTimeData> EstimateRunTimeForDotOpWithBlockParameters(
    const HloDotInstruction* dot, const BlockLevelParameters& block_params,
    const se::DeviceDescription& device_info, std::optional<int64_t> block_k) {
  ABSL_RETURN_IF_ERROR(IsSupported(dot));
  if (block_params.output_tile_sizes.size() != 1) {
    return absl::UnimplementedError(
        absl::StrCat("Only single tile size is supported, got ",
                     block_params.output_tile_sizes.size()));
  }

  int64_t block_k_val;
  if (block_k.has_value()) {
    block_k_val = *block_k;
  } else {
    ABSL_ASSIGN_OR_RETURN(block_k_val, ExtractBlockK(dot));
  }

  detail::DotProblemInfo dot_info(*dot);

  const std::vector<int64_t>& tile_shape = block_params.output_tile_sizes[0];
  if (tile_shape.size() < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Tile shape must be of size at least 2, got ", tile_shape.size()));
  }
  int64_t tile_b = 1;
  for (size_t i = 0; i < tile_shape.size() - 2; ++i) {
    tile_b *= tile_shape[i];
  }
  int64_t tile_m = tile_shape[tile_shape.size() - 2];
  int64_t tile_n = tile_shape[tile_shape.size() - 1];
  detail::DotTileSize dot_tile{/*m=*/tile_m,
                               /*n=*/tile_n,
                               /*k=*/block_k_val,
                               /*b=*/tile_b};

  EstimateRunTimeData estimates;

  // Calculate compute roofline with tile and wave quantization.
  ABSL_ASSIGN_OR_RETURN(detail::ComputeAndFlops compute_and_flops,
                   detail::CalculateComputeTimeWithTileAndWaveQuantization(
                       dot_info, dot_tile, device_info));
  estimates.compute_time = compute_and_flops.compute_time;
  estimates.flops = compute_and_flops.flops_with_wave_quant;

  // Calculate HBM roofline.
  detail::HbmEstimates hbm_timing =
      detail::CalculateHbmTime(dot_info, device_info);

  estimates.read_time = hbm_timing.read_time;
  estimates.write_time = hbm_timing.write_time;
  estimates.bytes_read = hbm_timing.bytes_read;
  estimates.bytes_written = hbm_timing.bytes_written;
  const int64_t num_stages = block_params.num_stages;

  int64_t threadblock_count =
      detail::CalculateNumThreadblocks(dot_info, dot_tile);
  estimates.l2_bytes_read =
      detail::CalculateL2Bytes(dot_info, dot_tile, threadblock_count);

  estimates.shared_memory_per_block_bytes =
      detail::CalculateSharedMemoryPerBlockBytes(dot_info, dot_tile,
                                                 num_stages);

  // Calculate L2 time.
  ABSL_ASSIGN_OR_RETURN(absl::Duration l2_time,
                   detail::CalculateL2Time(dot_info.k, dot_tile.k, device_info,
                                           estimates.l2_bytes_read,
                                           block_params.is_tma_allowed));

  TF_RET_CHECK(block_k_val > 0)
      << "block_k_val must be strictly positive, got " << block_k_val;
  TF_RET_CHECK(dot_info.k > 0)
      << "dot_info.k must be strictly positive, got " << dot_info.k;
  const int64_t k_loop_iterations =
      CeilOfRatio<int64_t>(dot_info.k, block_k_val);

  absl::Duration pipelined_loop_time =
      detail::CalculatePipelinedLoopTimeWithLaunchWaves(
          num_stages, k_loop_iterations, threadblock_count,
          compute_and_flops.compute_time, hbm_timing,
          estimates.shared_memory_per_block_bytes, block_params.num_warps,
          device_info);

  // Assuming perfect overlap between compute and memory for the rest,
  // but main loop is now modeled precisely.
  estimates.exec_time = std::max({pipelined_loop_time, l2_time});
  estimates.compute_utilization = detail::CalculateComputeUtilization(
      estimates, device_info, dot_info.output_element_type);
  estimates.memory_utilization =
      detail::CalculateMemoryUtilization(estimates, device_info);

  return estimates;
}

}  // namespace xla::gpu::gpu_dot_fusion_cost_model
