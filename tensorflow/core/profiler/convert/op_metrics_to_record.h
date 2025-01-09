/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
#define TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

std::vector<const OpMetrics*> SortedOpMetricsDb(const OpMetricsDb& metrics_db,
                                                int max_records = -1);

inline double GigaFlopsPerSecondPerCore(const OpMetrics& metrics) {
  // flops and time_ps are accumulated across all occurrences on all cores.
  // time_ps is used instead of self_time_ps because flops for an op includes
  // the flops executed by children (nested) ops.
  return tsl::profiler::SafeDivide(
      metrics.flops(), tsl::profiler::PicoToNano(metrics.time_ps()));
}

inline double GigaModelFlopsPerSecondPerCore(const OpMetrics& metrics) {
  // flops and time_ps are accumulated across all occurrences on all cores.
  // time_ps is used instead of self_time_ps because flops for an op includes
  // the flops executed by children (nested) ops.
  return tsl::profiler::SafeDivide(
      metrics.model_flops(), tsl::profiler::PicoToNano(metrics.time_ps()));
}

// Return ByteAccessed for memory_space and operation_type.
inline double BytesAccessedPerCore(
    const OpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType operation_type) {
  uint64_t bytes = 0;
  if (memory_space == MemorySpace::MEMORY_SPACE_ALL) {
    bytes = metrics.bytes_accessed();
  } else {
    for (const auto& breakdown : metrics.memory_accessed_breakdown()) {
      // Count either on-chip or off-chip bytes.
      if ((breakdown.operation_type() != operation_type) &&
          (operation_type != OpMetrics::MemoryAccessed::UNKNOWN)) {
        continue;
      }
      if (((memory_space == MemorySpace::MEMORY_SPACE_HBM) &&
           (breakdown.memory_space() == MemorySpace::MEMORY_SPACE_HBM)) ||
          ((memory_space == MemorySpace::MEMORY_SPACE_ON_CHIP) &&
           (breakdown.memory_space() != MemorySpace::MEMORY_SPACE_HBM))) {
        bytes += breakdown.bytes_accessed();
      }
    }
  }
  return bytes;
}

inline double GigaBytesPerSecondPerCore(
    const OpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType operation_type) {
  // bytes_accessed and time_ps are accumulated across all occurrences on all
  // cores.
  // time_ps is used instead of self_time_ps because bytes_accessed for an op
  // includes the bytes accessed by children (nested) ops.
  return tsl::profiler::SafeDivide(
      BytesAccessedPerCore(metrics, memory_space, operation_type),
      tsl::profiler::PicoToNano(metrics.time_ps()));
}

inline double GibiBytesPerSecondPerCore(
    const OpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType op_type) {
  return tsl::profiler::GigaToGibi(
      GigaBytesPerSecondPerCore(metrics, memory_space, op_type));
}

template <typename Record>
inline void SetExecutionTimes(const OpMetrics& metrics, Record* record) {
  record->set_occurrences(metrics.occurrences());
  record->set_total_time_in_us(tsl::profiler::PicoToMicro(metrics.time_ps()));
  record->set_avg_time_in_us(
      SafeDivide(record->total_time_in_us(), metrics.occurrences()));
  record->set_total_self_time_in_us(
      tsl::profiler::PicoToMicro(metrics.self_time_ps()));
  record->set_avg_self_time_in_us(
      SafeDivide(record->total_self_time_in_us(), metrics.occurrences()));
}

template <typename Record>
inline void SetTpuUnitFractions(const OpMetrics& metrics, Record* record) {
  record->set_dma_stall_fraction(
      tsl::profiler::SafeDivide(metrics.dma_stall_ps(), metrics.time_ps()));
}

template <typename Record>
inline void SetRankAndTimeFractions(double total_time_us,
                                    const Record& prev_record, Record* record) {
  record->set_rank(prev_record.rank() + 1);
  record->set_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_cumulative_total_self_time_as_fraction(
      prev_record.cumulative_total_self_time_as_fraction() +
      record->total_self_time_as_fraction());
}

template <typename Record>
inline void SetRankAndDeviceTimeFractions(double total_time_us,
                                          const Record& prev_record,
                                          Record* record) {
  record->set_rank(prev_record.rank() + 1);
  record->set_device_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_device_cumulative_total_self_time_as_fraction(
      prev_record.device_cumulative_total_self_time_as_fraction() +
      record->device_total_self_time_as_fraction());
}

template <typename Record>
inline void SetRankAndHostTimeFractions(double total_time_us,
                                        const Record& prev_record,
                                        Record* record) {
  record->set_rank(prev_record.rank() + 1);
  record->set_host_total_self_time_as_fraction(
      SafeDivide(record->total_self_time_in_us(), total_time_us));
  record->set_host_cumulative_total_self_time_as_fraction(
      prev_record.host_cumulative_total_self_time_as_fraction() +
      record->host_total_self_time_as_fraction());
}

// Returns the memory bandwidth in GigaBytes/s in the PerfEnv.
// memory space is chosen by index following order in xplane_to_op_stats.cc
static inline double GetMemoryPeakBandwidth(const PerfEnv& perf_env,
                                            const int index) {
  if (perf_env.peak_bws_giga_bytes_per_second_size() > index) {
    return perf_env.peak_bws_giga_bytes_per_second(index);
  }
  return perf_env.peak_hbm_bw_giga_bytes_per_second();
}

template <typename Record>
inline void SetRooflineMetrics(const OpMetrics& metrics, const PerfEnv perf_env,
                               const RunEnvironment& run_env, Record* record) {
  using ::tensorflow::profiler::MemorySpace;
  using ::tensorflow::profiler::PerformanceInfo;
  using ::tensorflow::profiler::PicoToNano;

  // Set overall performance metrics.
  record->set_measured_flop_rate(GigaFlopsPerSecondPerCore(metrics));
  record->set_model_flop_rate(GigaModelFlopsPerSecondPerCore(metrics));
  record->set_measured_memory_bw(GibiBytesPerSecondPerCore(
      metrics, tensorflow::profiler::MemorySpace::MEMORY_SPACE_ALL,
      OpMetrics::MemoryAccessed::UNKNOWN));
  record->set_flops(metrics.flops());
  record->set_bytes_accessed(metrics.bytes_accessed());
  record->set_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), metrics.bytes_accessed()));
  // Set performance metrics per memory access type.
  uint64_t hbm_bytes = 0;
  uint64_t cmem_read_bytes = 0;
  uint64_t cmem_write_bytes = 0;
  uint64_t vmem_read_bytes = 0;
  uint64_t vmem_write_bytes = 0;
  for (const auto& memory_access : metrics.memory_accessed_breakdown()) {
    if (memory_access.memory_space() == PerformanceInfo::MemoryAccessed::HBM) {
      hbm_bytes += memory_access.bytes_accessed();
    } else if (memory_access.memory_space() ==
               PerformanceInfo::MemoryAccessed::CMEM) {
      if (memory_access.operation_type() == OpMetrics::MemoryAccessed::READ) {
        cmem_read_bytes += memory_access.bytes_accessed();
      } else if (memory_access.operation_type() ==
                 OpMetrics::MemoryAccessed::WRITE) {
        cmem_write_bytes += memory_access.bytes_accessed();
      }
    } else if (memory_access.memory_space() ==
               PerformanceInfo::MemoryAccessed::VMEM) {
      if (memory_access.operation_type() == OpMetrics::MemoryAccessed::READ) {
        vmem_read_bytes += memory_access.bytes_accessed();
      } else if (memory_access.operation_type() ==
                 OpMetrics::MemoryAccessed::WRITE) {
        vmem_write_bytes += memory_access.bytes_accessed();
      }
    }
  }
  if (metrics.memory_accessed_breakdown_size() == 0) {
    // For legacy profiles without memory access breakdown, consider all memory
    // access as HBM access.
    hbm_bytes = metrics.bytes_accessed();
  }
  record->set_hbm_bw(tsl::profiler::GibibytesPerSecond(
      hbm_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_cmem_read_bw(tsl::profiler::GibibytesPerSecond(
      cmem_read_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_cmem_write_bw(tsl::profiler::GibibytesPerSecond(
      cmem_write_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_vmem_read_bw(tsl::profiler::GibibytesPerSecond(
      vmem_read_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_vmem_write_bw(tsl::profiler::GibibytesPerSecond(
      vmem_write_bytes, tsl::profiler::PicoToNano(metrics.time_ps())));
  record->set_hbm_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), hbm_bytes));
  record->set_cmem_read_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), cmem_read_bytes));
  record->set_cmem_write_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), cmem_write_bytes));
  record->set_vmem_read_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), vmem_read_bytes));
  record->set_vmem_write_operational_intensity(
      tsl::profiler::SafeDivide(metrics.flops(), vmem_write_bytes));
  // Resources considered for roofline analysis.
  constexpr absl::string_view kUnknown = "Unknown";
  constexpr absl::string_view kCompute = "Compute";
  constexpr absl::string_view kHbm = "HBM";
  constexpr absl::string_view kCmemRead = "CMEM Read";
  constexpr absl::string_view kCmemWrite = "CMEM Write";
  constexpr absl::string_view kVmemRead = "VMEM Read";
  constexpr absl::string_view kVmemWrite = "VMEM Write";
  constexpr absl::string_view kShmL1 = "Shm/L1";
  // Compute the bound time assuming the peak capacity of each resource and
  // choose the highest one as the bottleneck. See go/xprof-roofline-pxc for
  // more details.
  // NOTE: The roofline analysis result is the same for Megacore because every
  // resource's capacity is doubled for Megacore so the comparison result is the
  // same.
  absl::string_view bottleneck_resource = kUnknown;
  double bottleneck_utilization = 0;
  double bottleneck_operational_intensity = 0;
  double peak_flops =
      tsl::profiler::TeraToGiga(perf_env.peak_tera_flops_per_second());
  double flops_utilization =
      SafeDivide(record->measured_flop_rate(), peak_flops);
  if (bottleneck_utilization < flops_utilization) {
    bottleneck_resource = kCompute;
    bottleneck_utilization = flops_utilization;
    bottleneck_operational_intensity = record->operational_intensity();
  }
  double peak_hbm_bw = GetMemoryPeakBandwidth(perf_env, 0);
  double hbm_bw_utilization =
      SafeDivide(record->hbm_bw(), tsl::profiler::GigaToGibi(peak_hbm_bw));
  if (bottleneck_utilization < hbm_bw_utilization) {
    bottleneck_resource = kHbm;
    bottleneck_utilization = hbm_bw_utilization;
    bottleneck_operational_intensity = record->hbm_operational_intensity();
  }
  tensorflow::profiler::HardwareType hardware_type = run_env.hardware_type();
  if (hardware_type == tensorflow::profiler::HardwareType::TPU) {
    if (cmem_read_bytes) {
      double peak_cmem_read_bw = GetMemoryPeakBandwidth(perf_env, 3);
      if (peak_cmem_read_bw) {
        double cmem_read_bw_utilization =
            SafeDivide(record->cmem_read_bw(),
                       tsl::profiler::GigaToGibi(peak_cmem_read_bw));
        if (bottleneck_utilization < cmem_read_bw_utilization) {
          bottleneck_resource = kCmemRead;
          bottleneck_utilization = cmem_read_bw_utilization;
          bottleneck_operational_intensity =
              record->cmem_read_operational_intensity();
        }
      }
    }
    if (cmem_write_bytes) {
      double peak_cmem_write_bw = GetMemoryPeakBandwidth(perf_env, 4);
      if (peak_cmem_write_bw) {
        double cmem_write_bw_utilization =
            SafeDivide(record->cmem_write_bw(),
                       tsl::profiler::GigaToGibi(peak_cmem_write_bw));
        if (bottleneck_utilization < cmem_write_bw_utilization) {
          bottleneck_resource = kCmemWrite;
          bottleneck_utilization = cmem_write_bw_utilization;
          bottleneck_operational_intensity =
              record->cmem_write_operational_intensity();
        }
      }
    }
    if (vmem_read_bytes) {
      double peak_vmem_read_bw = GetMemoryPeakBandwidth(perf_env, 5);
      if (peak_vmem_read_bw) {
        double vmem_read_bw_utilization =
            SafeDivide(record->vmem_read_bw(),
                       tsl::profiler::GigaToGibi(peak_vmem_read_bw));
        if (bottleneck_utilization < vmem_read_bw_utilization) {
          bottleneck_resource = kVmemRead;
          bottleneck_utilization = vmem_read_bw_utilization;
          bottleneck_operational_intensity =
              record->vmem_read_operational_intensity();
        }
      }
    }
    if (vmem_write_bytes) {
      double peak_vmem_write_bw = GetMemoryPeakBandwidth(perf_env, 6);
      if (peak_vmem_write_bw) {
        double vmem_write_bw_utilization =
            SafeDivide(record->vmem_write_bw(),
                       tsl::profiler::GigaToGibi(peak_vmem_write_bw));
        if (bottleneck_utilization < vmem_write_bw_utilization) {
          bottleneck_resource = kVmemWrite;
          bottleneck_utilization = vmem_write_bw_utilization;
          bottleneck_operational_intensity =
              record->vmem_write_operational_intensity();
        }
      }
    }
  }
  if (hardware_type == tensorflow::profiler::HardwareType::GPU) {
    double peak_shm_l1_bw = GetMemoryPeakBandwidth(perf_env, 2);
    if (peak_shm_l1_bw) {
      // Currently, we only have general read/write bandwidth in record.
      double shm_l1_bw_utilization = SafeDivide(
          record->hbm_bw(), tsl::profiler::GigaToGibi(peak_shm_l1_bw));
      if (bottleneck_utilization < shm_l1_bw_utilization) {
        bottleneck_resource = kShmL1;
        bottleneck_utilization = shm_l1_bw_utilization;
        bottleneck_operational_intensity = record->hbm_operational_intensity();
      }
    }
  }
  record->set_bound_by(std::string(bottleneck_resource));
  record->set_bottleneck_operational_intensity(
      bottleneck_operational_intensity);
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
