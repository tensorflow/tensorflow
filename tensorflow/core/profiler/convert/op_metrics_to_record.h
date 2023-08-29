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
#include <vector>

#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"

namespace tensorflow {
namespace profiler {

std::vector<const OpMetrics*> SortedOpMetricsDb(const OpMetricsDb& metrics_db,
                                                int max_records = -1);

inline double GigaFlopsPerSecondPerCore(const OpMetrics& metrics) {
  // flops and time_ps are accumulated across all occurrences on all cores.
  // time_ps is used instead of self_time_ps because flops for an op includes
  // the flops executed by children (nested) ops.
  return SafeDivide(metrics.flops(), PicoToNano(metrics.time_ps()));
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
  return SafeDivide(BytesAccessedPerCore(metrics, memory_space, operation_type),
                    PicoToNano(metrics.time_ps()));
}

inline double GibiBytesPerSecondPerCore(
    const OpMetrics& metrics, uint64_t memory_space,
    OpMetrics::MemoryAccessed::OperationType op_type) {
  return GigaToGibi(GigaBytesPerSecondPerCore(metrics, memory_space, op_type));
}

template <typename Record>
inline void SetExecutionTimes(const OpMetrics& metrics, Record* record) {
  record->set_occurrences(metrics.occurrences());
  record->set_total_time_in_us(PicoToMicro(metrics.time_ps()));
  record->set_avg_time_in_us(
      SafeDivide(record->total_time_in_us(), metrics.occurrences()));
  record->set_total_self_time_in_us(PicoToMicro(metrics.self_time_ps()));
  record->set_avg_self_time_in_us(
      SafeDivide(record->total_self_time_in_us(), metrics.occurrences()));
}

template <typename Record>
inline void SetTpuUnitFractions(const OpMetrics& metrics, Record* record) {
  record->set_dma_stall_fraction(
      SafeDivide(metrics.dma_stall_ps(), metrics.time_ps()));
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

template <typename Record>
inline void SetRooflineMetrics(const OpMetrics& metrics,
                               double ridge_point_operational_intensity,
                               Record* record) {
  using ::tensorflow::profiler::PicoToNano;
  record->set_measured_flop_rate(GigaFlopsPerSecondPerCore(metrics));
  record->set_measured_memory_bw(
      GigaBytesPerSecondPerCore(metrics, MemorySpace::MEMORY_SPACE_ALL,
                                OpMetrics::MemoryAccessed::UNKNOWN));
  record->set_operational_intensity(
      SafeDivide(metrics.flops(), metrics.bytes_accessed()));
  record->set_bound_by((metrics.bytes_accessed() != 0)
                           ? ((record->operational_intensity() >=
                               ridge_point_operational_intensity)
                                  ? "Compute"
                                  : "Memory")
                           : ((metrics.flops() != 0) ? "Compute" : "Unknown"));
}

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_CONVERT_OP_METRICS_TO_RECORD_H_
