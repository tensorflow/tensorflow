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

#include "xla/backends/profiler/gpu/cupti_pm_sampler_utils.h"

#include <string>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace profiler {
namespace {

absl::flat_hash_map<absl::string_view, absl::string_view>
ProfileMetricNameMap() {
  return {
      // PM Sampling metrics, may require multipass if metrics are not under
      // TriageCompute group.
      {"gpc__cycles_elapsed.avg.per_second", "GPC Clock Frequency (Hz)"},
      {"sys__cycles_elapsed.avg.per_second", "SYS Clock Frequency (Hz)"},
      {"gr__cycles_active.sum.pct_of_peak_sustained_elapsed", "GR Active (%)"},
      {"TPC.TriageCompute.sm__cycles_active.avg.pct_of_peak_sustained_elapsed",
       "SMs Active (%)"},
      {"sm__inst_executed_realtime.avg.pct_of_peak_sustained_elapsed",
       "SM Instructions - SM Issue (%)"},
      {"TPC.TriageCompute.sm__pipe_tensor_cycles_active_realtime.avg.pct_of_"
       "peak_sustained_elapsed",
       "SM Instructions - Tensor Active (%)"},
      {"sm__warps_active.avg.pct_of_peak_sustained_active",
       "Achieved Occupancy in Active Cycles (%)"},
      {"nvlrx__bytes.avg.pct_of_peak_sustained_elapsed",
       "NVLink RX Bandwidth (%)"},
      {"nvltx__bytes.avg.pct_of_peak_sustained_elapsed",
       "NVLink TX Bandwidth (%)"},
      {"dramc__read_throughput.avg.pct_of_peak_sustained_elapsed",
       "DRAM Read Bandwidth (%)"},
      {"dramc__write_throughput.avg.pct_of_peak_sustained_elapsed",
       "DRAM Write Bandwidth (%)"},
      {"pcie__read_bytes.avg.pct_of_peak_sustained_elapsed",
       "PCIe RX Throughput (%)"},
      {"pcie__write_bytes.avg.pct_of_peak_sustained_elapsed",
       "PCIe TX Throughput (%)"},
      {"sm__inst_executed_pipe_tensor.sum",
       "Total Tensor Instructions (Count)"},
      // Range Profiling metrics.
      {"sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
       "FP16/BF16 Tensor Pipe Utilization (%)"},
      {"sm__pipe_tensor_op_imma.avg.pct_of_peak_sustained_active",
       "INT8 Tensor Pipe Utilization (%)"},
      {"sm__pipe_tensor_op_dmma.avg.pct_of_peak_sustained_active",
       "TF32/FP64 Tensor Pipe Utilization (%)"},
      {"sm__sass_inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active",
       "Tensor Core Speed of Light (SOL) (%)"},
  };
}

const absl::flat_hash_map<absl::string_view, absl::string_view>&
MetricNameMap() {
  static const absl::NoDestructor<
      absl::flat_hash_map<absl::string_view, absl::string_view>>
      kMetricNameMap(ProfileMetricNameMap());
  return *kMetricNameMap;
}

}  // namespace

std::string GetGpuProfileMetricName(absl::string_view metric_name) {
  const auto& metric_name_map = MetricNameMap();
  if (auto it = metric_name_map.find(metric_name);
      it != metric_name_map.end()) {
    return std::string(it->second);
  }
  return std::string(metric_name);
}

}  // namespace profiler
}  // namespace xla
