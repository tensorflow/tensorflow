/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"

#include <algorithm>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

namespace {

// The maximum number of Kernels displayed on Kernel Stats page.
const int kMaxNumOfKernels = 1000;

}  // namespace

void ParseKernelLaunchParams(absl::string_view xstat_kernel_details,
                             KernelReport* kernel) {
  const std::vector<absl::string_view> params =
      absl::StrSplit(xstat_kernel_details, absl::ByAnyChar(" \n"));

  constexpr uint32 kNumDimensions = 3;
  for (uint32 dim = 0; dim < kNumDimensions; ++dim) {
    kernel->add_block_dim(1);
    kernel->add_grid_dim(1);
  }

  // Process tokens.
  for (const auto& param : params) {
    const std::vector<absl::string_view> key_value = absl::StrSplit(param, ':');
    if (key_value.size() != 2) {
      // Unrecognized token.
      continue;
    }
    absl::string_view key = key_value[0];
    absl::string_view value_str = key_value[1];
    uint32 value = 0;
    double pct = 0.0;
    // Cases that consume a pair of tokens "key:value".
    if (key == "regs" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_registers_per_thread(value);
    } else if (key == "static_shared" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_static_shmem_bytes(value);
    } else if (key == "dynamic_shared" && absl::SimpleAtoi(value_str, &value)) {
      kernel->set_dynamic_shmem_bytes(value);
    } else if (key == "block") {
      const std::vector<absl::string_view>& block =
          absl::StrSplit(value_str, ',');
      uint32 tmp[3];
      if (block.size() == 3 && absl::SimpleAtoi(block[0], &tmp[0]) &&
          absl::SimpleAtoi(block[1], &tmp[1]) &&
          absl::SimpleAtoi(block[2], &tmp[2])) {
        std::copy_n(tmp, 3, kernel->mutable_block_dim()->begin());
      }
    } else if (key == "grid") {
      const std::vector<absl::string_view>& grid =
          absl::StrSplit(value_str, ',');
      uint32 tmp[3];
      if (grid.size() == 3 && absl::SimpleAtoi(grid[0], &tmp[0]) &&
          absl::SimpleAtoi(grid[1], &tmp[1]) &&
          absl::SimpleAtoi(grid[2], &tmp[2])) {
        std::copy_n(tmp, 3, kernel->mutable_grid_dim()->begin());
      }
    } else if (key == "occ_pct" && absl::SimpleAtod(value_str, &pct)) {
      kernel->set_occupancy_pct(pct);
    }
  }
}

bool IsKernelUsingTensorCore(absl::string_view kernel_name) {
  // Some examples: volta_h884gemm, volta_fp16_s884gemm,
  // turing_fp16_s1688cudnn_fp16
  bool possible_tensor_kernel = absl::StrContains(kernel_name, "884") ||
                                absl::StrContains(kernel_name, "1688") ||
                                absl::StrContains(kernel_name, "hmma") ||
                                absl::StrContains(kernel_name, "xmma");
  if (possible_tensor_kernel) {
    VLOG(3) << "Possible tensor kernel: " << kernel_name;
  }

  return (absl::StartsWith(kernel_name, "volta_i884") ||
          absl::StartsWith(kernel_name, "volta_h884") ||
          absl::StartsWith(kernel_name, "volta_s884") ||
          absl::StartsWith(kernel_name, "volta_fp16_i884") ||
          absl::StartsWith(kernel_name, "volta_fp16_h884") ||
          absl::StartsWith(kernel_name, "volta_fp16_s884") ||
          absl::StartsWith(kernel_name, "turing_i1688") ||
          absl::StartsWith(kernel_name, "turing_h1688") ||
          absl::StartsWith(kernel_name, "turing_s1688") ||
          absl::StartsWith(kernel_name, "turing_fp16_i1688") ||
          absl::StartsWith(kernel_name, "turing_fp16_h1688") ||
          absl::StartsWith(kernel_name, "turing_fp16_s1688") ||
          absl::StrContains(kernel_name, "hmma") ||
          absl::StrContains(kernel_name, "xmma"));
}

// This list is not exhaustive.
bool IsOpTensorCoreEligible(absl::string_view tf_op_name) {
  // Disable formatting to keep inline comments vertically aligned.
  // clang-format off
  return false
      // Using EndsWith to match Fused operations.
      || absl::EndsWith(tf_op_name, "Conv2D")
      || absl::EndsWith(tf_op_name, "Conv2DBackpropFilter")
      || absl::EndsWith(tf_op_name, "Conv2DBackpropInput")
      || absl::EndsWith(tf_op_name, "Conv3D")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNative")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNativeBackpropFilter")
      || absl::EndsWith(tf_op_name, "DepthwiseConv2dNativeBackpropInput")
      // Using Contains to match V2/V3 suffixes.
      || absl::StrContains(tf_op_name, "BatchMatMul")
      // MatMul requires exact matching.
      || absl::EndsWith(tf_op_name, "/MatMul")
      || absl::EndsWith(tf_op_name, "FusedMatMul")
      // cuDNN operations.
      || absl::EndsWith(tf_op_name, "/CudnnRNN")
      || absl::StrContains(tf_op_name, "CudnnRNNV")
      || absl::StrContains(tf_op_name, "CudnnRNNForward")
      || absl::StrContains(tf_op_name, "CudnnRNNBackprop")
      // Special cases.
      || absl::EndsWith(tf_op_name, "XlaDot")
      || absl::EndsWith(tf_op_name, "XlaDotV2");
  // clang-format on
}

bool IsEinsumTensorCoreEligible(absl::string_view equation) {
  if (equation.empty()) {
    return false;
  }
  const std::vector<absl::string_view> input_output =
      absl::StrSplit(equation, "->");
  if (input_output.size() != 2) {
    return false;
  }
  const std::vector<absl::string_view> lhs_rhs =
      absl::StrSplit(input_output[0], ',');
  return lhs_rhs.size() == 2;
}

bool KernelReportLessThanComparator::operator()(const KernelReport& lhs,
                                                const KernelReport& rhs) const {
  // Disable formatting to keep vertical alignment for better readability,
  // and make it easier to reorder columns.
  // clang-format off
  auto lhs_tuple = std::make_tuple(
      lhs.name(),
      lhs.grid_dim(0),
      lhs.grid_dim(1),
      lhs.grid_dim(2),
      lhs.block_dim(0),
      lhs.block_dim(1),
      lhs.block_dim(2),
      lhs.registers_per_thread(),
      lhs.static_shmem_bytes(),
      lhs.dynamic_shmem_bytes(),
      lhs.is_kernel_using_tensor_core(),
      lhs.is_op_tensor_core_eligible(),
      lhs.op_name());

  auto rhs_tuple = std::make_tuple(
      rhs.name(),
      rhs.grid_dim(0),
      rhs.grid_dim(1),
      rhs.grid_dim(2),
      rhs.block_dim(0),
      rhs.block_dim(1),
      rhs.block_dim(2),
      rhs.registers_per_thread(),
      rhs.static_shmem_bytes(),
      rhs.dynamic_shmem_bytes(),
      rhs.is_kernel_using_tensor_core(),
      rhs.is_op_tensor_core_eligible(),
      rhs.op_name());
  // clang-format on
  return lhs_tuple < rhs_tuple;
}

bool KernelReportEqualToComparator::operator()(const KernelReport& lhs,
                                               const KernelReport& rhs) const {
  // Disable formatting to keep vertical alignment for better readability,
  // and make it easier to reorder columns.
  // clang-format off
  // Put the most expensive string comparisons last.
  return (
      lhs.is_kernel_using_tensor_core() == rhs.is_kernel_using_tensor_core() &&
      lhs.is_op_tensor_core_eligible() == rhs.is_op_tensor_core_eligible() &&
      lhs.block_dim(0) == rhs.block_dim(0) &&
      lhs.block_dim(1) == rhs.block_dim(1) &&
      lhs.block_dim(2) == rhs.block_dim(2) &&
      lhs.grid_dim(0) == rhs.grid_dim(0) &&
      lhs.grid_dim(1) == rhs.grid_dim(1) &&
      lhs.grid_dim(2) == rhs.grid_dim(2) &&
      lhs.registers_per_thread() == rhs.registers_per_thread() &&
      lhs.static_shmem_bytes() == rhs.static_shmem_bytes() &&
      lhs.dynamic_shmem_bytes() == rhs.dynamic_shmem_bytes() &&
      lhs.name() == rhs.name() &&
      lhs.op_name() == rhs.op_name());
  // clang-format on
}

void SortAndKeepTopKDurationKernelReportsInDb(KernelStatsDb* kernel_stats_db) {
  auto comp = [](const KernelReport& lhs, const KernelReport& rhs) {
    return lhs.total_duration_ns() > rhs.total_duration_ns() ||
           (lhs.total_duration_ns() == rhs.total_duration_ns() &&
            KernelReportLessThanComparator()(lhs, rhs));
  };

  // Sort and keep at most <kMaxNumOfKernels> kernel reports.
  if (kernel_stats_db->reports_size() > kMaxNumOfKernels) {
    std::partial_sort(
        kernel_stats_db->mutable_reports()->begin(),
        kernel_stats_db->mutable_reports()->begin() + kMaxNumOfKernels,
        kernel_stats_db->mutable_reports()->end(), comp);
    kernel_stats_db->mutable_reports()->erase(
        kernel_stats_db->mutable_reports()->begin() + kMaxNumOfKernels,
        kernel_stats_db->mutable_reports()->end());
  } else {
    std::sort(kernel_stats_db->mutable_reports()->begin(),
              kernel_stats_db->mutable_reports()->end(), comp);
  }
}

void CopyTopKDurationKernelReportsToDb(const KernelReportMap& reports,
                                       KernelStatsDb* dst) {
  std::vector<std::pair<const KernelReport*, const KernelReportValue*>>
      kernels_to_sort;
  kernels_to_sort.reserve(reports.size());
  for (const auto& report_value : reports) {
    kernels_to_sort.push_back(
        std::make_pair(&report_value.first, &report_value.second));
  }

  auto comp =
      [](const std::pair<const KernelReport*, const KernelReportValue*>& lhs,
         const std::pair<const KernelReport*, const KernelReportValue*>& rhs) {
        return lhs.second->total_duration_ns > rhs.second->total_duration_ns ||
               (lhs.second->total_duration_ns ==
                    rhs.second->total_duration_ns &&
                KernelReportLessThanComparator()(*lhs.first, *rhs.first));
      };

  // Sort and copy at most <kMaxNumOfKernels> kernels to <dst>.
  if (kernels_to_sort.size() > kMaxNumOfKernels) {
    absl::c_partial_sort(kernels_to_sort,
                         kernels_to_sort.begin() + kMaxNumOfKernels, comp);
  } else {
    absl::c_sort(kernels_to_sort, comp);
  }

  int copy_size =
      std::min(kMaxNumOfKernels, static_cast<int>(kernels_to_sort.size()));
  for (int i = 0; i < copy_size; i++) {
    KernelReport* report = dst->add_reports();
    *report = *kernels_to_sort[i].first;
    const KernelReportValue& kernel_value = *kernels_to_sort[i].second;
    // Set value using KernelReportValue.
    report->set_occurrences(kernel_value.occurrences);
    report->set_min_duration_ns(kernel_value.min_duration_ns);
    report->set_max_duration_ns(kernel_value.max_duration_ns);
    report->set_total_duration_ns(kernel_value.total_duration_ns);
  }
}

void InsertOrUpdateKernelReport(const KernelReport& kernel,
                                const KernelReportValue& value,
                                KernelReportMap* dst) {
  KernelReportValue& element = (*dst)[kernel];
  if (element.occurrences == 0) {
    element = value;
  } else {
    element.total_duration_ns += value.total_duration_ns;
    element.min_duration_ns =
        std::min(element.min_duration_ns, value.min_duration_ns);
    element.max_duration_ns =
        std::max(element.max_duration_ns, value.max_duration_ns);
    element.occurrences += 1;
  }
}

void MergeKernelReports(const KernelReportMap& reports, KernelReportMap* dst) {
  for (auto& kernel_value : reports) {
    InsertOrUpdateKernelReport(kernel_value.first, kernel_value.second, dst);
  }
}

KernelStatsByOpName GroupKernelReportsByOpName(
    const KernelStatsDb& kernel_stats_db) {
  KernelStatsByOpName op_level_kernel_stats;
  for (const KernelReport& kernel_report : kernel_stats_db.reports()) {
    auto ret = op_level_kernel_stats.emplace(kernel_report.op_name(),
                                             OpLevelKernelStats());
    if (ret.second) {
      // Inserted. Add a new op in <op_level_kernel_stats>.
      OpLevelKernelStats& stats = ret.first->second;
      stats.is_op_tensor_core_eligible =
          kernel_report.is_op_tensor_core_eligible();
      stats.total_duration_ns += kernel_report.total_duration_ns();
      if (kernel_report.is_kernel_using_tensor_core()) {
        stats.tensor_core_duration_ns += kernel_report.total_duration_ns();
      }
    } else {
      // Not inserted. Aggregate kernel stats to op level.
      OpLevelKernelStats& stats = ret.first->second;
      // Verifies operations with the same name have the same TensorCore
      // eligibility.
      DCHECK_EQ(stats.is_op_tensor_core_eligible,
                kernel_report.is_op_tensor_core_eligible());
      stats.total_duration_ns += kernel_report.total_duration_ns();
      if (kernel_report.is_kernel_using_tensor_core()) {
        stats.tensor_core_duration_ns += kernel_report.total_duration_ns();
      }
    }
  }
  return op_level_kernel_stats;
}

}  // namespace profiler
}  // namespace tensorflow
