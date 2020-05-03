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

#include <tuple>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

void ParseKernelLaunchParams(absl::string_view xstat_kernel_details,
                             KernelReport* kernel) {
  const std::vector<absl::string_view> params =
      absl::StrSplit(xstat_kernel_details, absl::ByAnyChar(":\n"));

  constexpr uint32_t kNumDimensions = 3;
  for (uint32_t dim = 0; dim < kNumDimensions; ++dim) {
    kernel->add_block_dim(1);
    kernel->add_grid_dim(1);
  }

  // Process value pairs.
  for (uint32_t ii = 0; ii < params.size(); ii += 2) {
    uint32_t value = 0;
    if (params[ii] == "registers_per_thread" &&
        absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->set_registers_per_thread(value);
    } else if (params[ii] == "static_shared_memory_usage" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->set_static_shmem_bytes(value);
    } else if (params[ii] == "dynamic_shared_memory_usage" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->set_dynamic_shmem_bytes(value);
    } else if (params[ii] == "block_x" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_block_dim()->Set(0, value);
    } else if (params[ii] == "block_y" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_block_dim()->Set(1, value);
    } else if (params[ii] == "block_z" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_block_dim()->Set(2, value);
    } else if (params[ii] == "grid_x" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_grid_dim()->Set(0, value);
    } else if (params[ii] == "grid_y" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_grid_dim()->Set(1, value);
    } else if (params[ii] == "grid_z" &&
               absl::SimpleAtoi(params[ii + 1], &value)) {
      kernel->mutable_grid_dim()->Set(2, value);
    }
  }
}

bool IsKernelUsingTensorCore(absl::string_view kernel_name) {
  // Some examples: volta_h884gemm, volta_fp16_s884gemm,
  // turing_fp16_s1688cudnn_fp16
  bool possible_tensor_kernel = absl::StrContains(kernel_name, "884") ||
                                absl::StrContains(kernel_name, "1688");
  if (possible_tensor_kernel) {
    VLOG(1) << "Possible tensor kernel: " << kernel_name << "\n";
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
          absl::StartsWith(kernel_name, "turing_fp16_s1688"));
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
      || absl::EndsWith(tf_op_name, "XlaDot");
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
                                                const KernelReport& rhs) {
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
                                               const KernelReport& rhs) {
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

void SortKernelsByTotalDurationDesc(KernelStatsDb* kernel_stats_db) {
  // Sort kernel reports by total duration descendingly.
  std::sort(kernel_stats_db->mutable_reports()->begin(),
            kernel_stats_db->mutable_reports()->end(),
            [](const KernelReport& lhs, const KernelReport& rhs) {
              return lhs.total_duration_ns() > rhs.total_duration_ns() ||
                     (lhs.total_duration_ns() == rhs.total_duration_ns() &&
                      KernelReportLessThanComparator()(lhs, rhs));
            });
}

void GroupKernelReports(std::vector<KernelReport>* reports,
                        KernelStatsDb* dst) {
  // Sort reports by grouping criteria.
  std::sort(reports->begin(), reports->end(), KernelReportLessThanComparator());

  // Group reports together.
  KernelReport* prev = nullptr;
  for (const KernelReport& report : *reports) {
    DCHECK_EQ(3, report.grid_dim_size());
    DCHECK_EQ(3, report.block_dim_size());
    if (prev != nullptr && KernelReportEqualToComparator()(*prev, report)) {
      // Previous element is identical to the one that we are adding, so
      // aggregate them.
      prev->set_occurrences(prev->occurrences() + 1);
      prev->set_max_duration_ns(
          std::max(prev->max_duration_ns(), report.max_duration_ns()));
      prev->set_min_duration_ns(
          std::min(prev->min_duration_ns(), report.min_duration_ns()));
      prev->set_total_duration_ns(prev->total_duration_ns() +
                                  report.total_duration_ns());
    } else {
      // Current element does not exist yet.
      prev = dst->add_reports();
      *prev = report;
      prev->set_occurrences(1);
    }
  }
}

}  // namespace profiler
}  // namespace tensorflow
