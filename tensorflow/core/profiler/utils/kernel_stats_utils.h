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

#ifndef TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_

#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Populates kernel launch information from a KernelDetails XStat.
void ParseKernelLaunchParams(absl::string_view xstat_kernel_details,
                             KernelReport* kernel);

// Returns true if kernel uses TensorCores.
bool IsKernelUsingTensorCore(absl::string_view kernel_name);

// Returns true if operation is eligible to use TensorCores.
bool IsOpTensorCoreEligible(absl::string_view tf_op_name);

// Returns true if Einsum equation is eligible to use TensorCores.
bool IsEinsumTensorCoreEligible(absl::string_view equation);

// Less than comparator for Kernel Reports.
struct KernelReportLessThanComparator {
  bool operator()(const KernelReport& lhs, const KernelReport& rhs);
};

// Equal to comparator for Kernel Reports.
struct KernelReportEqualToComparator {
  bool operator()(const KernelReport& lhs, const KernelReport& rhs);
};

// Sorts kernel reorts by total duration descendingly.
void SortKernelsByTotalDurationDesc(KernelStatsDb* kernel_stats_db);

// Groups and aggregate common reports into destination KernelStatsDb.
void GroupKernelReports(std::vector<KernelReport>* reports, KernelStatsDb* dst);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
