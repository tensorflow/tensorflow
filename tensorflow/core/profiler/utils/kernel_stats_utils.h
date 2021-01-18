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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"

namespace tensorflow {
namespace profiler {

// Populates kernel launch information from a kKernelDetails XStat.
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
  bool operator()(const KernelReport& lhs, const KernelReport& rhs) const;
};

// Equal to comparator for Kernel Reports.
struct KernelReportEqualToComparator {
  bool operator()(const KernelReport& lhs, const KernelReport& rhs) const;
};

// Sorts kernel reorts by total duration descendingly.
// Keeps only the top kernel reports with long kernel duration in the given
// KernelStatsDb. Kernel reports with shorter kernel duration are dropped.
void SortAndKeepTopKDurationKernelReportsInDb(KernelStatsDb* kernel_stats_db);

struct KernelReportValue {
  uint64 total_duration_ns = 0;
  uint64 min_duration_ns = 0;
  uint64 max_duration_ns = 0;
  uint64 occurrences = 0;
};

struct KernelKeyWrap {
  const KernelReport* key;
  template <typename H>
  friend H AbslHashValue(H h, KernelKeyWrap wrap) {
    // Kernel reports are grouped by these fields, hence they are used as
    // hashing criteria.
    // clang-format off
    return H::combine(
        std::move(h),
        wrap.key->is_kernel_using_tensor_core(),
        wrap.key->is_op_tensor_core_eligible(),
        wrap.key->block_dim(0),
        wrap.key->block_dim(1),
        wrap.key->block_dim(2),
        wrap.key->grid_dim(0),
        wrap.key->grid_dim(1),
        wrap.key->grid_dim(2),
        wrap.key->registers_per_thread(),
        wrap.key->static_shmem_bytes(),
        wrap.key->dynamic_shmem_bytes(),
        wrap.key->name(),
        wrap.key->op_name());
    // clang-format on
  }
};

struct KernelHash {
  size_t operator()(const KernelReport& key) const {
    return absl::Hash<KernelKeyWrap>()(KernelKeyWrap{&key});
  }
};

using KernelReportMap =
    absl::flat_hash_map<KernelReport, KernelReportValue, KernelHash,
                        KernelReportEqualToComparator>;

// Copies the top kernel reports with long kernel duration into the given
// KernelStatsDb.
void CopyTopKDurationKernelReportsToDb(const KernelReportMap& reports,
                                       KernelStatsDb* dst);

// Inserts or aggregates KernelReports into the given KernelReportMap.
void InsertOrUpdateKernelReport(const KernelReport& kernel,
                                const KernelReportValue& value,
                                KernelReportMap* dst);

// Aggregates values from one KernelReportMap into another.
void MergeKernelReports(const KernelReportMap& reports, KernelReportMap* dst);

// Kernel stats aggregated at TF operation level.
struct OpLevelKernelStats {
  // Whether op is eligible to use TensorCore.
  bool is_op_tensor_core_eligible = false;
  // The accumulated duration of all the kernels launched in this op.
  uint64 total_duration_ns = 0;
  // The accumulated duration of all the kernels using TensorCore in this op.
  // If this value is not 0, at least one of the kernels launched by this op
  // is using TensorCore.
  uint64 tensor_core_duration_ns = 0;
};

using KernelStatsByOpName =
    absl::flat_hash_map<absl::string_view, OpLevelKernelStats>;

// Groups KernelReport in <kernel_stats_db> by tensorflow operation name.
KernelStatsByOpName GroupKernelReportsByOpName(
    const KernelStatsDb& kernel_stats_db);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_KERNEL_STATS_UTILS_H_
