/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/buffer_comparator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "Eigen/Core"
#include "xla/index_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/gpu/buffer_comparator_kernel.h"
#include "xla/stream_executor/gpu/gpu_kernel_registry.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct ComparisonParams {
  double relative_tol = 0.1;
  bool verbose = true;
  bool run_host_compare = true;
  const Shape* shape = nullptr;
  se::Stream* stream = nullptr;
  se::DeviceAddressBase current{};
  se::DeviceAddressBase expected{};
  std::string* error_report = nullptr;
};

// Compares two buffers on the GPU.
//
// Returns `true` if two buffers are equal, `false` otherwise.
template <typename ElementT>
static absl::StatusOr<bool> DeviceCompare(const ComparisonParams& params) {
  se::StreamExecutor* executor = params.stream->parent();

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<se::MemoryAllocation> allocation,
                   executor->HostMemoryAllocate(sizeof(uint64_t)));
  se::DeviceAddressBase out_addr = allocation->address();

  ABSL_RETURN_IF_ERROR(params.stream->MemZero(&out_addr, sizeof(uint64_t)));
  if (params.current.size() != params.expected.size()) {
    return Internal("Mismatched buffer size: %d bytes vs. %d bytes",
                    params.current.size(), params.expected.size());
  }

  se::DeviceAddress<ElementT> current_typed(params.current);
  se::DeviceAddress<ElementT> expected_typed(params.expected);
  uint64_t buffer_size = current_typed.ElementCount();

  ABSL_ASSIGN_OR_RETURN(
      auto comparison_kernel,
      stream_executor::gpu::GpuKernelRegistry::GetGlobalRegistry()
          .LoadKernel<stream_executor::gpu::BufferComparatorKernel<ElementT>>(
              executor));

  const se::DeviceDescription& gpu_device_info =
      executor->GetDeviceDescription();

  LaunchDimensions dim =
      CalculateLaunchDimensions(*params.shape, gpu_device_info);
  // Limit # of blocks to some meaningful number which is large enough to
  // occupy all GPU cores if necessary but not too large to reduce # of idle
  // blocks
  dim = LaunchDimensions(
      se::BlockDim(std::min(dim.num_blocks(),
                            BufferComparator::kMaxNumThreadBlocksForKernel),
                   1, 1),
      dim.thread_counts_per_block());

  se::DeviceAddress<uint64_t> as_uint64(out_addr);
  ABSL_RETURN_IF_ERROR(comparison_kernel.Launch(
      dim.thread_counts_per_block(), dim.block_counts(), params.stream,
      current_typed, expected_typed, static_cast<float>(params.relative_tol),
      buffer_size, as_uint64));

  uint64_t result = -1;
  CHECK_EQ(out_addr.size(), sizeof(result));
  ABSL_RETURN_IF_ERROR(params.stream->Memcpy(&result, out_addr, sizeof(result)));
  ABSL_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
  return result == 0;
}

struct MismatchSample {
  int64_t linear_index = -1;
  double current_value = 0.0;
  double expected_value = 0.0;
  double abs_diff = 0.0;
  double rel_diff = 0.0;
};

struct BufferMismatchStats {
  int64_t total_elements = 0;
  int64_t mismatch_count = 0;
  int64_t nan_count = 0;
  int64_t inf_count = 0;
  double relative_tol = 0.0;
  double max_rel_diff = 0.0;
  double max_abs_diff = 0.0;
  MismatchSample max_rel_diff_sample;
  MismatchSample max_abs_diff_sample;
  std::vector<MismatchSample> sample_mismatches;
};

static std::string FormatSample(const MismatchSample& sample,
                                const Shape* shape) {
  if (shape != nullptr && !shape->dimensions().empty()) {
    DimensionVector multi_idx = IndexUtil::LinearIndexToMultidimensionalIndex(
        *shape, sample.linear_index);
    return absl::StrFormat(
        "[%s] (linear index %v): actual %g, expected %g (abs diff: %g, rel "
        "diff: %g)",
        absl::StrJoin(multi_idx, ", "), sample.linear_index,
        sample.current_value, sample.expected_value, sample.abs_diff,
        sample.rel_diff);
  }
  return absl::StrFormat(
      "linear index %v: actual %g, expected %g (abs diff: %g, rel diff: %g)",
      sample.linear_index, sample.current_value, sample.expected_value,
      sample.abs_diff, sample.rel_diff);
}

static std::string BuildBufferComparisonErrorReport(
    const BufferMismatchStats& stats, const Shape* shape) {
  std::string report = absl::StrFormat(
      "  Mismatch count: %v / %v (%0.4f%%)\n"
      "  Relative tolerance: %g\n",
      stats.mismatch_count, stats.total_elements,
      (stats.total_elements > 0
           ? (100.0 * stats.mismatch_count / stats.total_elements)
           : 0.0),
      stats.relative_tol);

  if (stats.max_rel_diff_sample.linear_index != -1) {
    absl::StrAppendFormat(&report, "  Max relative difference: %g at %s\n",
                          stats.max_rel_diff,
                          FormatSample(stats.max_rel_diff_sample, shape));
  } else {
    absl::StrAppend(
        &report,
        "  Max relative difference: N/A (all mismatches are NaN/Inf)\n");
  }

  if (stats.max_abs_diff_sample.linear_index != -1) {
    absl::StrAppendFormat(&report, "  Max absolute difference: %g at %s\n",
                          stats.max_abs_diff,
                          FormatSample(stats.max_abs_diff_sample, shape));
  } else {
    absl::StrAppend(
        &report,
        "  Max absolute difference: N/A (all mismatches are NaN/Inf)\n");
  }

  absl::StrAppendFormat(&report, "  NaN count: %v, Inf count: %v\n",
                        stats.nan_count, stats.inf_count);

  if (!stats.sample_mismatches.empty()) {
    absl::StrAppendFormat(&report, "  First %v mismatch sample(s):\n",
                          stats.sample_mismatches.size());
    for (const auto& sample : stats.sample_mismatches) {
      absl::StrAppendFormat(&report, "    %s\n", FormatSample(sample, shape));
    }
  }
  return report;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
//
// Returns true if no differences were seen, false otherwise.
template <typename ElementType, typename ComparisonType>
static absl::StatusOr<bool> HostCompare(const ComparisonParams& params) {
  int64_t n = params.current.size() / sizeof(ElementType);
  std::vector<ElementType> host_current(n), host_expected(n);
  ABSL_RETURN_IF_ERROR(params.stream->Memcpy(host_current.data(), params.current,
                                        params.current.size()));
  ABSL_RETURN_IF_ERROR(params.stream->Memcpy(host_expected.data(), params.expected,
                                        params.expected.size()));
  ABSL_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());

  const auto canonicalize = [](ComparisonType a) -> ComparisonType {
    if (std::is_same<ElementType, Eigen::half>::value && a) {
      constexpr ComparisonType kMaxFp16Value = 65505;
      if (std::isnan(a)) {
        return a;
      }
      return std::max(-kMaxFp16Value, std::min(a, kMaxFp16Value));
    }
    return a;
  };

  BufferMismatchStats stats;
  stats.total_elements = n;
  stats.relative_tol = params.relative_tol;
  constexpr int kMaxSamples = 5;

  for (int64_t i = 0; i < n; ++i) {
    auto current_value = static_cast<ComparisonType>(host_current[i]);
    auto expected_value = static_cast<ComparisonType>(host_expected[i]);
    ComparisonType current_value_canonical = canonicalize(current_value);
    ComparisonType expected_value_canonical = canonicalize(expected_value);
    if (std::isnan(current_value_canonical) &&
        std::isnan(expected_value_canonical)) {
      continue;
    }
    if (std::isinf(current_value_canonical) &&
        std::isinf(expected_value_canonical) &&
        current_value_canonical == expected_value_canonical) {
      continue;
    }

    const bool is_current_nan = std::isnan(current_value_canonical);
    const bool is_expected_nan = std::isnan(expected_value_canonical);
    const bool is_current_inf = std::isinf(current_value_canonical);
    const bool is_expected_inf = std::isinf(expected_value_canonical);

    bool is_mismatch = false;
    double abs_diff = 0.0;
    double rel_diff = 0.0;

    if (is_current_nan || is_expected_nan) {
      is_mismatch = true;
      ++stats.nan_count;
      abs_diff = std::numeric_limits<double>::quiet_NaN();
      rel_diff = std::numeric_limits<double>::quiet_NaN();
    } else if (is_current_inf || is_expected_inf) {
      is_mismatch = true;
      ++stats.inf_count;
      abs_diff = std::numeric_limits<double>::infinity();
      rel_diff = std::numeric_limits<double>::infinity();
    } else {
      abs_diff = std::abs(static_cast<double>(current_value_canonical) -
                          static_cast<double>(expected_value_canonical));
      rel_diff =
          abs_diff /
          (std::max(std::abs(static_cast<double>(current_value_canonical)),
                    std::abs(static_cast<double>(expected_value_canonical))) +
           1.0);
      if (rel_diff >= params.relative_tol) {
        is_mismatch = true;
      }
    }

    if (is_mismatch) {
      if (!params.verbose && params.error_report == nullptr) {
        return false;  // Fast path: return immediately if verbose is off and no
                       // report needed.
      }
      ++stats.mismatch_count;
      if (std::isfinite(rel_diff) && rel_diff > stats.max_rel_diff) {
        stats.max_rel_diff = rel_diff;
        stats.max_rel_diff_sample = {i, static_cast<double>(current_value),
                                     static_cast<double>(expected_value),
                                     abs_diff, rel_diff};
      }
      if (std::isfinite(abs_diff) && abs_diff > stats.max_abs_diff) {
        stats.max_abs_diff = abs_diff;
        stats.max_abs_diff_sample = {i, static_cast<double>(current_value),
                                     static_cast<double>(expected_value),
                                     abs_diff, rel_diff};
      }
      if (stats.sample_mismatches.size() < kMaxSamples) {
        stats.sample_mismatches.push_back(
            {i, static_cast<double>(current_value),
             static_cast<double>(expected_value), abs_diff, rel_diff});
      }
      if (params.verbose && stats.mismatch_count <= 10) {
        LOG(ERROR) << "Difference at " << i << ": " << current_value
                   << ", expected " << expected_value;
      }
      if (params.error_report == nullptr && stats.mismatch_count >= 10) {
        break;
      }
    }
  }

  if (params.error_report != nullptr && stats.mismatch_count > 0) {
    *params.error_report =
        BuildBufferComparisonErrorReport(stats, params.shape);
  }

  return stats.mismatch_count == 0;
}

template <typename ElementT, typename ComparisonT>
static absl::StatusOr<bool> CompareEqualParameterized(
    const ComparisonParams& params) {
  XLA_SCOPED_LOGGING_TIMER("BufferComparator::CompareEqual");
  ABSL_ASSIGN_OR_RETURN(bool result, DeviceCompare<ElementT>(params));
  if (result) {
    return true;
  }
  if (!params.run_host_compare && params.error_report == nullptr) {
    return false;
  }

  ABSL_ASSIGN_OR_RETURN(bool host_return,
                   (HostCompare<ElementT, ComparisonT>(params)));
  CHECK_EQ(host_return, result)
      << "Host comparison succeeded even though GPU comparison failed.";
  return false;
}

absl::StatusOr<bool> BufferComparator::CompareEqual(
    se::Stream* stream, const se::DeviceAddressBase& current,
    const se::DeviceAddressBase& expected, std::string* error_report) const {
  if (error_report != nullptr) {
    error_report->clear();
  }
  ComparisonParams params{relative_tol_, verbose_,    run_host_compare_,
                          &shape_,       stream,      current,
                          expected,      error_report};

  auto do_compare = [&](auto cst_type) {
    using ElementT = primitive_util::NativeTypeOf<cst_type>;
    using ComparisonT =
        std::conditional_t<std::is_same_v<ElementT, double>, double, float>;
    return CompareEqualParameterized<ElementT, ComparisonT>(params);
  };

  PrimitiveType element_type = shape_.element_type();
  // PRED is handled on buffer level the same way as S8.
  if (element_type == PRED) {
    element_type = S8;
  }
  if (primitive_util::IsFloatingPointType(element_type)) {
    return xla::primitive_util::FloatingPointTypeSwitch(do_compare,
                                                        element_type);
  }

  if (primitive_util::IsIntegralType(element_type)) {
    return xla::primitive_util::IntegralTypeSwitch(do_compare, element_type);
  }

  return absl::UnimplementedError(
      absl::StrCat("Unimplemented element type for host function: ",
                   primitive_util::LowercasePrimitiveTypeName(element_type)));
}

BufferComparator::BufferComparator(const Shape& shape, double tolerance,
                                   bool verbose, bool run_host_compare)
    : shape_(shape),
      relative_tol_(tolerance),
      verbose_(verbose),
      run_host_compare_(run_host_compare) {
  // Normalize complex shapes: since we treat the passed array as a contiguous
  // storage it does not matter which dimension are we doubling.
  auto double_dim_size = [&]() {
    // A 0D tensor is equal to a 1D tensor of size 1 in a buffer.
    if (shape_.dimensions().empty()) {
      shape_.add_dimensions(1);
    }
    int64_t prev_zero_dim_size = shape_.dimensions(0);
    shape_.set_dimensions(0, prev_zero_dim_size * 2);
  };

  if (shape_.element_type() == PrimitiveType::C64) {
    // C64 is just two F32s next to each other.
    shape_.set_element_type(PrimitiveType::F32);
    double_dim_size();
  } else if (shape_.element_type() == PrimitiveType::C128) {
    // C128 is just two F64s next to each other.
    shape_.set_element_type(PrimitiveType::F64);
    double_dim_size();
  }
}

}  // namespace gpu
}  // namespace xla
