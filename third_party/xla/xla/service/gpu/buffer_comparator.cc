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

#include "xla/service/gpu/buffer_comparator.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

#include "Eigen/Core"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/typed_kernel_factory.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

template <typename ElementT>
using ComparisonKernelT =
    se::TypedKernel<se::DeviceMemory<ElementT>, se::DeviceMemory<ElementT>,
                    float, uint64_t, se::DeviceMemory<uint64_t>>;

struct ComparisonParams {
  double relative_tol = 0.1;
  bool verbose = true;
  const Shape* shape = nullptr;
  se::Stream* stream = nullptr;
  se::DeviceMemoryBase current{};
  se::DeviceMemoryBase expected{};
};

// Compares two buffers on the GPU.
//
// Returns `true` if two buffers are equal, `false` otherwise.
template <typename ElementT>
static absl::StatusOr<bool> DeviceCompare(std::string_view kernel_name,
                                          void* kernel_symbol,
                                          const ComparisonParams& params) {
  se::StreamExecutor* executor = params.stream->parent();

  se::DeviceMemoryHandle out(executor, executor->AllocateScalar<uint64_t>());

  TF_RETURN_IF_ERROR(
      params.stream->MemZero(out.memory_ptr(), sizeof(uint64_t)));
  if (params.current.size() != params.expected.size()) {
    return Internal("Mismatched buffer size: %d bytes vs. %d bytes",
                    params.current.size(), params.expected.size());
  }

  se::DeviceMemory<ElementT> current_typed(params.current);
  se::DeviceMemory<ElementT> expected_typed(params.expected);
  uint64_t buffer_size = current_typed.ElementCount();

  TF_ASSIGN_OR_RETURN(
      ComparisonKernelT<ElementT> comparison_kernel,
      (se::TypedKernelFactory<
          se::DeviceMemory<ElementT>, se::DeviceMemory<ElementT>, float,
          uint64_t, se::DeviceMemory<uint64_t>>::Create(executor, kernel_name,
                                                        kernel_symbol)));

  const se::DeviceDescription& gpu_device_info =
      executor->GetDeviceDescription();

  LaunchDimensions dim =
      CalculateLaunchDimensions(*params.shape, gpu_device_info);

  se::DeviceMemory<uint64_t> as_uint64(out.memory());
  TF_RETURN_IF_ERROR(params.stream->ThenLaunch(
      dim.thread_counts_per_block(), dim.block_counts(), comparison_kernel,
      current_typed, expected_typed, static_cast<float>(params.relative_tol),
      buffer_size, as_uint64));

  uint64_t result = -1;
  CHECK_EQ(out.memory().size(), sizeof(result));
  TF_RETURN_IF_ERROR(
      params.stream->Memcpy(&result, out.memory(), sizeof(result)));
  TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());
  return result == 0;
}

// Host side comparison code that does the same thing, but reports some of the
// differences as well. It only print logs for debugging.
//
// Returns true if no differences were seen, false otherwise.
template <typename ElementType, typename ComparisonType>
static absl::StatusOr<bool> HostCompare(const ComparisonParams& params) {
  int64_t n = params.current.size() / sizeof(ElementType);
  std::vector<ElementType> host_current(n), host_expected(n);
  TF_RETURN_IF_ERROR(params.stream->Memcpy(host_current.data(), params.current,
                                           params.current.size()));
  TF_RETURN_IF_ERROR(params.stream->Memcpy(
      host_expected.data(), params.expected, params.expected.size()));
  TF_RETURN_IF_ERROR(params.stream->BlockHostUntilDone());

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
  int differences_seen = 0;

  for (int64_t i = 0; i < n && differences_seen < 10; ++i) {
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
    if (std::isfinite(current_value_canonical) !=
            std::isfinite(expected_value_canonical) ||
        !(std::abs(current_value_canonical - expected_value_canonical) /
              (std::max(std::abs(current_value_canonical),
                        std::abs(expected_value_canonical)) +
               1) <
          params.relative_tol)) {
      if (!params.verbose) return false;  // Return immediately if not verbose.
      ++differences_seen;
      LOG(ERROR) << "Difference at " << i << ": " << current_value
                 << ", expected " << expected_value;
    }
  }
  return differences_seen == 0;
}

template <typename ElementT, typename ComparisonT>
static absl::StatusOr<bool> CompareEqualParameterized(
    std::string_view kernel_name, void* kernel_symbol,
    const ComparisonParams& params) {
  XLA_SCOPED_LOGGING_TIMER("BufferComparator::CompareEqual");
  TF_ASSIGN_OR_RETURN(
      bool result, DeviceCompare<ElementT>(kernel_name, kernel_symbol, params));

  if (result) {
    return true;
  }

  TF_ASSIGN_OR_RETURN(bool host_return,
                      (HostCompare<ElementT, ComparisonT>(params)));
  CHECK_EQ(host_return, result)
      << "Host comparison succeeded even though GPU comparison failed.";
  return false;
}

absl::StatusOr<bool> BufferComparator::CompareEqual(
    se::Stream* stream, se::DeviceMemoryBase current,
    se::DeviceMemoryBase expected) const {
  ComparisonParams params{relative_tol_, verbose_, &shape_,
                          stream,        current,  expected};

  switch (shape_.element_type()) {
#if GOOGLE_CUDA  // not available for ROCm yet..
    case xla::F8E4M3FN:
      return CompareEqualParameterized<tsl::float8_e4m3fn, float>(
          "fp8_e4m3fn_comparison", buffer_comparator::fp8_e4m3fn_comparison(),
          params);
    case xla::F8E5M2:
      return CompareEqualParameterized<tsl::float8_e5m2, float>(
          "fp8_e5m2_comparison", buffer_comparator::fp8_e5m2_comparison(),
          params);
#endif  // GOOGLE_CUDA
#if TENSORFLOW_USE_ROCM && TF_ROCM_VERSION >= 60200
    case xla::F8E4M3FNUZ:
      return CompareEqualParameterized<tsl::float8_e4m3fnuz, float>(
          "fp8_e4m3fnuz_comparison",
          buffer_comparator::fp8_e4m3fnuz_comparison(), params);
    case xla::F8E5M2FNUZ:
      return CompareEqualParameterized<tsl::float8_e5m2fnuz, float>(
          "fp8_e5m2fnuz_comparison",
          buffer_comparator::fp8_e5m2fnuz_comparison(), params);
#endif  // TENSORFLOW_USE_ROCM && TF_ROCM_VERSION >= 60200
    case xla::F16:
      return CompareEqualParameterized<Eigen::half, float>(
          "fp16_comparison", buffer_comparator::fp16_comparison(), params);
    case xla::BF16:
      return CompareEqualParameterized<Eigen::bfloat16, float>(
          "bf16_comparison", buffer_comparator::bf16_comparison(), params);
    case xla::F32:
      return CompareEqualParameterized<float, float>(
          "fp32_comparison", buffer_comparator::fp32_comparison(), params);
    case xla::F64:
      return CompareEqualParameterized<double, double>(
          "fp64_comparison", buffer_comparator::fp64_comparison(), params);
    case xla::S8:
      return CompareEqualParameterized<int8_t, float>(
          "int8_comparison", buffer_comparator::int8_comparison(), params);
    case xla::S32:
      return CompareEqualParameterized<int32_t, float>(
          "int32_comparison", buffer_comparator::int32_comparison(), params);
    default:
      return Unimplemented("Unimplemented element type");
  }
}

BufferComparator::BufferComparator(const Shape& shape, double tolerance,
                                   bool verbose)
    : shape_(shape), relative_tol_(tolerance), verbose_(verbose) {
  // Normalize complex shapes: since we treat the passed array as a contiguous
  // storage it does not matter which dimension are we doubling.
  auto double_dim_size = [&]() {
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
