/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/comparison/comparison_tool.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "re2/re2.h"
#include "xla/hlo/tools/comparison/comparison_options.pb.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/tools/debug_event.pb.h"
#include "xla/tsl/platform/byte_order.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/fingerprint.h"

namespace xla::numerics::comparison {
using ::xla::LogHloOutputMetadata;
using ::xla::LogHloScope;

namespace {

// Gets the value of the linear index in a tensor as a double.
//
// This is very similar to Literal::GetLinear, except that  it converts any
// numerical types, including predicates, to double using static_cast.
//
// Args:
//   `literal`: The literal to get the value from.
//   `linear_index`: The linear index to get the value at.
//
// Returns:
//   The value of the linear index as a double, or std::nullopt if the type is
//   not supported.
std::optional<double> GetLinearAsDouble(const Literal& literal,
                                        int64_t linear_index) {
  const Shape& s = literal.shape();
  CHECK(s.IsArray());
  return primitive_util::PrimitiveTypeSwitch<std::optional<double>>(
      [&](auto primitive_type_constant) -> std::optional<double> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return static_cast<double>(literal.GetLinear<NativeT>(linear_index));
        } else if constexpr (primitive_util::IsIntegralType(
                                 primitive_type_constant)) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return static_cast<double>(literal.GetLinear<NativeT>(linear_index));
        } else if constexpr (primitive_type_constant == PrimitiveType::PRED) {
          using NativeT = primitive_util::NativeTypeOf<primitive_type_constant>;
          return static_cast<double>(literal.GetLinear<NativeT>(linear_index));
        }
        return std::nullopt;
      },
      s.element_type());
}

// Portable uniform integer distribution in [0, max_val] using rejection
// sampling. This is used instead of std::uniform_int_distribution because
// different C++ standard libraries implement uniform_int_distribution
// differently, leading to non-deterministic sampling across
// compilers/platforms.
int64_t UniformInt0ToMax(int64_t max_val, std::mt19937& gen) {
  if (max_val <= 0) {
    return 0;
  }
  uint64_t m = max_val + 1;
  int w = absl::bit_width(static_cast<uint64_t>(max_val));
  uint64_t mask = (w == 64) ? ~0ULL : ((1ULL << w) - 1);
  while (true) {
    uint64_t u = gen() & mask;
    if (u < m) {
      return static_cast<int64_t>(u);
    }
  }
}

// Portable sampling without replacement using Algorithm S (Knuth TAOCP Vol 2
// Sec 3.4.2 Algorithm S). We avoid std::sample as its algorithm and underlying
// uniform integer distribution are implementation-defined and vary between
// libstdc++ (GCC) and libc++ (LLVM).
std::vector<int64_t> SampleIndicesWithoutReplacement(int64_t total_elements,
                                                     int64_t num_samples,
                                                     std::mt19937& gen) {
  std::vector<int64_t> sampled_indices;
  sampled_indices.reserve(num_samples);
  int64_t unsampled_sz = total_elements;
  int64_t remaining_samples = num_samples;
  for (int64_t i = 0; remaining_samples > 0 && i < total_elements; ++i) {
    int64_t r = UniformInt0ToMax(unsampled_sz - 1, gen);
    --unsampled_sz;
    if (r < remaining_samples) {
      sampled_indices.push_back(i);
      --remaining_samples;
    }
  }
  return sampled_indices;
}

// Populates the samples field of the TensorSummary proto using the given
// literal and sampling parameters.
//
// Args:
//   `summary`: The TensorSummary proto to populate.
//   `literal`: The literal to sample from.
//   `sample_seed`: The seed to use for the random number generator.
//   `num_samples`: The number of samples to collect. This number must be less
//      than or equal to the number of elements in the literal.
void PopulateSamplesImpl(TensorSummary& summary, const xla::Literal& literal,
                         int64_t sample_seed, int64_t num_samples) {
  // Here we cannot use absl::BitGen because it is not consistent across process
  // runs, which is necessary because the comparison tool require deterministic
  // results across baseline and target runs.
  std::mt19937 gen(sample_seed);

  int64_t total_elements = literal.element_count();

  summary.mutable_samples()->Reserve(static_cast<int>(num_samples));

  std::vector<int64_t> sampled_indices =
      SampleIndicesWithoutReplacement(total_elements, num_samples, gen);

  for (int64_t index : sampled_indices) {
    std::optional<double> value = GetLinearAsDouble(literal, index);
    if (value.has_value()) {
      summary.add_samples(static_cast<float>(*value));
    }
  }
}

void PopulateSamples(TensorSummary& summary, const xla::Literal& literal,
                     const ComparisonOptions& options) {
  int64_t num_elements = literal.element_count();
  if (num_elements == 0) {
    return;
  }

  CHECK_GE(options.max_sample_count(), options.min_sample_count());
  CHECK_GE(options.min_sample_count(), 0);

  // Determine the target expected number of samples as a double
  double expected_samples_double;
  expected_samples_double =
      static_cast<double>(num_elements) * options.sample_ratio();

  // Apply the minimum sample count as a lower bound.
  expected_samples_double = std::max(
      expected_samples_double, static_cast<double>(options.min_sample_count()));

  // Apply the maximum sample count as an upper bound, if it's positive.
  expected_samples_double = std::min(
      expected_samples_double, static_cast<double>(options.max_sample_count()));

  // Finally, ensure the number of samples does not exceed the total number of
  // elements.
  expected_samples_double =
      std::min(expected_samples_double, static_cast<double>(num_elements));
  CHECK_GE(expected_samples_double, 0.0);

  if (expected_samples_double < 1.0) {
    return;
  }

  int64_t num_samples = static_cast<int64_t>(expected_samples_double);

  // If the expectation is to sample all elements, do that directly.
  if (num_samples == num_elements) {
    summary.mutable_samples()->Reserve(static_cast<int>(num_elements));
    for (int64_t i = 0; i < num_elements; ++i) {
      std::optional<double> value = GetLinearAsDouble(literal, i);
      if (value.has_value()) {
        summary.add_samples(static_cast<float>(*value));
      }
    }
    return;
  }
  PopulateSamplesImpl(summary, literal, options.sample_seed(), num_samples);
}

TensorSummary::TensorSummaryMetadata CreateTensorSummaryMetadata(
    const LogData& log_record, ComparisonOptions::ComparisonVariant variant) {
  TensorSummary::TensorSummaryMetadata metadata;
  metadata.set_comparison_variant(variant);
  const LogHloOutputMetadata& hlo_output_metadata =
      log_record.hlo_output_metadata();
  metadata.set_hlo_module_name(hlo_output_metadata.module_name());
  for (const LogHloScope& scope : hlo_output_metadata.scopes()) {
    if (!scope.original_value().elements().empty()) {
      const OriginalValueElementProto& leaf =
          scope.original_value().elements(0);
      TensorPosition& original_position = *metadata.add_original_positions();
      original_position.set_instruction_name(
          leaf.original_array().instruction_name());
      *original_position.mutable_shape_index() = leaf.shape_index();
    }
  }

  if (hlo_output_metadata.has_original_value()) {
    for (const OriginalValueElementProto& leaf :
         hlo_output_metadata.original_value().elements()) {
      const OriginalArrayProto& leaf_original_array = leaf.original_array();
      if (absl::c_equal(leaf_original_array.shape_index(),
                        hlo_output_metadata.shape_index())) {
        TensorPosition& last_original_position =
            *metadata.add_original_positions();
        last_original_position.set_instruction_name(
            leaf_original_array.instruction_name());
        *last_original_position.mutable_shape_index() = leaf.shape_index();
      }
    }
  }
  return metadata;
}

}  // namespace

ComparisonTool::ComparisonTool(const ComparisonOptions& options,
                               tsl::thread::ThreadPool* async_queue)
    : options_(options),
      hlo_module_name_regex_(options.hlo_module_name_regex()),
      async_queue_(async_queue) {
  if (async_queue_ == nullptr) {
    owned_async_queue_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "comparison_tool", 1);
    async_queue_ = owned_async_queue_.get();
  }
}

TensorSummary ComparisonTool::CreateTensorSummary(const LogData& log_record,
                                                  const xla::Literal& literal) {
  TensorSummary summary;
  *summary.mutable_metadata() =
      CreateTensorSummaryMetadata(log_record, options_.comparison_variant());

  // TODO(tgeng): Add support for sparse arrays.
  auto& shape = literal.shape();
  CHECK(shape.IsArray() && shape.IsArray());
  *summary.mutable_shape() = shape.ToProto();

  int64_t num_elements = literal.element_count();

  double sum_val = 0;
  double non_zero_sum_val = 0;
  double non_zero_count = 0;
  double min_val = std::numeric_limits<double>::infinity();
  double max_val = -std::numeric_limits<double>::infinity();
  int64_t valid_elements_count = 0;
  uint64_t hash = 0;

  for (int64_t i = 0; i < num_elements; ++i) {
    std::optional<double> value_opt = GetLinearAsDouble(literal, i);
    if (value_opt.has_value()) {
      double value = *value_opt;
      hash = tsl::FingerprintCat64(hash, absl::bit_cast<uint64_t>(value));
      if (!std::isnan(value)) {
        min_val = std::min(min_val, value);
        max_val = std::max(max_val, value);
        sum_val += value;
        valid_elements_count++;
        if (value != 0.0) {
          non_zero_sum_val += value;
          non_zero_count++;
        }
      }
    }
  }

  if (valid_elements_count > 0) {
    double mean = sum_val / static_cast<double>(valid_elements_count);
    double sum_of_squares = 0;
    double non_zero_mean =
        non_zero_count > 0
            ? static_cast<double>(non_zero_sum_val / non_zero_count)
            : std::numeric_limits<double>::quiet_NaN();
    double non_zero_sum_of_squares =
        non_zero_count > 0 ? 0 : std::numeric_limits<double>::quiet_NaN();
    for (int64_t i = 0; i < num_elements; ++i) {
      std::optional<double> value_opt = GetLinearAsDouble(literal, i);
      if (value_opt.has_value()) {
        double value = *value_opt;
        if (!std::isnan(value)) {
          sum_of_squares += (value - mean) * (value - mean);
          if (value != 0.0) {
            non_zero_sum_of_squares +=
                (value - non_zero_mean) * (value - non_zero_mean);
          }
        }
      }
    }
    summary.set_mean(static_cast<float>(mean));
    summary.set_min(static_cast<float>(min_val));
    summary.set_max(static_cast<float>(max_val));
    summary.set_stddev(static_cast<float>(
        std::sqrt(sum_of_squares / static_cast<double>(valid_elements_count))));
    summary.set_non_zero_mean(static_cast<float>(non_zero_mean));
    if (non_zero_count > 0) {
      summary.set_non_zero_stddev(static_cast<float>(std::sqrt(
          non_zero_sum_of_squares / static_cast<double>(non_zero_count))));
    } else {
      summary.set_non_zero_stddev(
          static_cast<float>(std::numeric_limits<double>::quiet_NaN()));
    }
  } else {
    // Handle cases with no valid numeric elements (e.g. empty tensor, all NaNs)
    summary.set_mean(static_cast<float>(NAN));
    summary.set_min(static_cast<float>(NAN));
    summary.set_max(static_cast<float>(NAN));
    summary.set_stddev(static_cast<float>(NAN));
  }

  uint64_t be_hash = tsl::port::kLittleEndian ? absl::byteswap(hash) : hash;
  summary.set_checksum(
      std::string(reinterpret_cast<const char*>(&be_hash), sizeof(be_hash)));

  // Populate the samples using the implemented method
  PopulateSamples(summary, literal, options_);

  return summary;
}

void ComparisonTool::RecordTensor(
    const LogData& log_record,
    const std::shared_ptr<const xla::Literal>& literal) {
  if (!RE2::FullMatch(log_record.hlo_output_metadata().module_name(),
                      hlo_module_name_regex_)) {
    return;
  }

  absl::MutexLock lock(mutex_);
  ModuleStats& module_stats =
      module_stats_map_[log_record.hlo_output_metadata().module_name()];
  module_stats.set_num_tensors_recorded(module_stats.num_tensors_recorded() +
                                        1);

  // TODO(tgeng): Handle sharded tensors.
  LOG(INFO) << "[Comparison Tool] RecordTensor with shape "
            << literal->shape().ToString() << " and metadata:\n"
            << log_record.hlo_output_metadata().DebugString();
  if (!log_record.hlo_output_metadata().has_original_value()) {
    LOG(WARNING) << "[Comparison Tool] Skipped comparison because no original "
                    "value found in metadata for instruction "
                 << log_record.hlo_output_metadata().instruction_name();
    return;
  }
  module_stats.set_num_tensors_with_original_value(
      module_stats.num_tensors_with_original_value() + 1);

  // Copy the inputs to ensure they are not destroyed before the async
  // processing is done.
  LogData log_record_copy = log_record;

  // We need to run this asynchronously because blocking the logging handler
  // can cause tensors to be dropped. This is because the TPU continues to
  // generate new tensors while the logging handler is running.
  async_queue_->Schedule([this, log_record_copy = std::move(log_record_copy),
                          literal = literal]() {
    TensorSummary summary = CreateTensorSummary(log_record_copy, *literal);
    absl::Status status = ProcessTensorSummary(
        log_record_copy.hlo_output_metadata().module_name(), summary);
    if (!status.ok()) {
      LOG(ERROR) << "[Comparison Tool] Asynchronous processing failed for "
                 << log_record_copy.hlo_output_metadata().instruction_name()
                 << " from module "
                 << log_record_copy.hlo_output_metadata().module_name() << ": "
                 << status;
    }
  });
}

absl::Status ComparisonTool::RegisterOriginalHloModule(
    const xla::HloModuleProto& module) {
  if (!RE2::FullMatch(module.name(), hlo_module_name_regex_)) {
    LOG(INFO) << "[Comparison Tool] Skipping registering original HLO module '"
              << module.name() << "' because it doesn't match the regex.";
    return absl::OkStatus();
  }
  absl::MutexLock lock(mutex_);
  if (registered_hlo_module_names_.contains(module.name())) {
    LOG(INFO) << "[Comparison Tool] Skipping registering original HLO module '"
              << module.name() << "' because it is already registered.";
    return absl::OkStatus();
  }
  absl::Status status = RegisterOriginalHloModuleImpl(module);
  if (status.ok()) {
    registered_hlo_module_names_.insert(module.name());
  }
  return status;
}

}  // namespace xla::numerics::comparison
