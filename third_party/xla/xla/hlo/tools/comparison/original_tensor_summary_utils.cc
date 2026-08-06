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

#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/tools/comparison/comparison_result.pb.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/permutation_util.h"
#include "xla/shape_util.h"
#include "xla/tools/debug_event.pb.h"

namespace xla::numerics::comparison {

ProgressReporter::ProgressReporter(::absl::string_view message_prefix,
                                   int64_t total_count, bool use_percent,
                                   std::optional<::absl::Duration> log_interval)
    : message_prefix_(message_prefix),
      total_count_(total_count),
      log_interval_(log_interval.value_or(::absl::Milliseconds(100))),
      processed_count_(0),
      last_log_time_(::absl::InfinitePast()),
      use_percent_(use_percent) {}

ProgressReporter::~ProgressReporter() {
  ::absl::Time now = ::absl::Now();
  std::string time_str =
      ::absl::FormatTime("%Y-%m-%d %H:%M:%S", now, ::absl::LocalTimeZone());
  ::absl::FPrintF(stderr, "\r[%s] [Done] %s\033[K\n", time_str.c_str(),
                  message_prefix_.c_str());
}

void ProgressReporter::Report(int64_t new_processed_count,
                              int64_t new_total_count) {
  if (new_processed_count >= 0) {
    processed_count_ = new_processed_count;
  } else {
    processed_count_++;
  }
  if (new_total_count >= 0) {
    total_count_ = new_total_count;
  }
  ::absl::Time now = ::absl::Now();
  if (now - last_log_time_ >= log_interval_) {
    std::string time_str =
        ::absl::FormatTime("%Y-%m-%d %H:%M:%S", now, ::absl::LocalTimeZone());
    if (total_count_ > 0) {
      if (use_percent_) {
        ::absl::FPrintF(stderr, "\r[%s] [%.2f%%] %s", time_str.c_str(),
                        processed_count_ * 100.0 / total_count_,
                        message_prefix_.c_str());
      } else {
        ::absl::FPrintF(stderr, "\r[%s] [%ld/%ld] %s", time_str.c_str(),
                        processed_count_, total_count_,
                        message_prefix_.c_str());
      }
    } else {
      ::absl::FPrintF(stderr, "\r[%s] [%ld] %s", time_str.c_str(),
                      processed_count_, message_prefix_.c_str());
    }
    // Clear the line in case the previous message was longer
    ::absl::FPrintF(stderr, "\033[K");
    last_log_time_ = now;
  }
}

namespace tensor_transformation {

std::string ToString(const TensorTransformation* transformation) {
  if (transformation == nullptr) {
    return "nullptr";
  }
  return ::absl::StrCat(*transformation);
}

void ToProto(const TensorTransformation* transformation,
             google::protobuf::RepeatedPtrField<TensorTransformationProto>* proto_field) {
  while (transformation != nullptr) {
    TensorTransformationProto* proto = proto_field->Add();
    if (std::holds_alternative<Reshape>(*transformation)) {
      const auto& reshape = std::get<Reshape>(*transformation);
      auto* reshape_proto = proto->mutable_reshape();
      for (int64_t dim : reshape.output_dimensions) {
        reshape_proto->add_output_dimensions(dim);
      }
      transformation = reshape.continuation.get();
    } else if (std::holds_alternative<Broadcast>(*transformation)) {
      const auto& broadcast = std::get<Broadcast>(*transformation);
      auto* broadcast_proto = proto->mutable_broadcast();
      for (int64_t dim : broadcast.output_dimensions) {
        broadcast_proto->add_output_dimensions(dim);
      }
      for (int64_t dim : broadcast.broadcast_dimensions) {
        broadcast_proto->add_broadcast_dimensions(dim);
      }
      transformation = broadcast.continuation.get();
    } else if (std::holds_alternative<Unshard>(*transformation)) {
      const auto& unshard = std::get<Unshard>(*transformation);
      auto* unshard_proto = proto->mutable_unshard();
      for (int64_t dim : unshard.original_dimensions) {
        unshard_proto->add_original_dimensions(dim);
      }
      *unshard_proto->mutable_sharding() = unshard.sharding.ToProto();
      transformation = unshard.continuation.get();
    } else {
      CHECK(false) << "Unknown transformation type";
    }
  }
}

::absl::StatusOr<std::shared_ptr<const TensorTransformation>> FromProto(
    const google::protobuf::RepeatedPtrField<TensorTransformationProto>& proto_field) {
  std::shared_ptr<const TensorTransformation> transformation = nullptr;
  for (int i = proto_field.size() - 1; i >= 0; --i) {
    const auto& proto = proto_field.Get(i);
    if (proto.has_reshape()) {
      std::vector<int64_t> output_dimensions(
          proto.reshape().output_dimensions().begin(),
          proto.reshape().output_dimensions().end());
      transformation = std::make_shared<const TensorTransformation>(
          Reshape{/*continuation=*/transformation,
                  /*output_dimensions=*/std::move(output_dimensions)});
    } else if (proto.has_broadcast()) {
      std::vector<int64_t> output_dimensions(
          proto.broadcast().output_dimensions().begin(),
          proto.broadcast().output_dimensions().end());
      std::vector<int64_t> broadcast_dimensions(
          proto.broadcast().broadcast_dimensions().begin(),
          proto.broadcast().broadcast_dimensions().end());
      transformation = std::make_shared<const TensorTransformation>(
          Broadcast{/*continuation=*/transformation,
                    /*output_dimensions=*/std::move(output_dimensions),
                    /*broadcast_dimensions=*/std::move(broadcast_dimensions)});
    } else if (proto.has_unshard()) {
      std::vector<int64_t> original_dimensions(
          proto.unshard().original_dimensions().begin(),
          proto.unshard().original_dimensions().end());
      ::absl::StatusOr<HloSharding> hlo_sharding =
          HloSharding::FromProto(proto.unshard().sharding());
      if (!hlo_sharding.ok()) {
        return hlo_sharding.status();
      }
      transformation = std::make_shared<const TensorTransformation>(
          Unshard{/*continuation=*/transformation,
                  /*original_dimensions=*/std::move(original_dimensions),
                  /*sharding=*/*hlo_sharding});
    } else {
      return ::absl::InvalidArgumentError(
          "Unknown transformation type in proto");
    }
  }
  return transformation;
}

namespace {
// Helper function to recursively append a continuation to the end of a
// transformation chain.
// This function exists separately from `AppendContinuation` to avoid
// compilation issues. If this logic is folded into `AppendContinuation`, the
// recursive call to `AppendContinuation` inside the std::visit lambda becomes
// ambiguous. This is because the compiler sees the function declaration
// from the header file and the function definition being currently parsed,
// and cannot definitively choose between them within the dependent context of
// the template. Having a distinct name for the recursive implementation avoids
// this name lookup ambiguity.
std::shared_ptr<const TensorTransformation> AppendContinuationRecursive(
    std::shared_ptr<const TensorTransformation> current_transformation,
    std::shared_ptr<const TensorTransformation> to_append) {
  if (current_transformation == nullptr) {
    return to_append;
  }

  return std::visit(
      [&](auto arg) -> std::shared_ptr<const TensorTransformation> {
        arg.continuation =
            AppendContinuationRecursive(arg.continuation, to_append);
        return std::make_shared<const TensorTransformation>(arg);
      },
      *current_transformation);
}
}  // namespace

std::shared_ptr<const TensorTransformation> AppendContinuation(
    std::shared_ptr<const TensorTransformation> current_transformation,
    std::shared_ptr<const TensorTransformation> to_append) {
  if (current_transformation == nullptr) {
    return to_append;
  }
  if (to_append == nullptr) {
    return current_transformation;
  }
  return AppendContinuationRecursive(current_transformation, to_append);
}

}  // namespace tensor_transformation

using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;
using FloatSummary = ::xla::comparison::FloatSummary;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;
using tensor_transformation::Broadcast;
using tensor_transformation::Reshape;
using tensor_transformation::TensorTransformation;
using tensor_transformation::Unshard;

FloatSummary ApplyReshapeToSummary(
    const FloatSummary& summary,
    const ::absl::Span<const int64_t> current_shape,
    const ::absl::Span<const int64_t> new_shape) {
  int d_in = current_shape.size();
  int d_out = new_shape.size();

  int prefix_len = 0;
  while (prefix_len < d_in && prefix_len < d_out &&
         current_shape[prefix_len] == new_shape[prefix_len]) {
    prefix_len++;
  }

  int suffix_len = 0;
  while (suffix_len < d_in && suffix_len < d_out &&
         d_in - 1 - suffix_len >= 0 && d_out - 1 - suffix_len >= 0 &&
         current_shape[d_in - 1 - suffix_len] ==
             new_shape[d_out - 1 - suffix_len]) {
    if (prefix_len > d_in - 1 - suffix_len ||
        prefix_len > d_out - 1 - suffix_len) {
      // Stop if suffix overlaps with prefix
      break;
    }
    suffix_len++;
  }

  std::vector<DimSplitSpec> new_split_spec;
  std::vector<int> kept_split_indices;  // Indices in the original split_spec

  for (int j = 0; j < summary.split_spec.size(); ++j) {
    const auto& spec = summary.split_spec[j];
    int64_t dim = spec.dim_index;
    if (dim < prefix_len) {
      // Dimension is in the unchanged prefix.
      new_split_spec.push_back(spec);
      kept_split_indices.push_back(j);
    } else if (dim >= d_in - suffix_len) {
      // Dimension is in the unchanged suffix.
      int64_t new_dim = dim + (d_out - d_in);
      new_split_spec.push_back(
          {/*dim_index=*/new_dim, /*block_count=*/spec.block_count});
      kept_split_indices.push_back(j);
    }
    // Otherwise, the dimension `dim` is part of the reshaped portion.
    // The split along this dimension is no longer valid and will be merged.
  }

  if (kept_split_indices.size() == summary.split_spec.size()) {
    // No split dimensions were affected, so block summaries remain the same.
    // The split_spec is already updated.
    return {/*block_summaries=*/summary.block_summaries,
            /*split_spec=*/new_split_spec};
  }

  // Some split dimensions are affected and need to be merged.
  // Group block summaries by the indices of the dimensions that were NOT
  // affected.
  ::absl::flat_hash_map<std::vector<int64_t>, std::vector<FloatBlockSummary>>
      groups;
  for (const auto& block : summary.block_summaries) {
    std::vector<int64_t> key;
    key.reserve(kept_split_indices.size());
    for (int k : kept_split_indices) {
      CHECK_LT(k, block.block_indices.size());
      key.push_back(block.block_indices[k]);
    }
    groups[key].push_back(block);
  }

  // Combine the summaries within each group.
  std::vector<FloatBlockSummary> new_block_summaries;
  // NOLINTNEXTLINE
  for (const auto& pair : groups) {
    // The new block indices are the keys of the groups.
    new_block_summaries.push_back(
        ::xla::comparison::CombineBlockSummaries(pair.first, pair.second));
  }

  // Sort new_block_summaries to maintain a canonical order.
  std::sort(new_block_summaries.begin(), new_block_summaries.end(),
            [](const auto& a, const auto& b) {
              return a.block_indices < b.block_indices;
            });

  return {/*block_summaries=*/new_block_summaries,
          /*split_spec=*/new_split_spec};
}

::xla::comparison::FloatSummary ApplyTransposeToSummary(
    const ::xla::comparison::FloatSummary& summary,
    ::absl::Span<const int64_t> current_shape,
    ::absl::Span<const int64_t> permutation) {
  CHECK_EQ(current_shape.size(), permutation.size());
  std::vector<int64_t> current_shape_vec(current_shape.begin(),
                                         current_shape.end());
  std::vector<int64_t> new_shape_vec =
      xla::Permute(current_shape_vec, permutation);
  return ApplyBroadcastToSummary(summary, current_shape, new_shape_vec,
                                 InversePermutation(permutation));
}

::xla::comparison::FloatSummary ApplyBroadcastToSummary(
    const ::xla::comparison::FloatSummary& summary,
    ::absl::Span<const int64_t> current_shape,
    ::absl::Span<const int64_t> new_shape,
    ::absl::Span<const int64_t> broadcast_dimensions) {
  if (summary.block_summaries.empty()) {
    return summary;
  }

  int64_t current_elements = std::accumulate(
      current_shape.begin(), current_shape.end(), 1LL, std::multiplies<>());
  int64_t new_elements = std::accumulate(new_shape.begin(), new_shape.end(),
                                         1LL, std::multiplies<>());

  double count_multiplier =
      current_elements == 0
          ? 1.0
          : static_cast<double>(new_elements) / current_elements;

  if (summary.split_spec.empty()) {
    FloatSummary new_summary = summary;
    for (auto& block : new_summary.block_summaries) {
      block.count *= count_multiplier;
    }
    return new_summary;
  }

  std::vector<std::pair<DimSplitSpec, int>> kept_split_spec_with_orig_k;
  kept_split_spec_with_orig_k.reserve(summary.split_spec.size());
  bool needs_merging = false;
  for (int k = 0; k < summary.split_spec.size(); ++k) {
    const auto& spec = summary.split_spec[k];
    if (broadcast_dimensions[spec.dim_index] == -1) {
      needs_merging = true;
    } else {
      kept_split_spec_with_orig_k.push_back(
          {{/*dim_index=*/broadcast_dimensions[spec.dim_index],
            /*block_count=*/spec.block_count},
           k});
    }
  }

  std::sort(kept_split_spec_with_orig_k.begin(),
            kept_split_spec_with_orig_k.end(),
            [](const auto& a, const auto& b) {
              return a.first.dim_index < b.first.dim_index;
            });

  std::vector<DimSplitSpec> new_split_spec;
  new_split_spec.reserve(kept_split_spec_with_orig_k.size());
  std::vector<int> k_indices;
  k_indices.reserve(kept_split_spec_with_orig_k.size());
  for (const auto& pair : kept_split_spec_with_orig_k) {
    new_split_spec.push_back(pair.first);
    k_indices.push_back(pair.second);
  }

  std::vector<FloatBlockSummary> new_block_summaries;
  if (!needs_merging) {
    new_block_summaries.reserve(summary.block_summaries.size());
    for (const auto& block : summary.block_summaries) {
      std::vector<int64_t> new_block_indices(block.block_indices.size());
      for (int j = 0; j < k_indices.size(); ++j) {
        new_block_indices[j] = block.block_indices[k_indices[j]];
      }
      float new_count = std::round(block.count * count_multiplier);
      float new_nan_count = std::round(block.nan_count * count_multiplier);
      float new_pos_inf_count =
          std::round(block.pos_inf_count * count_multiplier);
      float new_neg_inf_count =
          std::round(block.neg_inf_count * count_multiplier);
      float new_zero_count = std::round(block.zero_count * count_multiplier);
      new_block_summaries.push_back({/*block_indices=*/new_block_indices,
                                     /*min=*/block.min,
                                     /*max=*/block.max,
                                     /*mean=*/block.mean,
                                     /*stddev=*/block.stddev,
                                     /*count=*/new_count,
                                     /*nan_count=*/new_nan_count,
                                     /*pos_inf_count=*/new_pos_inf_count,
                                     /*neg_inf_count=*/new_neg_inf_count,
                                     /*zero_count=*/new_zero_count});
    }
  } else {
    ::absl::flat_hash_map<std::vector<int64_t>, std::vector<FloatBlockSummary>>
        groups;
    for (const auto& block : summary.block_summaries) {
      std::vector<int64_t> key;
      key.reserve(k_indices.size());
      for (int k : k_indices) {
        key.push_back(block.block_indices[k]);
      }
      groups[key].push_back(block);
    }
    // NOLINTNEXTLINE
    for (const auto& pair : groups) {
      auto merged_block =
          ::xla::comparison::CombineBlockSummaries(pair.first, pair.second);
      merged_block.count *= count_multiplier;
      new_block_summaries.push_back(merged_block);
    }
  }

  // Sort new_block_summaries to maintain a canonical order.
  std::sort(new_block_summaries.begin(), new_block_summaries.end(),
            [](const auto& a, const auto& b) {
              return a.block_indices < b.block_indices;
            });
  return {/*block_summaries=*/new_block_summaries,
          /*split_spec=*/new_split_spec};
}

::absl::StatusOr<OriginalTensorSummary>
ApplyNonUnshardTensorTransformationToSummary(
    const OriginalTensorSummary& original_tensor_summary,
    const TensorTransformation* transformation,
    const TensorTransformation* stopping_transformation) {
  std::vector<FloatSummary> current_summaries =
      original_tensor_summary.summaries;
  std::vector<int64_t> current_dimensions(
      original_tensor_summary.dimensions.begin(),
      original_tensor_summary.dimensions.end());

  while (transformation != nullptr) {
    // Stop if we have reached the stopping_transformation by value.
    if (stopping_transformation != nullptr &&
        *transformation == *stopping_transformation) {
      break;
    }

    if (std::holds_alternative<Reshape>(*transformation)) {
      const auto& reshape = std::get<Reshape>(*transformation);
      std::vector<FloatSummary> new_summaries;
      new_summaries.reserve(current_summaries.size());
      for (int i = 0; i < current_summaries.size(); ++i) {
        new_summaries.push_back(
            ApplyReshapeToSummary(current_summaries[i], current_dimensions,
                                  reshape.output_dimensions));
      }
      current_summaries = new_summaries;
      current_dimensions = reshape.output_dimensions;
      transformation = reshape.continuation.get();
    } else if (std::holds_alternative<Broadcast>(*transformation)) {
      const auto& broadcast = std::get<Broadcast>(*transformation);
      std::vector<FloatSummary> new_summaries;
      new_summaries.reserve(current_summaries.size());
      for (int i = 0; i < current_summaries.size(); ++i) {
        new_summaries.push_back(ApplyBroadcastToSummary(
            current_summaries[i], current_dimensions,
            broadcast.output_dimensions, broadcast.broadcast_dimensions));
      }
      current_summaries = new_summaries;
      current_dimensions = broadcast.output_dimensions;
      transformation = broadcast.continuation.get();
    } else if (std::holds_alternative<Unshard>(*transformation)) {
      return ::absl::InvalidArgumentError(
          "Unshard transformation is not supported in "
          "ApplyNonUnshardTensorTransformationToSummary");
    } else {
      return ::absl::InternalError("Unknown transformation type");
    }
  }
  return ::absl::StatusOr<OriginalTensorSummary>(
      {/*dimensions=*/current_dimensions, /*summaries=*/current_summaries});
}

namespace {
::xla::comparison::FloatSummary CreateAlignedSummary(
    const OriginalTensorSummary& original_summary,
    const std::vector<::xla::comparison::DimSplitSpec>& aligned_split_spec,
    int summary_index) {
  using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;
  ::absl::flat_hash_map<int64_t, int> dim_to_orig_spec_idx;
  for (int i = 0;
       i < original_summary.summaries[summary_index].split_spec.size(); ++i) {
    dim_to_orig_spec_idx
        [original_summary.summaries[summary_index].split_spec[i].dim_index] = i;
  }

  std::map<std::vector<int64_t>, std::vector<FloatBlockSummary>>
      blocks_to_merge;

  for (const auto& block :
       original_summary.summaries[summary_index].block_summaries) {
    std::vector<int64_t> aligned_indices;
    aligned_indices.reserve(aligned_split_spec.size());
    for (const auto& aligned_spec : aligned_split_spec) {
      auto it = dim_to_orig_spec_idx.find(aligned_spec.dim_index);
      CHECK(it != dim_to_orig_spec_idx.end());
      int orig_spec_idx = it->second;
      int64_t orig_block_idx = block.block_indices[orig_spec_idx];
      int64_t orig_block_count = original_summary.summaries[summary_index]
                                     .split_spec[orig_spec_idx]
                                     .block_count;
      int64_t aligned_block_count = aligned_spec.block_count;
      aligned_indices.push_back(orig_block_idx * aligned_block_count /
                                orig_block_count);
    }
    blocks_to_merge[aligned_indices].push_back(block);
  }

  ::xla::comparison::FloatSummary aligned_summary;
  aligned_summary.split_spec = aligned_split_spec;
  for (auto const& [aligned_indices, block_list] : blocks_to_merge) {
    aligned_summary.block_summaries.push_back(
        ::xla::comparison::CombineBlockSummaries(aligned_indices, block_list));
  }
  return aligned_summary;
}

}  // namespace
::absl::StatusOr<std::pair<OriginalTensorSummary, OriginalTensorSummary>>
AlignTensorSummaries(const OriginalTensorSummary& baseline_tensor_summary,
                     const OriginalTensorSummary& target_tensor_summary) {
  if (baseline_tensor_summary.summaries.size() !=
      target_tensor_summary.summaries.size()) {
    return ::absl::InvalidArgumentError(
        "Baseline and target tensor summaries different number of summaries.");
  }
  OriginalTensorSummary baseline_aligned, target_aligned;
  baseline_aligned.dimensions = baseline_tensor_summary.dimensions;
  baseline_aligned.summaries.reserve(baseline_tensor_summary.summaries.size());
  target_aligned.dimensions = target_tensor_summary.dimensions;
  target_aligned.summaries.reserve(target_tensor_summary.summaries.size());
  for (int i = 0; i < baseline_tensor_summary.summaries.size(); ++i) {
    ::absl::flat_hash_map<int64_t, int64_t> baseline_splits, target_splits;
    for (const auto& spec : baseline_tensor_summary.summaries[i].split_spec) {
      baseline_splits[spec.dim_index] = spec.block_count;
    }
    for (const auto& spec : target_tensor_summary.summaries[i].split_spec) {
      target_splits[spec.dim_index] = spec.block_count;
    }

    std::vector<DimSplitSpec> aligned_split_spec_vec;
    for (int64_t i = 0; i < baseline_tensor_summary.dimensions.size(); ++i) {
      auto baseline_it = baseline_splits.find(i);
      auto target_it = target_splits.find(i);
      if (baseline_it != baseline_splits.end() &&
          target_it != target_splits.end()) {
        int64_t gcd = std::gcd(baseline_it->second, target_it->second);
        if (gcd > 1) {
          aligned_split_spec_vec.push_back(
              {/*dim_index=*/i, /*block_count=*/gcd});
        }
      }
    }
    std::sort(aligned_split_spec_vec.begin(), aligned_split_spec_vec.end(),
              [](const DimSplitSpec& a, const DimSplitSpec& b) {
                return a.dim_index < b.dim_index;
              });
    baseline_aligned.summaries.push_back(CreateAlignedSummary(
        baseline_tensor_summary, aligned_split_spec_vec, i));
    target_aligned.summaries.push_back(
        CreateAlignedSummary(target_tensor_summary, aligned_split_spec_vec, i));
  }
  return std::make_pair(baseline_aligned, target_aligned);
}

AbsoluteScopedTensorKey GetAbsoluteScopedTensorKey(
    xla::LogHloOutputMetadata log_hlo_output_metadata) {
  std::vector<ScopeInstruction> scope_instructions;
  scope_instructions.reserve(log_hlo_output_metadata.scopes_size());
  for (const auto& scope : log_hlo_output_metadata.scopes()) {
    scope_instructions.push_back(
        ScopeInstruction::Create(scope.instruction_name(), scope.it_count()));
  }
  return AbsoluteScopedTensorKey::Create(
      TensorKey::Create(log_hlo_output_metadata.instruction_name(),
                        ShapeIndex(log_hlo_output_metadata.shape_index())),
      std::move(scope_instructions));
}

namespace {

DimSplitSpec DimSplitSpecFromProto(
    const xla::numerics::comparison::TensorSummaryProto::DimSplitSpecProto&
        proto) {
  return {/*dim_index=*/proto.dim_index(), /*block_count=*/proto.block_count()};
}

xla::numerics::comparison::TensorSummaryProto::DimSplitSpecProto
DimSplitSpecToProto(const DimSplitSpec& spec) {
  xla::numerics::comparison::TensorSummaryProto::DimSplitSpecProto proto;
  proto.set_dim_index(spec.dim_index);
  proto.set_block_count(spec.block_count);
  return proto;
}

FloatBlockSummary FloatBlockSummaryFromProto(
    const TensorSummaryProto::BlockSummaryProto& proto) {
  return {
      /*block_indices=*/std::vector<int64_t>(proto.block_indices().begin(),
                                             proto.block_indices().end()),
      /*min=*/proto.min(),
      /*max=*/proto.max(),
      /*mean=*/proto.mean(),
      /*stddev=*/proto.stddev(),
      /*count=*/proto.count(),
      /*nan_count=*/proto.nan_count(),
      /*pos_inf_count=*/proto.pos_inf_count(),
      /*neg_inf_count=*/proto.neg_inf_count(),
      /*zero_count=*/proto.zero_count(),
  };
}

TensorSummaryProto::BlockSummaryProto FloatBlockSummaryToProto(
    const FloatBlockSummary& summary) {
  TensorSummaryProto::BlockSummaryProto proto;
  for (int64_t index : summary.block_indices) {
    proto.add_block_indices(index);
  }
  proto.set_min(summary.min);
  proto.set_max(summary.max);
  proto.set_mean(summary.mean);
  proto.set_stddev(summary.stddev);
  proto.set_count(summary.count);
  proto.set_nan_count(summary.nan_count);
  proto.set_pos_inf_count(summary.pos_inf_count);
  proto.set_neg_inf_count(summary.neg_inf_count);
  proto.set_zero_count(summary.zero_count);
  return proto;
}

FloatSummary FloatSummaryFromProto(const TensorSummaryProto& proto) {
  FloatSummary summary;
  for (const auto& spec_proto : proto.split_spec()) {
    summary.split_spec.push_back(DimSplitSpecFromProto(spec_proto));
  }
  for (const auto& block_proto : proto.block_summaries()) {
    summary.block_summaries.push_back(FloatBlockSummaryFromProto(block_proto));
  }
  return summary;
}

TensorSummaryProto FloatSummaryToProto(const FloatSummary& summary) {
  TensorSummaryProto proto;
  for (const auto& spec : summary.split_spec) {
    *proto.add_split_spec() = DimSplitSpecToProto(spec);
  }
  for (const auto& block : summary.block_summaries) {
    *proto.add_block_summaries() = FloatBlockSummaryToProto(block);
  }
  return proto;
}
}  // namespace

RecoveredTensorSummaryProto::OriginalTensorSummaryProto
OriginalTensorSummary::ToProto() const {
  RecoveredTensorSummaryProto::OriginalTensorSummaryProto proto;
  for (int64_t dim : dimensions) {
    proto.add_dimensions(dim);
  }
  for (const auto& summary : summaries) {
    *proto.add_summaries() = FloatSummaryToProto(summary);
  }
  return proto;
}

OriginalTensorSummary OriginalTensorSummary::FromProto(
    const RecoveredTensorSummaryProto::OriginalTensorSummaryProto& proto) {
  OriginalTensorSummary summary;
  summary.dimensions = std::vector<int64_t>(proto.dimensions().begin(),
                                            proto.dimensions().end());
  for (const auto& summary_proto : proto.summaries()) {
    summary.summaries.push_back(FloatSummaryFromProto(summary_proto));
  }
  return summary;
}

RecoveredTensorSummaryProto CreateRecoveredTensorSummaryProto(
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  RecoveredTensorSummaryProto proto;
  *proto.mutable_tensor_key() = original_tensor_key.ToProto();
  tensor_transformation::ToProto(pending_transformation.get(),
                                 proto.mutable_pending_transformation());
  *proto.mutable_original_tensor_summary() = original_tensor_summary.ToProto();
  return proto;
}

::absl::StatusOr<RecoveredTensorSummary> RecoveredTensorSummaryFromProto(
    const RecoveredTensorSummaryProto& proto) {
  ::absl::StatusOr<
      std::shared_ptr<const tensor_transformation::TensorTransformation>>
      pending_transformation =
          tensor_transformation::FromProto(proto.pending_transformation());
  if (!pending_transformation.ok()) {
    return pending_transformation.status();
  }
  return RecoveredTensorSummary{
      /*original_tensor_key=*/ScopedTensorKey::FromProto(proto.tensor_key()),
      /*pending_transformation=*/*pending_transformation,
      /*original_tensor_summary=*/
      OriginalTensorSummary::FromProto(proto.original_tensor_summary())};
}

bool IsCallLike(const HloInstruction& instr) {
  switch (instr.opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kWhile:
    case HloOpcode::kConditional:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
      // Note that kFusion should not be included here because it does not
      // actually correspond to any calls in the original graph. It's also not
      // reported by logging in LogHloOutputMetadata.scopes.
      return true;
    default:
      return false;
  }
}

}  // namespace xla::numerics::comparison
