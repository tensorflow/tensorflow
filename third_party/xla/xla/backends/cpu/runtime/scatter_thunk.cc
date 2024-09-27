/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/scatter_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/index_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<ScatterThunk>> ScatterThunk::Create(
    Info info, std::vector<Operand> operands, ScatterIndices scatter_indices,
    std::vector<Update> updates, std::vector<Result> results,
    const ScatterDimensionNumbers& dim_numbers, std::string functor_name) {
  return absl::WrapUnique(new ScatterThunk(
      std::move(info), std::move(operands), std::move(scatter_indices),
      std::move(updates), std::move(results), dim_numbers,
      std::move(functor_name)));
}

ScatterThunk::ScatterThunk(Info info, std::vector<Operand> operands,
                           ScatterIndices scatter_indices,
                           std::vector<Update> updates,
                           std::vector<Result> results,
                           const ScatterDimensionNumbers& dim_numbers,
                           std::string functor_name)
    : Thunk(Kind::kScatter, std::move(info)),
      operands_(std::move(operands)),
      scatter_indices_(std::move(scatter_indices)),
      updates_(std::move(updates)),
      results_(std::move(results)),
      dim_numbers_(dim_numbers),
      functor_name_(std::move(functor_name)) {
  // Append dimension of size 1 to scatter indices if index vector dimension
  // is equal to the scatter indices rank (see operation specification).
  if (scatter_indices_.shape.dimensions_size() ==
      dim_numbers_.index_vector_dim()) {
    scatter_indices_.shape.add_dimensions(1);
  }
}

namespace {

// Scatter index iteration is copy-pasted from HloEvaluator::HandleScatter. It
// is intentionally copied because we want to detect when they'll get out of
// sync and don't want to accidentally introduce bugs in two backends.
//
// Also here we optimize it for performance, while HloEvaluator is more
// focused on correctness.

template <bool for_update_window_indices>
ShapeUtil::IndexIterationSpace GetIterationSpaceImpl(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  int64_t updates_rank = updates_dims.size();
  std::vector<int64_t> index_base(updates_rank, 0);
  std::vector<int64_t> index_count(updates_rank, 1);
  for (int64_t i = 0; i < updates_rank; i++) {
    if constexpr (for_update_window_indices) {
      bool is_update_window_dim =
          absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_window_dim) {
        index_count[i] = updates_dims[i];
      }
    } else {
      bool is_update_scatter_dim =
          !absl::c_binary_search(dim_numbers.update_window_dims(), i);
      if (is_update_scatter_dim) {
        index_count[i] = updates_dims[i];
      }
    }
  }
  return {std::move(index_base), std::move(index_count),
          std::vector<int64_t>(updates_rank, 1)};
}

// Returns an ShapeUtil::IndexIterationSpace that iterates over the update
// scatter dimensions while keeping the rest of the update dimensions clamped
// to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForUpdateScatterIndices(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  return GetIterationSpaceImpl</*for_update_window_indices=*/false>(
      updates_dims, dim_numbers);
}

// Return an ShapeUtil::IndexIterationSpace that iterates over the update
// window dimensions while keeping the rest of the update dimensions clamped
// to 0.
ShapeUtil::IndexIterationSpace IterationSpaceForUpdateWindowIndices(
    absl::Span<const int64_t> updates_dims,
    const ScatterDimensionNumbers& dim_numbers) {
  return GetIterationSpaceImpl</*for_update_window_indices=*/true>(updates_dims,
                                                                   dim_numbers);
}

// This functor computes the contribution of scatter_indices to an input index
// corresponding to an update index.  That is, given an update index I, it
// picks out the scatter indices in I and uses them to look up a scatter
// index, S, from the scatter indices tensor, and expands S into the input
// space according to scatter_dims_to_operand_dims.
//
// This is similar to the class HloEvaluator::OutputGatherIndexToInputIndex
// that does the corresponding function for Gather.
class UpdateScatterIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit UpdateScatterIndexToInputIndex(
      const ScatterDimensionNumbers& dim_numbers, int64_t input_rank,
      int64_t updates_rank, const BorrowingLiteral& scatter_indices)
      : dim_numbers_(dim_numbers), scatter_indices_(scatter_indices) {
    for (int64_t i = 0; i < updates_rank; i++) {
      update_dim_is_scatter_dims_.push_back(
          !absl::c_binary_search(dim_numbers_.update_window_dims(), i));
    }

    for (int64_t i = 0; i < input_rank; i++) {
      int64_t index_of_input_dim_in_index_vector =
          FindIndex(dim_numbers_.scatter_dims_to_operand_dims(), i);
      if (index_of_input_dim_in_index_vector ==
          dim_numbers_.scatter_dims_to_operand_dims_size()) {
        input_dim_value_to_index_vector_.push_back(-1);
      } else {
        input_dim_value_to_index_vector_.push_back(
            index_of_input_dim_in_index_vector);
      }
    }

    index_vector_index_.resize(scatter_indices_.shape().dimensions_size());
    input_index_.resize(input_rank);
    int64_t index_vector_size =
        scatter_indices_.shape().dimensions(dim_numbers_.index_vector_dim());
    index_vector_.resize(index_vector_size);
  }

  // Returns the contribution of scatter_indices to the input index
  // corresponding to update_index.  See scatter_inner_loop_body.
  //
  // This is conceptually  a stateless transformation from update_index to the
  // scatter input index, but:
  //
  //  - Instead of allocating memory to represent the scatter input index on
  //    every invocation we reuse the same storage for the result
  //    (input_index_), mutating it in place.
  //  - Instead of allocating buffers for temporary values like
  //    index_vector_index_ and index_vector on every invocation, we reuse the
  //    same storage for all invocations.
  //
  // This returns a Span into memory owned by the class.
  absl::Span<const int64_t> operator()(absl::Span<const int64_t> update_index) {
    PropagateUpdateIndexScatterDimsToIndexVectorIndex(update_index);
    FetchIndexVector();
    PropagateIndexVectorToInputIndex();
    return absl::Span<const int64_t>(input_index_);
  }

 private:
  // Propagates the scatter index dimensions from the update index into
  // index_vector_index_ by mutating index_vector_index_ in place.  Does not
  // update the dim_numbers.index_vector_dim() dimension -- that's the
  // dimension we iterate over in FetchIndexVector.
  void PropagateUpdateIndexScatterDimsToIndexVectorIndex(
      absl::Span<const int64_t> update_index) {
    int64_t index_vector_index_i = 0;
    for (int64_t i = 0, e = update_index.size(); i < e; i++) {
      if (!update_dim_is_scatter_dims_[i]) {
        continue;
      }

      if (index_vector_index_i == dim_numbers_.index_vector_dim()) {
        index_vector_index_i++;
      }

      index_vector_index_[index_vector_index_i++] = update_index[i];
    }
  }

  // Populates index_vector_ by iterating over scatter_indices_ according to
  // index_vector_index_.
  void FetchIndexVector() {
    int64_t index_vector_dim = dim_numbers_.index_vector_dim();
    for (int64_t i = 0, e = index_vector_.size(); i < e; i++) {
      index_vector_index_[index_vector_dim] = i;
      index_vector_[i] =
          *scatter_indices_.GetIntegralAsS64(index_vector_index_);
    }
  }

  // Populates input_index_.
  void PropagateIndexVectorToInputIndex() {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_index_vector_[i] != -1) {
        input_index_[i] = index_vector_[input_dim_value_to_index_vector_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the index vector.  See
  // PropagateIndexVectorToInputIndex.
  std::vector<int64_t> input_dim_value_to_index_vector_;

  // update_dim_is_scatter_dims_[i] is true iff the update index i is a
  // scatter dimension.
  std::vector<bool> update_dim_is_scatter_dims_;

  // The buffer into which we construct an index into scatter_indices_ to
  // fetch the index vector.
  std::vector<int64_t> index_vector_index_;

  // The index vector fetched from scatter_indices_.
  std::vector<int64_t> index_vector_;

  // The result computed by this functor.  operator() returns a Span
  // into this vector.
  std::vector<int64_t> input_index_;

  const ScatterDimensionNumbers& dim_numbers_;
  const BorrowingLiteral& scatter_indices_;
};

// This functor computes the contribution of the window indices in an update
// index to an input index.  That is, given an update index I it picks out the
// update window indices in I and expands it into a window index into the
// input shape.
//
// This is similar to the class HloEvaluator::OutputWindowIndexToInputIndex
// that does the corresponding function for Gather.
class UpdateWindowIndexToInputIndex {
 public:
  // The constructor does some setup work that is amortized across all
  // iterations.
  explicit UpdateWindowIndexToInputIndex(
      const ScatterDimensionNumbers& dim_numbers, int64_t input_rank,
      int64_t update_rank) {
    std::vector<int64_t> window_index_to_update_index;
    int64_t update_index_count = 0;
    for (int64_t i = 0; i < update_rank; i++) {
      if (absl::c_binary_search(dim_numbers.update_window_dims(), i)) {
        window_index_to_update_index.push_back(update_index_count++);
      } else {
        update_index_count++;
      }
    }

    int64_t window_dim_count = 0;
    for (int64_t i = 0; i < input_rank; i++) {
      if (absl::c_binary_search(dim_numbers.inserted_window_dims(), i)) {
        input_dim_value_to_update_index_.push_back(-1);
      } else {
        input_dim_value_to_update_index_.push_back(
            window_index_to_update_index[window_dim_count++]);
      }
    }

    input_index_.resize(input_rank);
  }

  // Returns the contribution of the window indices to the input index
  // corresponding to update_index.  See scatter_inner_loop_body.
  //
  // This is conceptually a stateless transformation from update_index to the
  // window input index, but instead of allocating memory to represent the
  // scatter input index on every invocation we reuse the same storage for the
  // result (input_index_), mutating it in place.
  //
  // This returns a Span into memory owned by the class.
  absl::Span<const int64_t> operator()(absl::Span<const int64_t> update_index) {
    PropagateUpdateIndexWindowDimsToInputIndex(update_index);
    return absl::Span<const int64_t>(input_index_);
  }

  // Returns for a given 'input_dim' the corresponding update dimension index,
  // or -1 if 'input_dim' is an elided window dimension.
  int64_t input_dim_value_to_update_index(int64_t input_dim) {
    return input_dim_value_to_update_index_[input_dim];
  }

 private:
  // Propagates window dimensions from the update index to input_index_ by
  // mutating input_index_ in place.
  void PropagateUpdateIndexWindowDimsToInputIndex(
      absl::Span<const int64_t> update_index) {
    for (int64_t i = 0, e = input_index_.size(); i < e; i++) {
      if (input_dim_value_to_update_index_[i] != -1) {
        input_index_[i] = update_index[input_dim_value_to_update_index_[i]];
      }

      // If input_dim_value_to_index_vector_[i] == -1 then input_index_[i]
      // remains 0, as set by the constructor.
    }
  }

  // input_dim_value_to_index_vector_[i] tells us how to compute dimension i
  // of the input index from the update index. See
  // PropagateUpdateIndexWindowDimsToInputIndex.
  std::vector<int64_t> input_dim_value_to_update_index_;

  // The result computed by this functor.  operator() returns a Span
  // into this vector.
  std::vector<int64_t> input_index_;
};

}  // namespace

tsl::AsyncValueRef<ScatterThunk::ExecuteEvent> ScatterThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  VLOG(3) << absl::StreamFormat(
      "Scatter %d updates; update_window_dims=[%s] inserted_window_dims=[%s] "
      "scatter_dims_to_operand_dims=[%s] index_vector_dim=%d "
      "input_batching_dims=[%s] scatter_indices_batching_dims=[%s]",
      operands_.size(), absl::StrJoin(dim_numbers_.update_window_dims(), ","),
      absl::StrJoin(dim_numbers_.inserted_window_dims(), ","),
      absl::StrJoin(dim_numbers_.scatter_dims_to_operand_dims(), ","),
      dim_numbers_.index_vector_dim(),
      absl::StrJoin(dim_numbers_.input_batching_dims(), ","),
      absl::StrJoin(dim_numbers_.scatter_indices_batching_dims(), ","));

  size_t num_operands = operands_.size();
  size_t num_updates = updates_.size();
  size_t num_results = results_.size();

  absl::InlinedVector<se::DeviceMemoryBase, 1> operands_data;
  absl::InlinedVector<BorrowingLiteral, 1> operands_literals;
  operands_data.reserve(num_operands);
  operands_literals.reserve(num_operands);

  absl::InlinedVector<se::DeviceMemoryBase, 1> updates_data;
  absl::InlinedVector<BorrowingLiteral, 1> updates_literals;
  updates_data.reserve(num_updates);
  updates_literals.reserve(num_updates);

  absl::InlinedVector<se::DeviceMemoryBase, 1> results_data;
  absl::InlinedVector<MutableBorrowingLiteral, 1> results_literals;
  results_data.reserve(num_results);
  results_literals.reserve(num_results);

  // Resolve buffer for scatter indices.
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase scatter_indices_data,
      params.buffer_allocations->GetDeviceAddress(scatter_indices_.slice));
  BorrowingLiteral scatter_indices_literal(
      static_cast<const char*>(scatter_indices_data.opaque()),
      scatter_indices_.shape);

  VLOG(3) << absl::StreamFormat(
      "  scatter indices: %s in slice %s (%p)",
      scatter_indices_.shape.ToString(/*print_layout=*/true),
      scatter_indices_.slice.ToString(), scatter_indices_data.opaque());

  // Resolve buffers for operands.
  for (const Operand& operand : operands_) {
    size_t idx = operands_data.size();

    TF_ASSIGN_OR_RETURN(
        operands_data.emplace_back(),
        params.buffer_allocations->GetDeviceAddress(operand.slice));
    operands_literals.emplace_back(
        static_cast<const char*>(operands_data.back().opaque()), operand.shape);

    // Annotate memory that might have been initialized by jit-compiled code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(operands_data.back().opaque(),
                                        operands_data.back().size());

    VLOG(3) << absl::StreamFormat("  operand #%d: %s in slice %s (%p)", idx,
                                  operand.shape.ToString(/*print_layout=*/true),
                                  operand.slice.ToString(),
                                  operands_data.back().opaque());
  }

  // Resolve buffers for updates.
  for (const Update& update : updates_) {
    size_t idx = updates_data.size();

    TF_ASSIGN_OR_RETURN(
        updates_data.emplace_back(),
        params.buffer_allocations->GetDeviceAddress(update.slice));
    updates_literals.emplace_back(
        static_cast<const char*>(updates_data.back().opaque()), update.shape);

    // Annotate memory that might have been initialized by jit-compiled code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(updates_data.back().opaque(),
                                        updates_data.back().size());

    VLOG(3) << absl::StreamFormat("  update #%d: %s in slice %s (%p)", idx,
                                  update.shape.ToString(/*print_layout=*/true),
                                  update.slice.ToString(),
                                  updates_data.back().opaque());
  }

  // Resolve buffers for results.
  for (const Result& result : results_) {
    size_t idx = results_data.size();

    TF_ASSIGN_OR_RETURN(
        results_data.emplace_back(),
        params.buffer_allocations->GetDeviceAddress(result.slice));
    results_literals.emplace_back(
        static_cast<char*>(results_data.back().opaque()), result.shape);

    // Annotate memory that might have been initialized by jit-compiled code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(results_data.back().opaque(),
                                        results_data.back().size());

    VLOG(3) << absl::StreamFormat("  result #%d: %s in slice %s (%p)", idx,
                                  result.shape.ToString(/*print_layout=*/true),
                                  result.slice.ToString(),
                                  results_data.back().opaque());
  }

  Functor* functor = functor_ptr_.load();

  // Because thunks are owned by a parent CpuExecutable, we can safely assume
  // that scatter functor pointer will not change after we find it the first
  // time, and we can create a functor adaptor to a Scatter function.
  if (ABSL_PREDICT_FALSE(functor == nullptr)) {
    TF_ASSIGN_OR_RETURN(FunctionRegistry::Scatter scatter,
                        params.function_registry->FindScatter(functor_name_));

    absl::MutexLock lock(&mutex_);
    functor_ = [scatter](void* result, const void** args) {
      scatter(result, nullptr, args, nullptr, nullptr, nullptr);
    };
    functor_ptr_.store(functor = &*functor_);
  }

  absl::Span<const int64_t> operand_dims = operands_[0].shape.dimensions();
  absl::Span<const int64_t> updates_dims = updates_[0].shape.dimensions();

  ShapeUtil::IndexIterationSpace scatter_indices_iteration_space =
      IterationSpaceForUpdateScatterIndices(updates_dims, dim_numbers_);
  ShapeUtil::IndexIterationSpace window_indices_iteration_space =
      IterationSpaceForUpdateWindowIndices(updates_dims, dim_numbers_);

  std::vector<int64_t> input_index(operand_dims.size());
  std::vector<int64_t> update_index(updates_dims.size());

  UpdateScatterIndexToInputIndex update_scatter_index_to_input_index(
      dim_numbers_, operand_dims.size(), updates_dims.size(),
      scatter_indices_literal);
  UpdateWindowIndexToInputIndex update_window_index_to_input_index(
      dim_numbers_, operand_dims.size(), updates_dims.size());

  auto scatter_inner_loop_body =
      [&](absl::Span<const int64_t> update_window_index,
          absl::Span<const int64_t> input_scatter_index,
          absl::Span<const int64_t> update_scatter_index)
      -> absl::StatusOr<bool> {
    absl::Span<const int64_t> input_window_index =
        update_window_index_to_input_index(update_window_index);
    for (int i = 0, e = update_index.size(); i < e; i++) {
      update_index[i] = update_scatter_index[i] + update_window_index[i];
      DCHECK_LT(update_index[i], updates_dims[i]);
    }
    for (int i = 0, e = input_scatter_index.size(); i < e; i++) {
      int64_t update_dim =
          update_window_index_to_input_index.input_dim_value_to_update_index(i);
      // If 'update_dim' is -1, it means 'i' is an elided window dim. This
      // means we set the iteration index to 0, so for the purpose of the
      // following calculations we can consider the update dimension size to
      // be 1.
      int64_t update_dim_size = update_dim == -1 ? 1 : updates_dims[update_dim];
      // If any part of the update region is out-of-bounds, then do not
      // perform any update on the input.
      if ((input_scatter_index[i] < 0) ||
          (input_scatter_index[i] > operand_dims[i] - update_dim_size)) {
        return true;
      }
    }
    for (int i = 0, e = input_index.size(); i < e; i++) {
      input_index[i] = input_scatter_index[i] + input_window_index[i];
    }

    absl::InlinedVector<const void*, 2> to_apply_args;
    to_apply_args.reserve(operands_.size() + updates_.size());

    absl::InlinedVector<void*, 1> to_apply_results;
    to_apply_args.reserve(operands_.size());

    for (int32_t i = 0, n = operands_.size(); i < n; ++i) {
      to_apply_args.push_back(primitive_util::PrimitiveTypeSwitch<const void*>(
          [&](auto tag) -> const void* {
            if constexpr (primitive_util::IsArrayType(tag)) {
              using NativeT =
                  typename primitive_util::PrimitiveTypeToNative<tag>::type;
              return &operands_literals[i].data<NativeT>()
                          [IndexUtil::MultidimensionalIndexToLinearIndex(
                              operands_literals[i].shape(), input_index)];
            }
            LOG(FATAL) << "Unsupported primitive type: " << tag;
            return nullptr;
          },
          operands_[i].shape.element_type()));
    }

    for (int32_t i = 0, n = operands_.size(); i < n; ++i) {
      to_apply_args.push_back(primitive_util::PrimitiveTypeSwitch<const void*>(
          [&](auto tag) -> const void* {
            if constexpr (primitive_util::IsArrayType(tag)) {
              using NativeT =
                  typename primitive_util::PrimitiveTypeToNative<tag>::type;
              return &updates_literals[i].data<NativeT>()
                          [IndexUtil::MultidimensionalIndexToLinearIndex(
                              updates_literals[i].shape(), update_index)];
            }
            LOG(FATAL) << "Unsupported primitive type: " << tag;
            return nullptr;
          },
          updates_[i].shape.element_type()));
    }

    for (int32_t i = 0, n = operands_.size(); i < n; ++i) {
      to_apply_results.push_back(primitive_util::PrimitiveTypeSwitch<void*>(
          [&](auto tag) -> void* {
            if constexpr (primitive_util::IsArrayType(tag)) {
              using NativeT =
                  typename primitive_util::PrimitiveTypeToNative<tag>::type;
              return &results_literals[i].data<NativeT>()
                          [IndexUtil::MultidimensionalIndexToLinearIndex(
                              results_literals[i].shape(), input_index)];
            }
            LOG(FATAL) << "Unsupported primitive type: " << tag;
            return nullptr;
          },
          results_[i].shape.element_type()));
    }

    if (results_.size() == 1) {
      (*functor)(to_apply_results[0], to_apply_args.data());
    } else {
      (*functor)(to_apply_results.data(), to_apply_args.data());
    }

    return true;
  };

  auto scatter_outer_loop_body =
      [&](absl::Span<const int64_t> update_scatter_index)
      -> absl::StatusOr<bool> {
    absl::Span<const int64_t> input_scatter_index =
        update_scatter_index_to_input_index(update_scatter_index);
    TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
        updates_[0].shape, window_indices_iteration_space,
        [&](absl::Span<const int64_t> update_window_index) {
          return scatter_inner_loop_body(
              update_window_index, input_scatter_index, update_scatter_index);
        }));
    return true;
  };

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachIndexWithStatus(
      updates_[0].shape, scatter_indices_iteration_space,
      scatter_outer_loop_body));

  return OkExecuteEvent();
}

ScatterThunk::BufferUses ScatterThunk::buffer_uses() const {
  BufferUses buffer_uses;
  for (const Operand& operand : operands_) {
    buffer_uses.push_back(BufferUse::Read(operand.slice));
  }
  buffer_uses.push_back(BufferUse::Read(scatter_indices_.slice));
  for (const Update& update : updates_) {
    buffer_uses.push_back(BufferUse::Read(update.slice));
  }
  for (const Result& result : results_) {
    buffer_uses.push_back(BufferUse::Write(result.slice));
  }
  return buffer_uses;
}

}  // namespace xla::cpu
