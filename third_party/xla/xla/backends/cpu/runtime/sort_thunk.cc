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

#include "xla/backends/cpu/runtime/sort_thunk.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/sort_lib.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

#define EIGEN_USE_THREADS
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla::cpu {

// Conceptually we have a 3-dimensional shape:
//
//   [outer_dim_size, sort_dim_size, inner_dim_size]
//
// We sort `outer_dim_size * inner_dim_size` vectors of length
// `sort_dim_size`, by iterating over `data` memory and calling `std::sort`
// (or `std::stable_sort`) on each (strided) slice of the buffer.
static SortThunk::SortDims GetSortDims(const Shape& shape, int64_t dimension) {
  int64_t sort_dimension =
      dimension >= 0 ? dimension : shape.dimensions().size() + dimension;

  // We need to normalize shape + layout into a descending layout, so that we
  // can compute access strides according to the physical layout.
  Shape physical_shape =
      ShapeUtil::MakeShapeWithDescendingLayoutAndSamePhysicalLayout(shape);

  // Map `sort_dimension` from logical to physical.
  auto logical_to_physical = LayoutUtil::MakeLogicalToPhysical(shape.layout());
  sort_dimension = logical_to_physical[sort_dimension];

  auto product = [](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, int64_t{1}, std::multiplies<>());
  };

  // Use physical dimensions to compute access strides.
  absl::Span<const int64_t> dimensions = physical_shape.dimensions();

  int64_t outer_dim_size = product(dimensions.subspan(0, sort_dimension));
  int64_t sort_dim_size = dimensions[sort_dimension];
  int64_t inner_dim_size = product(dimensions.subspan(sort_dimension + 1));

  return SortThunk::SortDims{outer_dim_size, sort_dim_size, inner_dim_size};
}

static absl::StatusOr<SortThunk::SortDims> VerifySortInputs(
    absl::Span<const SortThunk::Input> inputs, int64_t dimension) {
  // We should have at least one input buffer.
  if (inputs.empty()) {
    return Internal("Inputs must not be empty");
  }

  // All inputs must have the same shape and layout (ignoring element type).
  auto equal = Shape::Equal().IgnoreElementType();
  const Shape& shape = inputs[0].shape;

  for (const SortThunk::Input& input : inputs) {
    if (!equal(shape, input.shape)) {
      return Internal("Inputs must have the same shape");
    }
  }

  // Check that sort dimension is valid.
  int64_t sort_dimension =
      dimension >= 0 ? dimension : shape.dimensions().size() + dimension;
  if (shape.dimensions().size() <= sort_dimension) {
    return Internal(
        "Shape of dimensions [%s] can't be sorted along dimension %d",
        absl::StrJoin(shape.dimensions(), ","), dimension);
  }

  return GetSortDims(inputs[0].shape, dimension);
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, LessThan less_than,
    std::optional<SortDirection> direction) {
  ABSL_ASSIGN_OR_RETURN(auto sort_dims, VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(less_than),
                                        sort_dims, direction));
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, std::string comparator_name,
    std::optional<SortDirection> direction) {
  ABSL_ASSIGN_OR_RETURN(auto sort_dims, VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(comparator_name),
                                        sort_dims, direction));
}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable, LessThan less_than,
                     SortDims sort_dims, std::optional<SortDirection> direction)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      sort_dims_(sort_dims),
      direction_(direction),
      less_than_(std::move(less_than)) {}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable,
                     std::string comparator_name, SortDims sort_dims,
                     std::optional<SortDirection> direction)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      sort_dims_(sort_dims),
      direction_(direction),
      comparator_name_(std::move(comparator_name)) {}

tsl::AsyncValueRef<SortThunk::ExecuteEvent> SortThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat(
      "Sort %d inputs along dimension %d (is_stable=%v)", inputs_.size(),
      dimension_, is_stable_);

  absl::InlinedVector<se::DeviceAddressBase, 8> data;
  data.reserve(inputs_.size());

  absl::InlinedVector<std::byte*, 8> raw_data;
  raw_data.reserve(inputs_.size());

  absl::InlinedVector<size_t, 8> primitive_sizes;
  primitive_sizes.reserve(inputs_.size());

  for (const Input& input : inputs_) {
    size_t idx = data.size();
    ABSL_ASSIGN_OR_RETURN(data.emplace_back(),
                     params.buffer_allocations->GetDeviceAddress(input.slice));
    raw_data.push_back(reinterpret_cast<std::byte*>(data.back().opaque()));
    primitive_sizes.push_back(
        primitive_util::ByteWidth(input.shape.element_type()));

    VLOG(3) << absl::StreamFormat("  sort input #%d: %s in slice %s (%p)", idx,
                                  input.shape.ToString(/*print_layout=*/true),
                                  input.slice.ToString(), data.back().opaque());
  }

  // Because thunks are owned by a parent CpuExecutable, we can safely assume
  // that comparator pointer will not change after we find it the first time,
  // and we can create a comparator adaptor to a LessThan function.
  absl::call_once(less_than_init_flag_, [&]() {
    if (less_than_.ok()) {
      // `less_than_` may already be initialized in the constructor.
      return;
    }
    absl::StatusOr<FunctionLibrary::Comparator*> comparator =
        params.function_library->ResolveFunction<FunctionLibrary::Comparator>(
            comparator_name_);

    if (ABSL_PREDICT_TRUE(comparator.ok())) {
      less_than_ = [comparator](const void** data) {
        bool result;
        (*comparator)(&result, nullptr, data, nullptr, nullptr, nullptr);
        ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(&result, sizeof(result));
        return result;
      };
    } else {
      less_than_ = std::move(comparator.status());
    }
  });

  ABSL_RETURN_IF_ERROR(less_than_.status());
  LessThan* less_than = &less_than_.value();

  int64_t num_slices = sort_dims_.outer_dim_size * sort_dims_.inner_dim_size;

  // Annotate memory that might have been initialized by jit-compiled code.
  for (int64_t i = 0; i < data.size(); ++i) {
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(
        data[i].opaque(),
        primitive_sizes[i] * sort_dims_.sort_dim_size * num_slices);
  }

  PrimitiveType first_element_type = inputs_[0].shape.element_type();
  auto sort_slice_range = [raw_data, primitive_sizes, first_element_type,
                           sort_dims = sort_dims_, is_stable = is_stable_,
                           less_than, direction = direction_](
                              int64_t start_slice, int64_t end_slice) {
    if (raw_data.size() == 1 && direction.has_value()) {
      primitive_util::ArrayTypeSwitch(
          [&](auto type) {
            if constexpr ((primitive_util::IsFloatingPointType(type) &&
                           primitive_util::BitWidth(type) >= 32) ||
                          (primitive_util::IsIntegralType(type) &&
                           primitive_util::BitWidth(type) >= 8)) {
              using T = primitive_util::NativeTypeOf<type>;
              internal::SortInplace<T>(sort_dims, start_slice, end_slice,
                                       reinterpret_cast<T*>(raw_data[0]),
                                       is_stable, *direction);
            } else {
              internal::SortInplace(sort_dims, start_slice, end_slice, raw_data,
                                    primitive_sizes, is_stable, less_than);
            }
          },
          first_element_type);
    } else {
      internal::SortInplace(sort_dims, start_slice, end_slice, raw_data,
                            primitive_sizes, is_stable, less_than);
    }
  };

  // Target ~32K elements per work chunk to balance thread scheduling overhead
  // with multi-core load balancing.
  static constexpr int64_t kTargetGrainElements = 32768;
  int64_t grain_slices = std::max<int64_t>(
      1, kTargetGrainElements / std::max<int64_t>(1, sort_dims_.sort_dim_size));

  // If work fits in a single chunk or no threadpool, sort sequentially on the
  // caller thread.
  if (params.intra_op_threadpool == nullptr || num_slices <= grain_slices) {
    sort_slice_range(0, num_slices);
    return OkExecuteEvent();
  }

  // Use intra-op thread pool to sort slices in parallel.
  int64_t num_tasks = CeilOfRatio(num_slices, grain_slices);

  tsl::CountDownAsyncValueRef<ExecuteEvent> event(num_tasks);

  for (int64_t t = 1; t < num_tasks; ++t) {
    int64_t start_slice = t * grain_slices;
    int64_t end_slice = std::min(num_slices, start_slice + grain_slices);
    params.intra_op_threadpool->getPool()->Schedule(
        [sort_slice_range, start_slice, end_slice, event]() mutable {
          sort_slice_range(start_slice, end_slice);
          event.CountDown();
        });
  }

  // Execute the first task in the caller thread.
  sort_slice_range(0, grain_slices);
  event.CountDown();

  return event.AsRef();
}

SortThunk::BufferUses SortThunk::buffer_uses() const {
  BufferUses buffer_uses;
  buffer_uses.reserve(inputs_.size());
  for (const Input& input : inputs_) {
    buffer_uses.emplace_back(BufferUse::Write(input.slice, input.shape));
  }
  return buffer_uses;
}

}  // namespace xla::cpu
