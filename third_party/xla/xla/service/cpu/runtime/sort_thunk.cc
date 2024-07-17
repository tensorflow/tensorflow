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

#include "xla/service/cpu/runtime/sort_thunk.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

static absl::Status VerifySortInputs(absl::Span<const SortThunk::Input> inputs,
                                     int64_t dimension) {
  // We should have at least one input buffer.
  if (inputs.empty()) {
    return Internal("Inputs must not be empty");
  }

  // All inputs must have the same shape (ignoring element type) and layout.
  auto equal = Shape::Equal().IgnoreElementType();
  const Shape& shape = inputs[0].shape;

  for (const SortThunk::Input& input : inputs) {
    if (!equal(shape, input.shape)) {
      return Internal("Inputs must have the same shape");
    }
  }

  // Check that sort dimension is valid.
  int64_t sort_dimension =
      dimension >= 0 ? dimension : shape.rank() + dimension;
  if (shape.rank() <= sort_dimension) {
    return Internal(
        "Shape of dimensions [%s] can't be sorted along dimension %d",
        absl::StrJoin(shape.dimensions(), ","), dimension);
  }

  // We support only monotonic layouts with dim0 major.
  if (!LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return Internal("Unsupported sort input layout %s",
                    shape.ToString(/*print_layout=*/true));
  }

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, LessThan less_than) {
  TF_RETURN_IF_ERROR(VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(less_than)));
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, std::string comparator_name) {
  TF_RETURN_IF_ERROR(VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(comparator_name)));
}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable, LessThan less_than)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      less_than_(std::move(less_than)),
      less_than_ptr_(&*less_than_) {}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable,
                     std::string comparator_name)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      comparator_name_(std::move(comparator_name)),
      less_than_ptr_(nullptr) {}

namespace {

// We use a lot of template metaprogramming below to be able to construct
// iterators with statically known element sizes. We support a limited set of
// template instantiations that we need in practice.

// Forward declare reference type defined below.
template <typename... Ts>
struct Ref;

// Value type to store values loaded from the input buffers.
template <typename... Ts>
struct Value {
  Value(const Ref<Ts...>& ref);  // NOLINT

  template <size_t n>
  const void* compared_value() const {
    return &std::get<n>(value);
  }

  std::tuple<Ts...> value;
};

// Reference to values stored in the input buffers.
template <typename... Ts>
struct Ref {
  explicit Ref(std::tuple<Ts*...> ptr) : ptr(ptr) {}

  Ref& operator=(const Value<Ts...>& value);
  Ref& operator=(const Ref<Ts...>& other);

  template <size_t n>
  const void* compared_value() const {
    return std::get<n>(ptr);
  }

  std::tuple<Ts*...> ptr;
};

// Value to reference assignment.
template <typename... Ts, size_t... Is>
static void Assign(Ref<Ts...>& ref, const Value<Ts...>& value,
                   std::index_sequence<Is...>) {
  ((*std::get<Is>(ref.ptr) = std::get<Is>(value.value)), ...);
}

// Reference to reference assignment.
template <typename... Ts, size_t... Is>
static void Assign(Ref<Ts...>& ref, const Ref<Ts...>& other,
                   std::index_sequence<Is...>) {
  ((*std::get<Is>(ref.ptr) = *std::get<Is>(other.ptr)), ...);
}

template <typename... Ts>
Value<Ts...>::Value(const Ref<Ts...>& ref)
    : value(std::apply([](auto*... p) { return std::make_tuple(*p...); },
                       ref.ptr)) {}

template <typename... Ts>
Ref<Ts...>& Ref<Ts...>::operator=(const Value<Ts...>& value) {
  Assign(*this, value, std::make_index_sequence<sizeof...(Ts)>{});
  return *this;
}

template <typename... Ts>
Ref<Ts...>& Ref<Ts...>::operator=(const Ref<Ts...>& other) {
  Assign(*this, other, std::make_index_sequence<sizeof...(Ts)>{});
  return *this;
}

// Swap function required by `std::sort` and `std::stable_sort` implementations.
template <typename T0>
void swap(const Ref<T0>& lhs, const Ref<T0>& rhs) {
  std::swap(*std::get<0>(lhs.ptr), *std::get<0>(rhs.ptr));
}

template <typename T0, typename T1>
void swap(const Ref<T0, T1>& lhs, const Ref<T0, T1>& rhs) {
  std::swap(*std::get<0>(lhs.ptr), *std::get<0>(rhs.ptr));
  std::swap(*std::get<1>(lhs.ptr), *std::get<1>(rhs.ptr));
}

// Extracts pointers to compared elements and packs them in the layout expected
// by the comparator function.
template <typename Lhs, typename Rhs>
std::array<const void*, 2> ComparatorData1(const Lhs& lhs, const Rhs& rhs) {
  return {lhs.template compared_value<0>(), rhs.template compared_value<0>()};
}

template <typename Lhs, typename Rhs>
std::array<const void*, 4> ComparatorData2(const Lhs& lhs, const Rhs& rhs) {
  return {lhs.template compared_value<0>(), rhs.template compared_value<0>(),
          lhs.template compared_value<1>(), rhs.template compared_value<1>()};
}

// A pointer (tuple of pointers) to the input data.
template <typename... Ts>
struct Ptr {
  using difference_type = std::ptrdiff_t;

  Ptr() = default;
  explicit Ptr(Ts*... ptrs) : ptrs(ptrs...) {}
  explicit Ptr(std::tuple<Ts*...> ptrs) : ptrs(ptrs) {}

  Ref<Ts...> operator*() const { return Ref<Ts...>{ptrs}; }

  Ptr& operator+=(difference_type n) {
    ptrs = std::apply(
        [&](auto*... p) { return std::make_tuple<Ts*...>(p + n...); }, ptrs);
    return *this;
  }

  Ptr& operator-=(difference_type n) {
    ptrs = std::apply(
        [&](auto*... p) { return std::make_tuple<Ts*...>(p - n...); }, ptrs);
    return *this;
  }

  Ptr operator+(difference_type n) const {
    return Ptr{std::apply(
        [&](auto*... p) { return std::make_tuple<Ts*...>(p + n...); }, ptrs)};
  }

  Ptr operator-(difference_type n) const {
    return Ptr{std::apply(
        [&](auto*... p) { return std::make_tuple<Ts*...>(p - n...); }, ptrs)};
  }

  // In all comparison operators defined below we use only the ptr at index 0,
  // because we know that all pointers change together and this is an
  // implementation detail of sort iterator.

  difference_type operator-(const Ptr& rhs) const {
    return std::get<0>(ptrs) - std::get<0>(rhs.ptrs);
  }

  bool operator==(const Ptr& rhs) const {
    return std::get<0>(ptrs) == std::get<0>(rhs.ptrs);
  }
  bool operator!=(const Ptr& rhs) const {
    return std::get<0>(ptrs) != std::get<0>(rhs.ptrs);
  }
  bool operator>(const Ptr& rhs) const {
    return std::get<0>(ptrs) > std::get<0>(rhs.ptrs);
  }
  bool operator<(const Ptr& rhs) const {
    return std::get<0>(ptrs) < std::get<0>(rhs.ptrs);
  }
  bool operator>=(const Ptr& rhs) const {
    return std::get<0>(ptrs) >= std::get<0>(rhs.ptrs);
  }
  bool operator<=(const Ptr& rhs) const {
    return std::get<0>(ptrs) <= std::get<0>(rhs.ptrs);
  }

  std::tuple<Ts*...> ptrs;
};

// We rely on `std::sort` and `std::stable_sort` to sort the raw data. We sort
// multiple input buffers together using the same comparator function, so we
// need to provide a custom iterator that can access the data of all input
// buffers at the same time and swap elements in them.
template <typename... Ts>
class SortIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  using value_type = Value<Ts...>;
  using reference = Ref<Ts...>;
  using pointer = Ptr<Ts...>;

  SortIterator() = default;
  SortIterator(pointer ptr, difference_type stride)
      : ptr_(ptr), stride_(stride) {}

  SortIterator(const SortIterator& other) = default;
  SortIterator& operator=(const SortIterator& other) = default;
  SortIterator(SortIterator&& other) = default;
  SortIterator& operator=(SortIterator&& other) = default;

  reference operator*() const { return *ptr_; }

  difference_type operator-(const SortIterator& rhs) const {
    return (ptr_ - rhs.ptr_) / stride_;
  }

  SortIterator& operator+=(difference_type n) {
    ptr_ += n * stride_;
    return *this;
  }

  SortIterator& operator-=(difference_type n) {
    ptr_ -= n * stride_;
    return *this;
  }

  SortIterator& operator++() {
    ptr_ += stride_;
    return *this;
  }

  SortIterator& operator--() {
    ptr_ -= stride_;
    return *this;
  }

  SortIterator operator+(difference_type n) const {
    return SortIterator(ptr_ + n * stride_, stride_);
  }

  SortIterator operator-(difference_type n) const {
    return SortIterator(ptr_ - n * stride_, stride_);
  }

  bool operator==(const SortIterator& rhs) const { return ptr_ == rhs.ptr_; }
  bool operator!=(const SortIterator& rhs) const { return ptr_ != rhs.ptr_; }
  bool operator>(const SortIterator& rhs) const { return ptr_ > rhs.ptr_; }
  bool operator<(const SortIterator& rhs) const { return ptr_ < rhs.ptr_; }
  bool operator>=(const SortIterator& rhs) const { return ptr_ >= rhs.ptr_; }
  bool operator<=(const SortIterator& rhs) const { return ptr_ <= rhs.ptr_; }

 private:
  pointer ptr_;
  difference_type stride_ = 1;
};

struct SortDims {
  int64_t outer_dim_size;
  int64_t sort_dim_size;
  int64_t inner_dim_size;
  int64_t num_iterations;
};

}  // namespace

// Conceptually we have a 3-dimensional shape:
//
//   [outer_dim_size, sort_dim_size, inner_dim_size]
//
// We sort `outer_dim_size * inner_dim_size` vectors of length
// `sort_dim_size`, by iterating over `data` memory and calling `std::sort`
// (or `std::stable_sort`) on each (strided) slice of the buffer.
static SortDims GetSortDims(absl::Span<const int64_t> dimensions,
                            int64_t dimension) {
  int64_t sort_dimension =
      dimension >= 0 ? dimension : dimensions.size() + dimension;

  auto product = [](absl::Span<const int64_t> dims) {
    return absl::c_accumulate(dims, int64_t{1}, std::multiplies<>());
  };

  int64_t outer_dim_size = product(dimensions.subspan(0, dimension));
  int64_t sort_dim_size = dimensions[sort_dimension];
  int64_t inner_dim_size = product(dimensions.subspan(dimension + 1));
  int64_t num_iterations = outer_dim_size * inner_dim_size;

  return SortDims{outer_dim_size, sort_dim_size, inner_dim_size,
                  num_iterations};
}

// Sorts one input buffer of type `T0` inplace.
template <typename T0>
static void SortInplace(const SortDims& sort_dims, int64_t offset,
                        absl::Span<se::DeviceMemoryBase> data, bool is_stable,
                        SortThunk::LessThan* less_than) {
  T0* base0 = reinterpret_cast<T0*>(data[0].opaque());

  auto compare = [&](const auto& a, const auto& b) {
    auto data = ComparatorData1(a, b);
    return (*less_than)(data.data());
  };

  SortIterator<T0> begin(Ptr<T0>(base0 + offset),
                         /*stride=*/sort_dims.inner_dim_size);
  if (is_stable) {
    std::stable_sort(begin, begin + sort_dims.sort_dim_size, compare);
  } else {
    std::sort(begin, begin + sort_dims.sort_dim_size, compare);
  }
}

// Sorts two input buffers of type `T0` and `T1` inplace.
template <typename T0, typename T1>
static void SortInplace(const SortDims& sort_dims, int64_t offset,
                        absl::Span<se::DeviceMemoryBase> data, bool is_stable,
                        SortThunk::LessThan* less_than) {
  T0* base0 = reinterpret_cast<T0*>(data[0].opaque());
  T1* base1 = reinterpret_cast<T1*>(data[1].opaque());

  auto compare = [&](const auto& a, const auto& b) {
    auto data = ComparatorData2(a, b);
    return (*less_than)(data.data());
  };

  SortIterator<T0, T1> begin(Ptr<T0, T1>(base0 + offset, base1 + offset),
                             /*stride=*/sort_dims.inner_dim_size);
  if (is_stable) {
    std::stable_sort(begin, begin + sort_dims.sort_dim_size, compare);
  } else {
    std::sort(begin, begin + sort_dims.sort_dim_size, compare);
  }
}

// Sorts `data` of the given `shape` along the `dimension` inplace.
static absl::Status SortInplace(absl::Span<se::DeviceMemoryBase> data,
                                absl::Span<const Shape> shapes,
                                int64_t dimension, bool is_stable,
                                SortThunk::LessThan* less_than) {
  // All inputs have the same dimensions and layout, so we can use the first
  // shape to get the sort dimensions.
  SortDims sort_dims = GetSortDims(shapes[0].dimensions(), dimension);

  // Type tags for specializing the `sort` functor. Instead of specializing for
  // each individual primitive type, we use a byte array of correct size to
  // avoid the code bloat, as we use external comparator function anyway and
  // don't compare the values directly.
  using _4_bytes = std::array<std::byte, 4>;

  // Collect byte sizes of element types of all inputs.
  absl::InlinedVector<size_t, 2> byte_sizes;
  byte_sizes.reserve(data.size());
  for (const Shape& shape : shapes) {
    byte_sizes.push_back(primitive_util::ByteWidth(shape.element_type()));
  }

  auto is_byte_sizes = [&](auto... sizes) {
    return absl::c_equal(byte_sizes, absl::InlinedVector<size_t, 2>{
                                         static_cast<size_t>(sizes)...});
  };

  // Iterate over all the 1-dimensional slices of the buffers and sort them.
  for (int64_t i = 0; i < sort_dims.num_iterations; ++i) {
    int64_t inner_idx = i % sort_dims.inner_dim_size;
    int64_t offset = inner_idx + (i - inner_idx) * sort_dims.sort_dim_size;

    if (is_byte_sizes(4)) {
      SortInplace<_4_bytes>(sort_dims, offset, data, is_stable, less_than);
    } else if (is_byte_sizes(4, 4)) {
      SortInplace<_4_bytes, _4_bytes>(sort_dims, offset, data, is_stable,
                                      less_than);
    } else {
      return Internal("Unsupported sort element byte widths [%s]",
                      absl::StrJoin(byte_sizes, ","));
    }
  }

  return absl::OkStatus();
}

tsl::AsyncValueRef<SortThunk::ExecuteEvent> SortThunk::Execute(
    const ExecuteParams& params) {
  VLOG(3) << absl::StreamFormat(
      "Sort %d inputs along dimension %d (is_stable=%v)", inputs_.size(),
      dimension_, is_stable_);

  absl::InlinedVector<se::DeviceMemoryBase, 8> data;
  data.reserve(inputs_.size());

  absl::InlinedVector<Shape, 8> shapes;
  shapes.reserve(inputs_.size());

  for (const Input& input : inputs_) {
    size_t idx = data.size();
    TF_ASSIGN_OR_RETURN(
        data.emplace_back(),
        params.buffer_allocations->GetDeviceAddress(input.slice));
    shapes.push_back(input.shape);

    VLOG(3) << absl::StreamFormat("  sort input #%d: %s in slice %s (%p)", idx,
                                  input.shape.ToString(),
                                  input.slice.ToString(), data.back().opaque());
  }

  LessThan* less_than = less_than_ptr_.load();

  // Because thunks are owned by a parent CpuExecutable, we can safely assume
  // that comparator pointer will not change after we find it the first time,
  // and we can create a comparator adaptor to a LessThan function.
  if (ABSL_PREDICT_FALSE(less_than == nullptr)) {
    TF_ASSIGN_OR_RETURN(
        FunctionRegistry::Comparator comparator,
        params.function_registry->FindComparator(comparator_name_));

    absl::MutexLock lock(&mutex_);
    less_than_ = [comparator](const void** data) {
      bool result;
      comparator(&result, nullptr, data, nullptr, nullptr, nullptr);
      return result;
    };
    less_than_ptr_.store(less_than = &*less_than_);
  }

  TF_RETURN_IF_ERROR(SortInplace(absl::MakeSpan(data), shapes, dimension_,
                                 is_stable_, less_than));

  return OkExecuteEvent();
}

SortThunk::BufferUses SortThunk::buffer_uses() const {
  BufferUses buffer_uses;
  buffer_uses.reserve(inputs_.size());
  for (const Input& input : inputs_) {
    buffer_uses.emplace_back(BufferUse::Write(input.slice));
  }
  return buffer_uses;
}

}  // namespace xla::cpu
