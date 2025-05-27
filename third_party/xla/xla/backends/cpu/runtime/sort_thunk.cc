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
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/base/call_once.h"
#include "absl/base/dynamic_annotations.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/backends/cpu/runtime/function_library.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::cpu {

static absl::Status VerifySortInputs(absl::Span<const SortThunk::Input> inputs,
                                     int64_t dimension) {
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

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, LessThan less_than,
    std::optional<SortDirection> direction) {
  TF_RETURN_IF_ERROR(VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(less_than),
                                        direction));
}

absl::StatusOr<std::unique_ptr<SortThunk>> SortThunk::Create(
    Info info, absl::Span<const Input> inputs, int64_t dimension,
    bool is_stable, std::string comparator_name,
    std::optional<SortDirection> direction) {
  TF_RETURN_IF_ERROR(VerifySortInputs(inputs, dimension));
  return absl::WrapUnique(new SortThunk(std::move(info), inputs, dimension,
                                        is_stable, std::move(comparator_name),
                                        direction));
}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable, LessThan less_than,
                     std::optional<SortDirection> direction)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      direction_(direction),
      less_than_(std::move(less_than)) {}

SortThunk::SortThunk(Info info, absl::Span<const Input> inputs,
                     int64_t dimension, bool is_stable,
                     std::string comparator_name,
                     std::optional<SortDirection> direction)
    : Thunk(Kind::kSort, std::move(info)),
      inputs_(inputs.begin(), inputs.end()),
      dimension_(dimension),
      is_stable_(is_stable),
      direction_(direction),
      comparator_name_(std::move(comparator_name)) {}

namespace {

// We use a lot of template metaprogramming below to be able to construct
// iterators with statically known number of compared elements. We support a
// limited set of template instantiations that we need in practice.

// The size of the largest element we support (std::complex<double>).
static constexpr size_t kMaxElementSize = 16;

// Type erased storage suitable for storing any primitive type.
using ValueStorage = std::array<std::byte, kMaxElementSize>;

// Pointers to the input arrays together with their primitive sizes.
template <size_t n>
class Inputs {
 public:
  Inputs(std::array<std::byte*, n> ptrs,
         std::array<size_t, n> primitive_sizes) {
    for (size_t i = 0; i < n; ++i) {
      ptrs_and_primitive_sizes_[i] = {ptrs[i], primitive_sizes[i]};
    }
  }

  // Accessing arrays with `operator[]` has zero overheads, so we don't need to
  // use pointers to data in contrast to `DInputs` below.

  std::byte* ptr(size_t i, size_t offset) const {
    DCHECK_LT(i, n) << "Input index out of bounds";
    auto& [ptr, primitive_size] = ptrs_and_primitive_sizes_[i];
    return ptr + offset * primitive_size;
  }

  size_t primitive_size(size_t i) const {
    return ptrs_and_primitive_sizes_[i].second;
  }

 private:
  // Pointers into the input buffers and each input's primitive size. Keep
  // pointers and primitives sizes next to each other to avoid cache misses
  // on a hot path.
  std::array<std::pair<std::byte*, size_t>, n> ptrs_and_primitive_sizes_;
};

class DInputs {
 public:
  DInputs(std::vector<std::byte*> ptrs, std::vector<size_t> primitive_sizes)
      : n_(ptrs.size()), ptrs_and_primitive_sizes_(ptrs.size()) {
    DCHECK_EQ(ptrs.size(), primitive_sizes.size());
    for (size_t i = 0; i < ptrs.size(); ++i) {
      ptrs_and_primitive_sizes_[i] = {ptrs[i], primitive_sizes[i]};
    }
  }

  size_t n() const { return n_; }

  // Accessing vectors with `operator[]` is significantly slower than using a
  // pointer to data because of libc++ hardening which checks for OOB access on
  // every call. We know that we are not going to access out of bounds, so we
  // use a pointer to data instead.

  std::byte* ptr(size_t i, size_t offset) const {
    DCHECK_LT(i, n_) << "Input index out of bounds";
    auto& [ptr, primitive_size] = ptrs_and_primitive_sizes_.data()[i];
    return ptr + offset * primitive_size;
  }

  size_t primitive_size(size_t i) const {
    return ptrs_and_primitive_sizes_.data()[i].second;
  }

 private:
  size_t n_;  // number of sorted inputs

  // Pointers into the input buffers and each input's primitive size. Keep
  // pointers and primitives sizes next to each other to avoid cache misses
  // on a hot path.
  std::vector<std::pair<std::byte*, size_t>> ptrs_and_primitive_sizes_;
};

// Forward declare reference type defined below.
template <size_t n>
struct Ref;
struct DRef;

// Value type to store values loaded from the input buffers.
template <size_t n>
struct Value {
  Value(const Ref<n>& ref);  // NOLINT

  void FillComparedValues(const void** __restrict compared_values) const;

  std::array<ValueStorage, n> values;
};

struct DValue {
  DValue(const DRef& ref);  // NOLINT

  void FillComparedValues(const void** __restrict compared_values) const;

  std::vector<ValueStorage> values;
};

// Reference to values stored in the input buffers.
template <size_t n>
struct Ref {
  Ref(const Inputs<n>* inputs, size_t offset)
      : inputs(inputs), offset(offset) {}

  Ref& operator=(const Value<n>& value);
  Ref& operator=(const Ref<n>& other);

  void FillComparedValues(const void** __restrict compared_values) const;

  std::byte* ptr(size_t i) const { return inputs->ptr(i, offset); }
  size_t primitive_size(size_t i) const { return inputs->primitive_size(i); }

  const Inputs<n>* inputs;
  size_t offset;
};

struct DRef {
  DRef(const DInputs* inputs, size_t offset) : inputs(inputs), offset(offset) {}

  DRef& operator=(const DValue& value);
  DRef& operator=(const DRef& other);

  void FillComparedValues(const void** __restrict compared_values) const;

  size_t n() const { return inputs->n(); }
  std::byte* ptr(size_t i) const { return inputs->ptr(i, offset); }
  size_t primitive_size(size_t i) const { return inputs->primitive_size(i); }

  const DInputs* inputs;
  size_t offset;
};

// We know that we can only copy up to 16 bytes for the largest element type
// and can specialize `std::memcpy` to allow LLVM to inline it with statically
// known sizes.
static ABSL_ATTRIBUTE_ALWAYS_INLINE void Memcpy(void* __restrict dest,
                                                const void* __restrict src,
                                                size_t n) {
  switch (n) {
    case 1:
      std::memcpy(dest, src, 1);
      break;
    case 2:
      std::memcpy(dest, src, 2);
      break;
    case 4:
      std::memcpy(dest, src, 4);
      break;
    case 8:
      std::memcpy(dest, src, 8);
      break;
    case 16:
      std::memcpy(dest, src, 16);
      break;
    default:
      LOG(FATAL) << "Unsupported memcpy size: " << n;
  }
}

// Specialize swap for statically known sizes to avoid going through the same
// switch statement multiple times.
static ABSL_ATTRIBUTE_ALWAYS_INLINE void Swap(void* __restrict a,
                                              void* __restrict b, size_t n) {
  std::array<std::byte, kMaxElementSize> tmp;
  switch (n) {
    case 1:
      std::memcpy(tmp.data(), a, 1);
      std::memcpy(a, b, 1);
      std::memcpy(b, tmp.data(), 1);
      break;
    case 2:
      std::memcpy(tmp.data(), a, 2);
      std::memcpy(a, b, 2);
      std::memcpy(b, tmp.data(), 2);
      break;
    case 4:
      std::memcpy(tmp.data(), a, 4);
      std::memcpy(a, b, 4);
      std::memcpy(b, tmp.data(), 4);
      break;
    case 8:
      std::memcpy(tmp.data(), a, 8);
      std::memcpy(a, b, 8);
      std::memcpy(b, tmp.data(), 8);
      break;
    case 16:
      std::memcpy(tmp.data(), a, 16);
      std::memcpy(a, b, 16);
      std::memcpy(b, tmp.data(), 16);
      break;
    default:
      LOG(FATAL) << "Unsupported swap size: " << n;
  }
}

template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE Value<n>::Value(const Ref<n>& ref) {
  for (size_t i = 0; i < n; ++i) {
    Memcpy(values[i].data(), ref.ptr(i), ref.primitive_size(i));
  }
}

template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE void Value<n>::FillComparedValues(
    const void** __restrict compared_values) const {
  for (const ValueStorage& value : values) {
    *compared_values = value.data();
    compared_values += 2;
  }
}

ABSL_ATTRIBUTE_ALWAYS_INLINE DValue::DValue(const DRef& ref) : values(ref.n()) {
  for (size_t i = 0, end = ref.n(); i < end; ++i) {
    Memcpy(values.data()[i].data(), ref.ptr(i), ref.primitive_size(i));
  }
}

ABSL_ATTRIBUTE_ALWAYS_INLINE void DValue::FillComparedValues(
    const void** __restrict compared_values) const {
#pragma unroll 8
  for (const ValueStorage& value : values) {
    *compared_values = value.data();
    compared_values += 2;
  }
}

template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE Ref<n>& Ref<n>::operator=(const Value<n>& value) {
  for (size_t i = 0; i < n; ++i) {
    Memcpy(ptr(i), value.values.data()[i].data(), primitive_size(i));
  }
  return *this;
}

template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE Ref<n>& Ref<n>::operator=(const Ref<n>& other) {
  for (size_t i = 0; i < n; ++i) {
    DCHECK_EQ(primitive_size(i), other.primitive_size(i));
    Memcpy(ptr(i), other.ptr(i), primitive_size(i));
  }
  return *this;
}

template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE void Ref<n>::FillComparedValues(
    const void** __restrict compared_values) const {
  for (size_t i = 0; i < n; ++i) {
    *compared_values = ptr(i);
    compared_values += 2;
  }
}

ABSL_ATTRIBUTE_ALWAYS_INLINE DRef& DRef::operator=(const DValue& value) {
  for (size_t i = 0, end = n(); i < end; ++i) {
    Memcpy(ptr(i), value.values.data()[i].data(), primitive_size(i));
  }
  return *this;
}

ABSL_ATTRIBUTE_ALWAYS_INLINE DRef& DRef::operator=(const DRef& other) {
  for (size_t i = 0, end = n(); i < end; ++i) {
    DCHECK_EQ(primitive_size(i), other.primitive_size(i));
    Memcpy(ptr(i), other.ptr(i), primitive_size(i));
  }
  return *this;
}

ABSL_ATTRIBUTE_ALWAYS_INLINE void DRef::FillComparedValues(
    const void** __restrict compared_values) const {
#pragma unroll 8
  for (size_t i = 0, end = n(); i < end; ++i) {
    *compared_values = ptr(i);
    compared_values += 2;
  }
}

// Swap function required by `std::sort` and `std::stable_sort` implementations.
template <size_t n>
ABSL_ATTRIBUTE_ALWAYS_INLINE void swap(const Ref<n>& lhs, const Ref<n>& rhs) {
  for (size_t i = 0; i < n; ++i) {
    DCHECK_EQ(lhs.primitive_size(i), rhs.primitive_size(i));
    size_t primitive_size = lhs.primitive_size(i);
    Swap(lhs.ptr(i), rhs.ptr(i), primitive_size);
  }
}

ABSL_ATTRIBUTE_ALWAYS_INLINE void swap(const DRef& lhs, const DRef& rhs) {
  for (size_t i = 0, end = lhs.n(); i < end; ++i) {
    DCHECK_EQ(lhs.primitive_size(i), rhs.primitive_size(i));
    size_t primitive_size = lhs.primitive_size(i);
    Swap(lhs.ptr(i), rhs.ptr(i), primitive_size);
  }
}

// An array of pointers to the input data.
template <size_t n>
struct Ptr {
  using difference_type = std::ptrdiff_t;

  Ptr() = default;

  explicit Ptr(const Inputs<n>* inputs, size_t offset = 0)
      : inputs(inputs), offset(offset) {}

  Ref<n> operator*() const { return Ref<n>{inputs, offset}; }

  Ptr& operator+=(difference_type diff) {
    offset += diff;
    return *this;
  }

  Ptr& operator-=(difference_type diff) {
    offset -= diff;
    return *this;
  }

  Ptr operator+(difference_type diff) const {
    return Ptr(inputs, offset + diff);
  }

  Ptr operator-(difference_type diff) const {
    return Ptr(inputs, offset - diff);
  }

  difference_type operator-(const Ptr& rhs) const {
    return offset - rhs.offset;
  }

  bool operator==(const Ptr& rhs) const { return offset == rhs.offset; }
  bool operator!=(const Ptr& rhs) const { return offset != rhs.offset; }
  bool operator>(const Ptr& rhs) const { return offset > rhs.offset; }
  bool operator<(const Ptr& rhs) const { return offset < rhs.offset; }
  bool operator>=(const Ptr& rhs) const { return offset >= rhs.offset; }
  bool operator<=(const Ptr& rhs) const { return offset <= rhs.offset; }

  const Inputs<n>* inputs;  // pointer to the input arrays
  size_t offset;            // offset into the inputs arrays
};

struct DPtr {
  using difference_type = std::ptrdiff_t;

  DPtr() = default;

  explicit DPtr(const DInputs* inputs, size_t offset = 0)
      : inputs(inputs), offset(offset) {}

  DRef operator*() const { return DRef{inputs, offset}; }

  DPtr& operator+=(difference_type diff) {
    offset += diff;
    return *this;
  }

  DPtr& operator-=(difference_type diff) {
    offset -= diff;
    return *this;
  }

  DPtr operator+(difference_type diff) const {
    return DPtr(inputs, offset + diff);
  }

  DPtr operator-(difference_type diff) const {
    return DPtr(inputs, offset - diff);
  }

  difference_type operator-(const DPtr& rhs) const {
    return offset - rhs.offset;
  }

  bool operator==(const DPtr& rhs) const { return offset == rhs.offset; }
  bool operator!=(const DPtr& rhs) const { return offset != rhs.offset; }
  bool operator>(const DPtr& rhs) const { return offset > rhs.offset; }
  bool operator<(const DPtr& rhs) const { return offset < rhs.offset; }
  bool operator>=(const DPtr& rhs) const { return offset >= rhs.offset; }
  bool operator<=(const DPtr& rhs) const { return offset <= rhs.offset; }

  const DInputs* inputs;  // pointer to the input arrays
  size_t offset;          // offset into the inputs arrays
};

// We rely on `std::sort` and `std::stable_sort` to sort the raw data. We sort
// multiple input buffers together using the same comparator function, so we
// need to provide a custom iterator that can access the data of all input
// buffers at the same time and swap elements in them.
template <class Value, class Ref, class Ptr>
class SortIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = std::ptrdiff_t;

  using value_type = Value;
  using reference = Ref;
  using pointer = Ptr;

  SortIterator() = default;
  SortIterator(pointer ptr, difference_type stride)
      : ptr_(std::move(ptr)), stride_(stride) {}

  SortIterator(const SortIterator& other) = default;
  SortIterator& operator=(const SortIterator& other) = default;
  SortIterator(SortIterator&& other) = default;
  SortIterator& operator=(SortIterator&& other) = default;

  reference operator*() const { return *ptr_; }
  reference operator[](difference_type diff) const { return *(*this + diff); }

  difference_type operator-(const SortIterator& rhs) const {
    return (ptr_ - rhs.ptr_) / stride_;
  }

  SortIterator& operator+=(difference_type diff) {
    ptr_ += diff * stride_;
    return *this;
  }

  SortIterator& operator-=(difference_type diff) {
    ptr_ -= diff * stride_;
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

  SortIterator operator+(difference_type diff) const {
    return SortIterator(ptr_ + diff * stride_, stride_);
  }

  SortIterator operator-(difference_type diff) const {
    return SortIterator(ptr_ - diff * stride_, stride_);
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
static SortDims GetSortDims(const Shape& shape, int64_t dimension) {
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
  int64_t num_iterations = outer_dim_size * inner_dim_size;

  return SortDims{outer_dim_size, sort_dim_size, inner_dim_size,
                  num_iterations};
}

template <class Iterator, class NativeT>
static void Sort1DArrInplace(int64_t sort_dims_size, int64_t offset,
                             Iterator begin, bool is_stable,
                             SortThunk::SortDirection direction) {
  if (direction == SortThunk::SortDirection::kAscending) {
    if (is_stable) {
      std::stable_sort(begin, begin + sort_dims_size, std::less<NativeT>());
    } else {
      std::sort(begin, begin + sort_dims_size, std::less<NativeT>());
    }
  } else {
    if (is_stable) {
      std::stable_sort(begin, begin + sort_dims_size, std::greater<NativeT>());
    } else {
      std::sort(begin, begin + sort_dims_size, std::greater<NativeT>());
    }
  };
}

// The most efficient way to sort a single buffer is to use the builtin
// comparator functions.
template <PrimitiveType Type>
static void Sort1DArrInplace(const SortDims& sort_dims, int64_t offset,
                             absl::Span<se::DeviceMemoryBase> data,
                             bool is_stable,
                             SortThunk::SortDirection direction) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<Type>::type;
  DCHECK_EQ(data.size(), 1);
  NativeT* begin = reinterpret_cast<NativeT*>(data[0].opaque()) + offset;

  if (sort_dims.inner_dim_size == 1) {
    Sort1DArrInplace<NativeT*, NativeT>(sort_dims.sort_dim_size, offset, begin,
                                        is_stable, direction);
  } else {
    using Iterator = SortIterator<NativeT, NativeT&, NativeT*>;
    Iterator begin_iter(begin, /*stride=*/sort_dims.inner_dim_size);
    Sort1DArrInplace<Iterator, NativeT>(sort_dims.sort_dim_size, offset,
                                        begin_iter, is_stable, direction);
  }
}

// Sorts `n` buffers in place.
template <size_t n>
static void SortInplace(const SortDims& sort_dims, int64_t offset,
                        absl::Span<se::DeviceMemoryBase> data,
                        absl::Span<const Shape> shapes, bool is_stable,
                        SortThunk::LessThan* less_than) {
  std::array<std::byte*, n> ptrs;
  std::array<size_t, n> primitive_sizes;

  for (size_t i = 0; i < n; ++i) {
    std::byte* base = reinterpret_cast<std::byte*>(data[i].opaque());
    primitive_sizes[i] = primitive_util::ByteWidth(shapes[i].element_type());
    ptrs[i] = base + offset * primitive_sizes[i];
  }

  Inputs<n> inputs(ptrs, primitive_sizes);

  auto compare = [&](const auto& a, const auto& b) {
    std::array<const void*, 2 * n> values;
    a.FillComparedValues(&values[0]);
    b.FillComparedValues(&values[1]);
    return (*less_than)(values.data());
  };

  SortIterator<Value<n>, Ref<n>, Ptr<n>> begin(
      Ptr<n>(&inputs), /*stride=*/sort_dims.inner_dim_size);
  if (is_stable) {
    std::stable_sort(begin, begin + sort_dims.sort_dim_size, compare);
  } else {
    std::sort(begin, begin + sort_dims.sort_dim_size, compare);
  }
}

static void DSortInplace(const SortDims& sort_dims, int64_t offset,
                         absl::Span<se::DeviceMemoryBase> data,
                         absl::Span<const Shape> shapes, bool is_stable,
                         SortThunk::LessThan* less_than, size_t n) {
  std::vector<std::byte*> ptrs(n);
  std::vector<size_t> primitive_sizes(n);

  for (size_t i = 0; i < n; ++i) {
    std::byte* base = reinterpret_cast<std::byte*>(data[i].opaque());
    primitive_sizes[i] = primitive_util::ByteWidth(shapes[i].element_type());
    ptrs[i] = base + offset * primitive_sizes[i];
  }

  DInputs inputs(std::move(ptrs), std::move(primitive_sizes));

  // Allocate scratch space for sorted values outside of the lambda to avoid
  // allocating it on every call to `compare`.
  std::vector<const void*> values(2 * n);

  auto compare = [&, values = values.data()](const auto& a, const auto& b) {
    a.FillComparedValues(&values[0]);
    b.FillComparedValues(&values[1]);
    return (*less_than)(values);
  };

  SortIterator<DValue, DRef, DPtr> begin(DPtr(&inputs),
                                         /*stride=*/sort_dims.inner_dim_size);
  if (is_stable) {
    std::stable_sort(begin, begin + sort_dims.sort_dim_size, compare);
  } else {
    std::sort(begin, begin + sort_dims.sort_dim_size, compare);
  }
}

// Sorts `data` of the given `shape` along the `dimension` inplace.
static absl::Status SortInplace(
    absl::Span<se::DeviceMemoryBase> data, absl::Span<const Shape> shapes,
    int64_t dimension, bool is_stable, SortThunk::LessThan* less_than,
    std::optional<SortThunk::SortDirection> direction) {
  // All inputs have the same dimensions and layout, so we can use the first
  // shape to get the sort dimensions.
  SortDims sort_dims = GetSortDims(shapes[0], dimension);

  // Iterate over all the 1-dimensional slices of the buffers and sort them.
  for (int64_t i = 0; i < sort_dims.num_iterations; ++i) {
    int64_t inner_idx = i % sort_dims.inner_dim_size;
    int64_t offset = inner_idx + (i - inner_idx) * sort_dims.sort_dim_size;

    auto sort = [&](auto num_inputs) {
      SortInplace<decltype(num_inputs)::value>(sort_dims, offset, data, shapes,
                                               is_stable, less_than);
    };

    auto dsort = [&](size_t num_inputs) {
      DSortInplace(sort_dims, offset, data, shapes, is_stable, less_than,
                   num_inputs);
    };

    // Sorts array using builtin comparator functor
    auto builtin_sort = [&](PrimitiveType type,
                            SortThunk::SortDirection direction) {
      primitive_util::ArrayTypeSwitch(
          [&](auto cst_type) {
            if constexpr ((primitive_util::IsFloatingPointType(cst_type) ||
                           primitive_util::IsIntegralType(cst_type)) &&
                          primitive_util::BitWidth(cst_type) >= 8) {
              Sort1DArrInplace<cst_type>(sort_dims, offset, data, is_stable,
                                         direction);
            } else {
              sort(std::integral_constant<size_t, 1>{});
            }
          },
          type);
    };

    // Use "sort" for statically known number of sorted inputs (expected to be
    // faster) and "dsort" for dynamically known number of sorted inputs.
    switch (data.size()) {
      case 1:
        DCHECK_EQ(shapes.size(), 1);
        if (direction.has_value()) {
          builtin_sort(shapes[0].element_type(), *direction);
        } else {
          sort(std::integral_constant<size_t, 1>{});
        }
        break;
      case 2:
        sort(std::integral_constant<size_t, 2>{});
        break;
      case 3:
        sort(std::integral_constant<size_t, 3>{});
        break;
      case 4:
        sort(std::integral_constant<size_t, 4>{});
        break;
      case 5:
        sort(std::integral_constant<size_t, 5>{});
        break;
      case 6:
        sort(std::integral_constant<size_t, 6>{});
        break;
      case 7:
        sort(std::integral_constant<size_t, 7>{});
        break;
      case 8:
        sort(std::integral_constant<size_t, 8>{});
        break;
      case 9:
        sort(std::integral_constant<size_t, 9>{});
        break;
      case 10:
        sort(std::integral_constant<size_t, 10>{});
        break;
      case 11:
        sort(std::integral_constant<size_t, 11>{});
        break;
      case 12:
        sort(std::integral_constant<size_t, 12>{});
        break;
      case 13:
        sort(std::integral_constant<size_t, 13>{});
        break;
      case 14:
        sort(std::integral_constant<size_t, 14>{});
        break;
      case 15:
        sort(std::integral_constant<size_t, 15>{});
        break;
      case 16:
        sort(std::integral_constant<size_t, 16>{});
        break;
      default:
        dsort(data.size());
        break;
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

    // Annotate memory that might have been initialized by jit-compiled code.
    ABSL_ANNOTATE_MEMORY_IS_INITIALIZED(data.back().opaque(),
                                        data.back().size());

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

  TF_RETURN_IF_ERROR(less_than_.status());
  LessThan* less_than = &less_than_.value();

  TF_RETURN_IF_ERROR(SortInplace(absl::MakeSpan(data), shapes, dimension_,
                                 is_stable_, less_than, direction_));

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
