/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/sort_lib.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/types.h"

namespace xla::cpu::internal {

namespace {

template <typename T>
inline constexpr bool is_float_type_v =
    std::is_floating_point_v<T> || std::is_same_v<T, bfloat16> ||
    std::is_same_v<T, half>;

template <typename T>
inline bool IsNaN(const T& val) {
  if constexpr (std::is_same_v<T, double>) {
    return std::isnan(val);
  } else if constexpr (std::is_same_v<T, float> ||
                       std::is_same_v<T, bfloat16> || std::is_same_v<T, half>) {
    return std::isnan(static_cast<float>(val));
  } else {
    return false;
  }
}

template <typename T>
struct SortComparatorLess {
  bool operator()(const T& a, const T& b) const {
    if constexpr (is_float_type_v<T>) {
      bool a_nan = IsNaN(a);
      bool b_nan = IsNaN(b);
      if (a_nan || b_nan) {
        return !a_nan && b_nan;
      }
    }
    return a < b;
  }
};

template <typename T>
struct SortComparatorGreater {
  bool operator()(const T& a, const T& b) const {
    if constexpr (is_float_type_v<T>) {
      bool a_nan = IsNaN(a);
      bool b_nan = IsNaN(b);
      if (a_nan || b_nan) {
        return a_nan && !b_nan;
      }
    }
    return a > b;
  }
};

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
  Inputs(absl::Span<std::byte* const> ptrs,
         absl::Span<const size_t> primitive_sizes) {
    DCHECK_EQ(n, ptrs.size());
    DCHECK_EQ(n, primitive_sizes.size());
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
  DInputs(absl::Span<std::byte* const> ptrs,
          absl::Span<const size_t> primitive_sizes)
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

template <typename Key, typename Value>
struct ZipRef {
  Key& key;
  Value& value;

  ZipRef(Key& k, Value& v) : key(k), value(v) {}
  ZipRef(const ZipRef&) = default;

  // Non-defaulted copy assignment is required to write through to referenced
  // memory (defaulted assignment is implicitly deleted on reference members).
  // NOLINTNEXTLINE(modernize-use-equals-default)
  ZipRef& operator=(const ZipRef& other) {
    key = other.key;
    value = other.value;
    return *this;
  }

  ZipRef& operator=(const std::pair<Key, Value>& p) {
    key = p.first;
    value = p.second;
    return *this;
  }

  ZipRef& operator=(std::pair<Key, Value>&& p) {
    key = std::move(p.first);
    value = std::move(p.second);
    return *this;
  }

  // Implicit conversion to std::pair is required by std::sort / STL algorithms
  // for copy-initialization of temporary value_type variables (e.g. pivots).
  // NOLINTNEXTLINE(google-explicit-constructor)
  operator std::pair<Key, Value>() const { return {key, value}; }

  friend void swap(ZipRef a, ZipRef b) noexcept {
    std::swap(a.key, b.key);
    std::swap(a.value, b.value);
  }

  friend void swap(ZipRef a, std::pair<Key, Value>& b) noexcept {
    std::swap(a.key, b.first);
    std::swap(a.value, b.second);
  }

  friend void swap(std::pair<Key, Value>& a, ZipRef b) noexcept {
    std::swap(a.first, b.key);
    std::swap(a.second, b.value);
  }
};

template <typename Key, typename Value>
class ZipIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using value_type = std::pair<Key, Value>;
  using difference_type = std::ptrdiff_t;
  using reference = ZipRef<Key, Value>;
  // Required by std::iterator_traits in C++17 for proxy iterators.
  using pointer = void;

  ZipIterator() : key_ptr_(nullptr), val_ptr_(nullptr) {}
  ZipIterator(Key* key_ptr, Value* val_ptr)
      : key_ptr_(key_ptr), val_ptr_(val_ptr) {}

  reference operator*() const { return reference(*key_ptr_, *val_ptr_); }
  reference operator[](difference_type n) const {
    return reference(key_ptr_[n], val_ptr_[n]);
  }

  ZipIterator& operator++() {
    ++key_ptr_;
    ++val_ptr_;
    return *this;
  }
  ZipIterator operator++(int) {
    ZipIterator tmp = *this;
    ++(*this);
    return tmp;
  }
  ZipIterator& operator--() {
    --key_ptr_;
    --val_ptr_;
    return *this;
  }
  ZipIterator operator--(int) {
    ZipIterator tmp = *this;
    --(*this);
    return tmp;
  }

  ZipIterator& operator+=(difference_type n) {
    key_ptr_ += n;
    val_ptr_ += n;
    return *this;
  }
  ZipIterator& operator-=(difference_type n) {
    key_ptr_ -= n;
    val_ptr_ -= n;
    return *this;
  }

  friend ZipIterator operator+(ZipIterator it, difference_type n) {
    return it += n;
  }
  friend ZipIterator operator+(difference_type n, ZipIterator it) {
    return it += n;
  }
  friend ZipIterator operator-(ZipIterator it, difference_type n) {
    return it -= n;
  }
  friend difference_type operator-(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ - b.key_ptr_;
  }

  friend bool operator==(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ == b.key_ptr_;
  }
  friend bool operator!=(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ != b.key_ptr_;
  }
  friend bool operator<(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ < b.key_ptr_;
  }
  friend bool operator>(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ > b.key_ptr_;
  }
  friend bool operator<=(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ <= b.key_ptr_;
  }
  friend bool operator>=(const ZipIterator& a, const ZipIterator& b) {
    return a.key_ptr_ >= b.key_ptr_;
  }

  Key* key_ptr() const { return key_ptr_; }
  Value* val_ptr() const { return val_ptr_; }

  friend void iter_swap(ZipIterator a, ZipIterator b) noexcept {
    std::swap(*a.key_ptr_, *b.key_ptr_);
    std::swap(*a.val_ptr_, *b.val_ptr_);
  }

 private:
  Key* key_ptr_;
  Value* val_ptr_;
};

template <typename Key, typename Value, typename SliceSorter>
static void Sort2DSlices(const SortDims& sort_dims, int64_t start_slice,
                         int64_t end_slice, Key* keys, Value* values,
                         SliceSorter&& sort_slice) {
  DCHECK_LE(0, start_slice);
  DCHECK_LE(start_slice, end_slice);
  DCHECK_LE(end_slice, sort_dims.outer_dim_size * sort_dims.inner_dim_size);

  int64_t n = sort_dims.sort_dim_size;
  if (sort_dims.inner_dim_size == 1) {
    for (int64_t i = start_slice; i < end_slice; ++i) {
      int64_t offset = i * n;
      sort_slice(n, keys + offset, values + offset);
    }
    return;
  }

  // Strided case (inner_dim_size > 1): Gather into contiguous arrays (SoA),
  // sort with contiguous ZipIterator, and scatter back.
  DCHECK_GT(sort_dims.inner_dim_size, 0);
  int64_t stride = sort_dims.inner_dim_size;
  auto sort_strided_slice = [&](int64_t offset, Key* key_buf, Value* val_buf) {
    Key* key_ptr = keys + offset;
    Value* val_ptr = values + offset;
    for (int64_t i = 0; i < n; ++i) {
      key_buf[i] = key_ptr[i * stride];
      val_buf[i] = val_ptr[i * stride];
    }
    sort_slice(n, key_buf, val_buf);
    for (int64_t i = 0; i < n; ++i) {
      key_ptr[i * stride] = key_buf[i];
      val_ptr[i * stride] = val_buf[i];
    }
  };

  int64_t slice_stride = n * stride;
  int64_t outer = start_slice / stride;
  int64_t inner = start_slice % stride;
  int64_t offset = outer * slice_stride + inner;

  auto run_slices = [&](Key* key_buf, Value* val_buf) {
    for (int64_t i = start_slice; i < end_slice; ++i) {
      sort_strided_slice(offset, key_buf, val_buf);
      if (++inner == stride) {
        inner = 0;
        offset += slice_stride - stride + 1;
      } else {
        ++offset;
      }
    }
  };

  if (n <= 1024) {
    std::array<Key, 1024> key_buf;
    std::array<Value, 1024> val_buf;
    run_slices(key_buf.data(), val_buf.data());
  } else {
    std::vector<Key> key_buf(n);
    std::vector<Value> val_buf(n);
    run_slices(key_buf.data(), val_buf.data());
  }
}

template <typename Key, typename Value>
static void Sort2DSliceWithComparator(int64_t n, Key* keys, Value* values,
                                      bool is_stable, LessThan* less_than) {
  auto comp = [&](const auto& a, const auto& b) -> bool {
    auto get_ptrs =
        [](const auto& item) -> std::pair<const void*, const void*> {
      using T = std::decay_t<decltype(item)>;
      if constexpr (std::is_same_v<T, std::pair<Key, Value>>) {
        return {&item.first, &item.second};
      } else {
        return {&item.key, &item.value};
      }
    };
    auto [a0, a1] = get_ptrs(a);
    auto [b0, b1] = get_ptrs(b);
    const void* values_ptrs[4] = {a0, b0, a1, b1};
    return (*less_than)(values_ptrs);
  };

  ZipIterator<Key, Value> begin(keys, values);
  ZipIterator<Key, Value> end(keys + n, values + n);
  if (is_stable) {
    std::stable_sort(begin, end, comp);
  } else {
    std::sort(begin, end, comp);
  }
}

template <typename Key, typename Value>
static void Sort2DWithComparator(const SortDims& sort_dims, int64_t start_slice,
                                 int64_t end_slice, Key* keys, Value* values,
                                 bool is_stable, LessThan* less_than) {
  Sort2DSlices(sort_dims, start_slice, end_slice, keys, values,
               [is_stable, less_than](int64_t n, Key* k, Value* v) {
                 Sort2DSliceWithComparator<Key, Value>(n, k, v, is_stable,
                                                       less_than);
               });
}

// Dispatches to a generic functor parameterized by an unsigned integer type
// of matching byte size (uint8_t, uint16_t, uint32_t, uint64_t).
//
// When sorting with a non-inlined `less_than` comparator callback, comparisons
// are delegated via `const void*` pointers, so the sort algorithm does not need
// semantic type information (e.g. float vs int32_t). The types are only used
// for pointer arithmetic and data movement (swapping and temporary pivot copies
// on the stack). Because all trivially copyable types of the same byte width
// share the same size, alignment, and copy semantics, unsigned integers are
// binary-compatible stand-ins, avoiding the need to instantiate templates for
// every semantic type combination.
template <typename Fn>
bool DispatchBySize(size_t size, Fn&& fn) {
  switch (size) {
    case 1:
      return fn(uint8_t{});
    case 2:
      return fn(uint16_t{});
    case 4:
      return fn(uint32_t{});
    case 8:
      return fn(uint64_t{});
    default:
      return false;
  }
}

template <size_t n>
void Sort1DInplace(const SortDims& sort_dims, int64_t offset,
                   absl::Span<std::byte* const> data,
                   absl::Span<const size_t> primitive_sizes, bool is_stable,
                   LessThan* less_than) {
  DCHECK_EQ(n, data.size());
  DCHECK_EQ(n, primitive_sizes.size());

  std::array<std::byte*, n> ptrs;
  for (size_t i = 0; i < n; ++i) {
    ptrs[i] = data[i] + offset * primitive_sizes[i];
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

void DSort1DInplace(const SortDims& sort_dims, int64_t offset,
                    absl::Span<std::byte* const> data,
                    absl::Span<const size_t> primitive_sizes, bool is_stable,
                    LessThan* less_than) {
  DCHECK_EQ(data.size(), primitive_sizes.size());

  std::vector<std::byte*> ptrs(data.size());
  for (size_t i = 0; i < data.size(); ++i) {
    ptrs[i] = data[i] + offset * primitive_sizes[i];
  }

  DInputs inputs(std::move(ptrs), primitive_sizes);

  // Allocate scratch space for sorted values outside of the lambda to avoid
  // allocating it on every call to `compare`.
  std::vector<const void*> values(2 * data.size());

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

template <class Iterator, class T>
void Sort1DInplace(Iterator begin, Iterator end, bool is_stable,
                   SortDirection direction) {
  if constexpr (std::is_integral_v<T>) {
    if (direction == SortDirection::kAscending) {
      if (is_stable) {
        std::stable_sort(begin, end, std::less<T>());
      } else {
        std::sort(begin, end, std::less<T>());
      }
    } else {
      if (is_stable) {
        std::stable_sort(begin, end, std::greater<T>());
      } else {
        std::sort(begin, end, std::greater<T>());
      }
    }
  } else {
    if (direction == SortDirection::kAscending) {
      if (is_stable) {
        std::stable_sort(begin, end, SortComparatorLess<T>());
      } else {
        std::sort(begin, end, SortComparatorLess<T>());
      }
    } else {
      if (is_stable) {
        std::stable_sort(begin, end, SortComparatorGreater<T>());
      } else {
        std::sort(begin, end, SortComparatorGreater<T>());
      }
    }
  }
}

template <typename T>
void Sort1DInplace(const SortDims& sort_dims, int64_t offset, T* data,
                   bool is_stable, SortDirection direction) {
  T* begin = data + offset;
  T* end = begin + sort_dims.sort_dim_size;

  if (sort_dims.inner_dim_size == 1) {
    Sort1DInplace<T*, T>(begin, end, is_stable, direction);
  } else {
    using Iterator = internal::SortIterator<T, T&, T*>;
    Iterator begin_it(begin, /*stride=*/sort_dims.inner_dim_size);
    Iterator end_it = begin_it + sort_dims.sort_dim_size;
    Sort1DInplace<Iterator, T>(begin_it, end_it, is_stable, direction);
  }
}

template <typename Key, typename Value>
void SortKeyValueSlice(int64_t n, Key* keys, Value* values, bool is_stable,
                       SortDirection direction) {
  auto comp = [direction](const auto& a, const auto& b) -> bool {
    auto get_key = [](const auto& item) -> const Key& {
      using T = std::decay_t<decltype(item)>;
      if constexpr (std::is_same_v<T, std::pair<Key, Value>>) {
        return item.first;
      } else {
        return item.key;
      }
    };
    const Key& ka = get_key(a);
    const Key& kb = get_key(b);
    if constexpr (std::is_integral_v<Key>) {
      if (direction == SortDirection::kAscending) {
        return std::less<Key>()(ka, kb);
      }
      return std::greater<Key>()(ka, kb);
    } else {
      if (direction == SortDirection::kAscending) {
        return SortComparatorLess<Key>()(ka, kb);
      }
      return SortComparatorGreater<Key>()(ka, kb);
    }
  };

  ZipIterator<Key, Value> begin(keys, values);
  ZipIterator<Key, Value> end(keys + n, values + n);
  if (is_stable) {
    std::stable_sort(begin, end, comp);
  } else {
    std::sort(begin, end, comp);
  }
}

}  // namespace

// Sorts `data` using `less_than` comparator function for slices in
// [start_slice, end_slice).
void SortInplace(const SortDims& sort_dims, int64_t start_slice,
                 int64_t end_slice, absl::Span<std::byte* const> data,
                 absl::Span<const size_t> primitive_sizes, bool is_stable,
                 LessThan* less_than) {
  DCHECK_LE(0, start_slice);
  DCHECK_LE(start_slice, end_slice);
  DCHECK_LE(end_slice, sort_dims.outer_dim_size * sort_dims.inner_dim_size);
  DCHECK_EQ(data.size(), primitive_sizes.size());

  if (data.size() == 2 &&
      (sort_dims.inner_dim_size == 1 || sort_dims.sort_dim_size <= 65536)) {
    bool dispatched = DispatchBySize(primitive_sizes[0], [&](auto key_dummy) {
      using Key = decltype(key_dummy);
      return DispatchBySize(primitive_sizes[1], [&](auto val_dummy) {
        using Value = decltype(val_dummy);
        Sort2DWithComparator<Key, Value>(
            sort_dims, start_slice, end_slice, reinterpret_cast<Key*>(data[0]),
            reinterpret_cast<Value*>(data[1]), is_stable, less_than);
        return true;
      });
    });
    if (dispatched) {
      return;
    }
  }

  for (int64_t i = start_slice; i < end_slice; ++i) {
    int64_t inner_idx = i % sort_dims.inner_dim_size;
    int64_t offset = inner_idx + (i - inner_idx) * sort_dims.sort_dim_size;

    // Use "sort" for statically known number of sorted inputs (expected to be
    // faster) and "dsort" for dynamically known number of sorted inputs.
    auto sort = [&](auto num_inputs) {
      Sort1DInplace<decltype(num_inputs)::value>(
          sort_dims, offset, data, primitive_sizes, is_stable, less_than);
    };

    switch (data.size()) {
      case 1:
        sort(std::integral_constant<size_t, 1>{});
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
        DSort1DInplace(sort_dims, offset, data, primitive_sizes, is_stable,
                       less_than);
        break;
    }
  }
}

template <typename T>
void SortInplace(const SortDims& sort_dims, int64_t start_slice,
                 int64_t end_slice, T* data, bool is_stable,
                 SortDirection direction) {
  DCHECK_LE(0, start_slice);
  DCHECK_LE(start_slice, end_slice);
  DCHECK_LE(end_slice, sort_dims.outer_dim_size * sort_dims.inner_dim_size);

  for (int64_t i = start_slice; i < end_slice; ++i) {
    int64_t inner_idx = i % sort_dims.inner_dim_size;
    int64_t offset = inner_idx + (i - inner_idx) * sort_dims.sort_dim_size;

    Sort1DInplace<T>(sort_dims, offset, data, is_stable, direction);
  }
}

// Declare SortInplace for all supported types. Template is instantiated in
// the .cc file.
#define DEFINE_SORT_INPLACE(T)                                              \
  template void SortInplace<T>(const SortDims&, int64_t, int64_t, T*, bool, \
                               SortDirection)

DEFINE_SORT_INPLACE(float);
DEFINE_SORT_INPLACE(double);
DEFINE_SORT_INPLACE(bfloat16);
DEFINE_SORT_INPLACE(half);
DEFINE_SORT_INPLACE(int8_t);
DEFINE_SORT_INPLACE(int16_t);
DEFINE_SORT_INPLACE(int32_t);
DEFINE_SORT_INPLACE(int64_t);
DEFINE_SORT_INPLACE(uint8_t);
DEFINE_SORT_INPLACE(uint16_t);
DEFINE_SORT_INPLACE(uint32_t);
DEFINE_SORT_INPLACE(uint64_t);

#undef DEFINE_SORT_INPLACE

template <typename Key, typename Value>
void Sort2DKeyValue(const SortDims& sort_dims, int64_t start_slice,
                    int64_t end_slice, Key* keys, Value* values, bool is_stable,
                    SortDirection direction) {
  Sort2DSlices(sort_dims, start_slice, end_slice, keys, values,
               [is_stable, direction](int64_t n, Key* k, Value* v) {
                 SortKeyValueSlice<Key, Value>(n, k, v, is_stable, direction);
               });
}

#define DEFINE_SORT_2D_KEY_VALUE(Key, Value)                                  \
  template void Sort2DKeyValue<Key, Value>(const SortDims&, int64_t, int64_t, \
                                           Key*, Value*, bool, SortDirection)

#define DEFINE_SORT_2D_KEY_VALUE_KEY(Key)  \
  DEFINE_SORT_2D_KEY_VALUE(Key, uint8_t);  \
  DEFINE_SORT_2D_KEY_VALUE(Key, uint16_t); \
  DEFINE_SORT_2D_KEY_VALUE(Key, uint32_t); \
  DEFINE_SORT_2D_KEY_VALUE(Key, uint64_t)

DEFINE_SORT_2D_KEY_VALUE_KEY(float);
DEFINE_SORT_2D_KEY_VALUE_KEY(double);
DEFINE_SORT_2D_KEY_VALUE_KEY(bfloat16);
DEFINE_SORT_2D_KEY_VALUE_KEY(half);
DEFINE_SORT_2D_KEY_VALUE_KEY(int8_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(int16_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(int32_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(int64_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(uint8_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(uint16_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(uint32_t);
DEFINE_SORT_2D_KEY_VALUE_KEY(uint64_t);

#undef DEFINE_SORT_2D_KEY_VALUE_KEY
#undef DEFINE_SORT_2D_KEY_VALUE

}  // namespace xla::cpu::internal
