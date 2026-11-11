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

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/types.h"

namespace xla::cpu::internal {

namespace {

template <size_t kBytes>
using UnsignedKeyType = std::conditional_t<
    kBytes == 1, uint8_t,
    std::conditional_t<kBytes == 2, uint16_t,
                       std::conditional_t<kBytes == 4, uint32_t, uint64_t>>>;

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
bool LessThanWithNaN(const void** data) {
  T a;
  T b;
  std::memcpy(&a, data[0], sizeof(T));
  std::memcpy(&b, data[1], sizeof(T));
  if constexpr (is_float_type_v<T>) {
    bool a_nan = IsNaN(a);
    bool b_nan = IsNaN(b);
    if (a_nan || b_nan) {
      return !a_nan && b_nan;
    }
  }
  return a < b;
}

template <typename T>
bool GreaterThanWithNaN(const void** data) {
  T a;
  T b;
  std::memcpy(&a, data[0], sizeof(T));
  std::memcpy(&b, data[1], sizeof(T));
  if constexpr (is_float_type_v<T>) {
    bool a_nan = IsNaN(a);
    bool b_nan = IsNaN(b);
    if (a_nan || b_nan) {
      return a_nan && !b_nan;
    }
  }
  return a > b;
}

template <typename Key>
constexpr UnsignedKeyType<sizeof(Key)> FloatExpMask() {
  if constexpr (std::is_same_v<Key, half>) {
    return 0x7c00u;
  } else if constexpr (std::is_same_v<Key, bfloat16>) {
    return 0x7f80u;
  } else if constexpr (std::is_same_v<Key, float>) {
    return 0x7f800000u;
  } else {
    static_assert(std::is_same_v<Key, double>);
    return 0x7ff0000000000000ull;
  }
}

// Encodes IEEE-754 floating-point keys (`half`, `bfloat16`, `float`, `double`)
// in-place into unsigned integers of the same byte width (`UIntKey`) whose
// standard unsigned `<` order matches floating-point ascending/descending
// order:
//
// 1. IEEE-754 floats use sign-magnitude representation (`[sign][exp][mant]`).
//    - Positive floats (`sign == 0`, `0x0000..00` to `0x7f80..00`) already
//      increase monotonically in raw unsigned bits; flipping only the sign bit
//      (`u ^ kSignBit`) maps `[+0.0, +Inf]` onto `[0x8000..00, 0xff80..00]`.
//    - Negative floats (`sign == 1`, `0x8000..00` to `0xff80..00`) increase in
//      magnitude as their raw unsigned bits increase, which is the reverse of
//      numeric `<` order; flipping all bits (`u ^ ~0`) reverses their order and
//      maps `[-Inf, -0.0]` onto `[0x007f..ff, 0x7fff..ff]`.
// 2. Arithmetic right-shift `static_cast<SIntKey>(u) >> (kBits - 1)` broadcasts
//    the sign bit across all bits (`0` for positive, `~0` for negative), so
//    `mask = (signed(u) >> (kBits - 1)) | kSignBit` equals `kSignBit` for
//    positive floats and `~0` for negative floats without branching.
// 3. XORing with `dir_xor` (`0` for ascending, `~0` for descending) optionally
//    inverts all bits (`~x = MAX - x`) so descending sorts can reuse the same
//    ascending `std::less<UIntKey>` sort instantiation.
//
// Returns `true` if no element is `NaN` (`(u & kAbsMask) > exp_mask`) and the
// slice does not contain both `-0.0` (`u == kSignBit`) and `+0.0` (`u == 0`).
// Slices containing only `-0.0` (without `+0.0`) or only `+0.0` (without
// `-0.0`) stay on the fast integer sort path because all zeros in the slice
// share the same encoded key (`0x7fff..ff` or `0x8000..00`), which lies
// strictly between all negative and all positive finite numbers. If any `NaN`
// or both `-0.0` and `+0.0` are present, returns `false` so the caller can
// restore the original float bits via `DecodeFloatKeys` and fall back to
// `LessThanWithNaN` / `GreaterThanWithNaN` (preserving `-0.0 == +0.0` stability
// and XLA's total ordering of negative `NaN`s after `+Inf`).
template <typename UIntKey>
bool EncodeFloatKeys(int64_t n, UIntKey* keys, UIntKey dir_xor,
                     UIntKey exp_mask) {
  using SIntKey = std::make_signed_t<UIntKey>;
  constexpr size_t kBits = sizeof(UIntKey) * 8;
  constexpr UIntKey kSignBit = UIntKey{1} << (kBits - 1);
  constexpr UIntKey kAbsMask = kSignBit - 1;

  UIntKey has_nan = 0;
  UIntKey has_neg_zero = 0;
  UIntKey has_pos_zero = 0;
  for (int64_t i = 0; i < n; ++i) {
    UIntKey u;
    std::memcpy(&u, &keys[i], sizeof(UIntKey));
    has_nan |= static_cast<UIntKey>((u & kAbsMask) > exp_mask);
    has_neg_zero |= static_cast<UIntKey>(u == kSignBit);
    has_pos_zero |= static_cast<UIntKey>(u == 0);
    UIntKey mask =
        static_cast<UIntKey>(static_cast<SIntKey>(u) >> (kBits - 1)) | kSignBit;
    u = (u ^ mask) ^ dir_xor;
    std::memcpy(&keys[i], &u, sizeof(UIntKey));
  }
  return (has_nan | (has_neg_zero & has_pos_zero)) == 0;
}

// Exact inverse of `EncodeFloatKeys`: after undoing `dir_xor`, encoded negative
// floats have `MSB == 0` and encoded positive floats have `MSB == 1`, so
// `(~static_cast<SIntKey>(u) >> (kBits - 1)) | kSignBit` reconstructs the same
// per-element `mask` (`~0` for negative floats, `kSignBit` for positive floats)
// and `u ^ mask` restores the original IEEE-754 bit pattern in-place.
template <typename UIntKey>
void DecodeFloatKeys(int64_t n, UIntKey* keys, UIntKey dir_xor) {
  using SIntKey = std::make_signed_t<UIntKey>;
  constexpr size_t kBits = sizeof(UIntKey) * 8;
  constexpr UIntKey kSignBit = UIntKey{1} << (kBits - 1);

  for (int64_t i = 0; i < n; ++i) {
    UIntKey u;
    std::memcpy(&u, &keys[i], sizeof(UIntKey));
    u ^= dir_xor;
    UIntKey mask =
        static_cast<UIntKey>(~static_cast<SIntKey>(u) >> (kBits - 1)) |
        kSignBit;
    u ^= mask;
    std::memcpy(&keys[i], &u, sizeof(UIntKey));
  }
}

// Applies a self-inverse bitwise XOR mask (`(u ^ xor_mask) ^ xor_mask == u`)
// in-place before and after sorting integral keys as unsigned integers:
// - For two's-complement signed integers (`int8_t`..`int64_t`), flipping the
//   MSB (`kSignBit`) shifts `[INT_MIN, INT_MAX]` monotonically onto
//   `[0, UINT_MAX]` so unsigned `<` matches signed `<`.
// - For descending order, flipping all bits (`dir_xor = ~0`, i.e. `MAX - u`)
//   reverses unsigned `<` order (`~a < ~b <=> a > b`).
// - Because XOR is associative, `xor_mask = sign_bit_mask ^ dir_xor` combines
//   both transforms into a single pass.
template <typename UIntKey>
void XorKeys(int64_t n, UIntKey* keys, UIntKey xor_mask) {
  if (xor_mask == 0) {
    return;
  }
  for (int64_t i = 0; i < n; ++i) {
    UIntKey u;
    std::memcpy(&u, &keys[i], sizeof(UIntKey));
    u ^= xor_mask;
    std::memcpy(&keys[i], &u, sizeof(UIntKey));
  }
}

// The size of the largest element we support (std::complex<double>).
constexpr size_t kMaxElementSize = 16;

// Type-erased byte storage suitable for aliasing and moving any primitive type
// of `kBytes` bytes without violating C++ strict-aliasing rules (`[basic.lval]`
// permits `std::byte` aliasing). Capping alignment at `alignof(uint64_t)` (8
// bytes) matches `alignof(std::complex<double>)` for 16-byte `C128` buffers
// (which are only guaranteed 8-byte alignment) and avoids extended alignment in
// `std::stable_sort` on Windows x64 MSVC.
template <size_t kBytes>
struct alignas(std::min<size_t>(kBytes, alignof(uint64_t))) ByteStorage
    : std::array<std::byte, kBytes> {};

template <typename Elem, typename SliceSorter>
void Sort1DSlices(const SortDims& sort_dims, int64_t start_slice,
                  int64_t end_slice, Elem* data, SliceSorter&& sort_slice) {
  DCHECK_LE(0, start_slice);
  DCHECK_LE(start_slice, end_slice);
  DCHECK_LE(end_slice, sort_dims.outer_dim_size * sort_dims.inner_dim_size);

  int64_t n = sort_dims.sort_dim_size;
  if (n <= 1 || start_slice >= end_slice) {
    return;
  }
  if (sort_dims.inner_dim_size == 1) {
    for (int64_t i = start_slice; i < end_slice; ++i) {
      sort_slice(n, data + i * n);
    }
    return;
  }

  // Strided case (inner_dim_size > 1): Gather into a contiguous buffer,
  // sort contiguously, and scatter back.
  DCHECK_GT(sort_dims.inner_dim_size, 0);
  int64_t stride = sort_dims.inner_dim_size;
  auto sort_strided_slice = [&](int64_t offset, Elem* buf) {
    Elem* ptr = data + offset;
    for (int64_t i = 0; i < n; ++i) {
      buf[i] = ptr[i * stride];
    }
    sort_slice(n, buf);
    for (int64_t i = 0; i < n; ++i) {
      ptr[i * stride] = buf[i];
    }
  };

  int64_t slice_stride = n * stride;
  int64_t outer = start_slice / stride;
  int64_t inner = start_slice % stride;
  int64_t offset = outer * slice_stride + inner;

  auto run_slices = [&](Elem* buf) {
    for (int64_t i = start_slice; i < end_slice; ++i) {
      sort_strided_slice(offset, buf);
      if (++inner == stride) {
        inner = 0;
        offset += slice_stride - stride + 1;
      } else {
        ++offset;
      }
    }
  };

  if (n <= 1024) {
    std::array<Elem, 1024> buf;
    run_slices(buf.data());
  } else {
    std::vector<Elem> buf(n);
    run_slices(buf.data());
  }
}

template <typename Elem>
void Sort1DSliceWithComparator(int64_t n, Elem* data, bool is_stable,
                               LessThan* less_than) {
  auto comp = [less_than](const Elem& a, const Elem& b) -> bool {
    const void* values_ptrs[2] = {&a, &b};
    return (*less_than)(values_ptrs);
  };
  if (is_stable) {
    std::stable_sort(data, data + n, comp);
  } else {
    std::sort(data, data + n, comp);
  }
}

template <typename Elem>
void Sort1DWithComparator(const SortDims& sort_dims, int64_t start_slice,
                          int64_t end_slice, Elem* data, bool is_stable,
                          LessThan* less_than) {
  Sort1DSlices(sort_dims, start_slice, end_slice, data,
               [is_stable, less_than](int64_t n, Elem* slice) {
                 Sort1DSliceWithComparator<Elem>(n, slice, is_stable,
                                                 less_than);
               });
}

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
void Sort2DSlices(const SortDims& sort_dims, int64_t start_slice,
                  int64_t end_slice, Key* keys, Value* values,
                  SliceSorter&& sort_slice) {
  DCHECK_LE(0, start_slice);
  DCHECK_LE(start_slice, end_slice);
  DCHECK_LE(end_slice, sort_dims.outer_dim_size * sort_dims.inner_dim_size);

  int64_t n = sort_dims.sort_dim_size;
  if (n <= 1 || start_slice >= end_slice) {
    return;
  }
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
void Sort2DSliceWithComparator(int64_t n, Key* keys, Value* values,
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
void Sort2DWithComparator(const SortDims& sort_dims, int64_t start_slice,
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

template <typename Fn>
void DispatchByElementSize(size_t size, Fn&& fn) {
  switch (size) {
    case 1:
      fn(ByteStorage<1>{});
      return;
    case 2:
      fn(ByteStorage<2>{});
      return;
    case 4:
      fn(ByteStorage<4>{});
      return;
    case 8:
      fn(ByteStorage<8>{});
      return;
    case 16:
      fn(ByteStorage<16>{});
      return;
    default:
      LOG(FATAL) << "Unsupported primitive size: " << size;
  }
}

template <typename Elem>
void PermuteSlice(Elem* __restrict data, int64_t stride,
                  absl::Span<const size_t> indices, Elem* __restrict scratch) {
  const size_t n = indices.size();
  if (stride == 1) {
    for (size_t i = 0; i < n; ++i) {
      scratch[i] = data[indices[i]];
    }
    std::memcpy(data, scratch, n * sizeof(Elem));
  } else {
    for (size_t i = 0; i < n; ++i) {
      scratch[i] = data[indices[i] * stride];
    }
    for (size_t i = 0; i < n; ++i) {
      data[i * stride] = scratch[i];
    }
  }
}

// Sorts an arbitrary number of input buffers (`data.size() >= 2`) by sorting a
// contiguous array of slice element indices `0..n-1` and then permuting each
// input buffer once in a single linear pass.
//
// During the O(n log n) sort, all input buffers remain read-only and each
// element move or swap inside `std::sort` / `std::stable_sort` moves a single
// `size_t` index rather than copying or swapping across all `data.size()`
// buffers with per-input runtime `primitive_size` switches.
void SortNDInplace(const SortDims& sort_dims, int64_t start_slice,
                   int64_t end_slice, absl::Span<std::byte* const> data,
                   absl::Span<const size_t> primitive_sizes, bool is_stable,
                   LessThan* less_than) {
  const size_t n = sort_dims.sort_dim_size;
  const size_t num_inputs = data.size();
  if (n <= 1 || num_inputs == 0 || start_slice >= end_slice) {
    return;
  }

  struct InputSlice {
    std::byte* base;
    size_t byte_stride;
  };
  constexpr size_t kMaxStackInputs = 16;
  std::array<InputSlice, kMaxStackInputs> slices_stack;
  std::vector<InputSlice> slices_heap;
  InputSlice* slices = slices_stack.data();
  std::array<const void*, 2 * kMaxStackInputs> compared_stack;
  std::vector<const void*> compared_heap;
  const void** compared_values = compared_stack.data();
  if (num_inputs > kMaxStackInputs) {
    slices_heap.resize(num_inputs);
    slices = slices_heap.data();
    compared_heap.resize(2 * num_inputs);
    compared_values = compared_heap.data();
  }

  constexpr size_t kMaxStackIndices = 1024;
  std::array<size_t, kMaxStackIndices> indices_stack;
  std::vector<size_t> indices_heap;
  size_t* indices = indices_stack.data();
  if (n > kMaxStackIndices) {
    indices_heap.resize(n);
    indices = indices_heap.data();
  }

  using MaxByteStorage = ByteStorage<kMaxElementSize>;
  constexpr size_t kMaxStackScratch = 256;
  const size_t max_elem_size =
      *std::max_element(primitive_sizes.begin(), primitive_sizes.end());
  const size_t scratch_units =
      (n * max_elem_size + sizeof(MaxByteStorage) - 1) / sizeof(MaxByteStorage);
  std::array<MaxByteStorage, kMaxStackScratch> scratch_stack;
  std::vector<MaxByteStorage> scratch_heap;
  MaxByteStorage* scratch = scratch_stack.data();
  if (scratch_units > kMaxStackScratch) {
    scratch_heap.resize(scratch_units);
    scratch = scratch_heap.data();
  }

  const int64_t stride = sort_dims.inner_dim_size;
  for (int64_t s = start_slice; s < end_slice; ++s) {
    int64_t inner_idx = s % stride;
    int64_t offset = inner_idx + (s - inner_idx) * static_cast<int64_t>(n);

    for (size_t k = 0; k < num_inputs; ++k) {
      slices[k] = {data[k] + offset * primitive_sizes[k],
                   static_cast<size_t>(stride) * primitive_sizes[k]};
    }
    for (size_t i = 0; i < n; ++i) {
      indices[i] = i;
    }

    auto compare = [num_inputs, slices, compared_values, less_than](
                       size_t a, size_t b) -> bool {
      for (size_t k = 0; k < num_inputs; ++k) {
        const InputSlice& slice = slices[k];
        compared_values[2 * k] = slice.base + a * slice.byte_stride;
        compared_values[2 * k + 1] = slice.base + b * slice.byte_stride;
      }
      return (*less_than)(compared_values);
    };

    if (is_stable) {
      std::stable_sort(indices, indices + n, compare);
    } else {
      std::sort(indices, indices + n, compare);
    }

    absl::Span<const size_t> sorted_indices(indices, n);
    for (size_t k = 0; k < num_inputs; ++k) {
      std::byte* slice_base = slices[k].base;
      DispatchByElementSize(primitive_sizes[k], [&](auto elem) {
        using Elem = decltype(elem);
        PermuteSlice<Elem>(reinterpret_cast<Elem*>(slice_base), stride,
                           sorted_indices, reinterpret_cast<Elem*>(scratch));
      });
    }
  }
}

template <typename UIntKey>
void Sort1DUnsignedSlice(int64_t n, UIntKey* data, bool is_stable) {
  if (is_stable) {
    std::stable_sort(data, data + n, std::less<UIntKey>());
  } else {
    std::sort(data, data + n, std::less<UIntKey>());
  }
}

template <typename UIntKey>
void Sort1DUnsigned(const SortDims& sort_dims, int64_t start_slice,
                    int64_t end_slice, UIntKey* data, bool is_stable,
                    UIntKey xor_mask, UIntKey exp_mask,
                    bool (*fallback_less_than)(const void**)) {
  Sort1DSlices(
      sort_dims, start_slice, end_slice, data,
      [is_stable, xor_mask, exp_mask, fallback_less_than](int64_t n,
                                                          UIntKey* slice) {
        if (fallback_less_than == nullptr) {
          XorKeys(n, slice, xor_mask);
          Sort1DUnsignedSlice<UIntKey>(n, slice, is_stable);
          XorKeys(n, slice, xor_mask);
        } else if (ABSL_PREDICT_TRUE(
                       EncodeFloatKeys(n, slice, xor_mask, exp_mask))) {
          Sort1DUnsignedSlice<UIntKey>(n, slice, is_stable);
          DecodeFloatKeys(n, slice, xor_mask);
        } else {
          DecodeFloatKeys(n, slice, xor_mask);
          LessThan less_than = fallback_less_than;
          Sort1DSliceWithComparator<UIntKey>(n, slice, is_stable, &less_than);
        }
      });
}

template <typename UIntKey, typename Value>
void SortKeyValueSlice(int64_t n, UIntKey* keys, Value* values,
                       bool is_stable) {
  auto comp = [](const auto& a, const auto& b) -> bool {
    auto get_key = [](const auto& item) -> UIntKey {
      using T = std::decay_t<decltype(item)>;
      if constexpr (std::is_same_v<T, std::pair<UIntKey, Value>>) {
        return item.first;
      } else {
        return item.key;
      }
    };
    return get_key(a) < get_key(b);
  };

  ZipIterator<UIntKey, Value> begin(keys, values);
  ZipIterator<UIntKey, Value> end(keys + n, values + n);
  if (is_stable) {
    std::stable_sort(begin, end, comp);
  } else {
    std::sort(begin, end, comp);
  }
}

template <typename UIntKey, typename Value>
void Sort2DUnsignedKeyValue(const SortDims& sort_dims, int64_t start_slice,
                            int64_t end_slice, UIntKey* keys, Value* values,
                            bool is_stable, UIntKey xor_mask, UIntKey exp_mask,
                            bool (*fallback_less_than)(const void**)) {
  Sort2DSlices(sort_dims, start_slice, end_slice, keys, values,
               [is_stable, xor_mask, exp_mask, fallback_less_than](
                   int64_t n, UIntKey* k, Value* v) {
                 if (fallback_less_than == nullptr) {
                   XorKeys(n, k, xor_mask);
                   SortKeyValueSlice<UIntKey, Value>(n, k, v, is_stable);
                   XorKeys(n, k, xor_mask);
                 } else if (ABSL_PREDICT_TRUE(
                                EncodeFloatKeys(n, k, xor_mask, exp_mask))) {
                   SortKeyValueSlice<UIntKey, Value>(n, k, v, is_stable);
                   DecodeFloatKeys(n, k, xor_mask);
                 } else {
                   DecodeFloatKeys(n, k, xor_mask);
                   LessThan less_than = fallback_less_than;
                   Sort2DSliceWithComparator<UIntKey, Value>(n, k, v, is_stable,
                                                             &less_than);
                 }
               });
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

  if (data.size() == 1) {
    DispatchByElementSize(primitive_sizes[0], [&](auto elem) {
      using Elem = decltype(elem);
      Sort1DWithComparator<Elem>(sort_dims, start_slice, end_slice,
                                 reinterpret_cast<Elem*>(data[0]), is_stable,
                                 less_than);
    });
    return;
  }

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

  SortNDInplace(sort_dims, start_slice, end_slice, data, primitive_sizes,
                is_stable, less_than);
}

template <typename T>
void SortInplace(const SortDims& sort_dims, int64_t start_slice,
                 int64_t end_slice, T* data, bool is_stable,
                 SortDirection direction) {
  using UIntKey = UnsignedKeyType<sizeof(T)>;
  constexpr UIntKey kSignBit = UIntKey{1} << (sizeof(UIntKey) * 8 - 1);
  UIntKey dir_xor =
      direction == SortDirection::kAscending ? UIntKey{0} : ~UIntKey{0};
  UIntKey* udata = reinterpret_cast<UIntKey*>(data);

  if constexpr (std::is_integral_v<T>) {
    UIntKey xor_mask = (std::is_signed_v<T> ? kSignBit : UIntKey{0}) ^ dir_xor;
    Sort1DUnsigned<UIntKey>(sort_dims, start_slice, end_slice, udata, is_stable,
                            xor_mask, /*exp_mask=*/0,
                            /*fallback_less_than=*/nullptr);
  } else {
    bool (*fallback_less_than)(const void**) =
        direction == SortDirection::kAscending ? &LessThanWithNaN<T>
                                               : &GreaterThanWithNaN<T>;
    Sort1DUnsigned<UIntKey>(sort_dims, start_slice, end_slice, udata, is_stable,
                            dir_xor, FloatExpMask<T>(), fallback_less_than);
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
  using UIntKey = UnsignedKeyType<sizeof(Key)>;
  constexpr UIntKey kSignBit = UIntKey{1} << (sizeof(UIntKey) * 8 - 1);
  UIntKey dir_xor =
      direction == SortDirection::kAscending ? UIntKey{0} : ~UIntKey{0};
  UIntKey* ukeys = reinterpret_cast<UIntKey*>(keys);

  if constexpr (std::is_integral_v<Key>) {
    UIntKey xor_mask =
        (std::is_signed_v<Key> ? kSignBit : UIntKey{0}) ^ dir_xor;
    Sort2DUnsignedKeyValue<UIntKey, Value>(sort_dims, start_slice, end_slice,
                                           ukeys, values, is_stable, xor_mask,
                                           /*exp_mask=*/0,
                                           /*fallback_less_than=*/nullptr);
  } else {
    bool (*fallback_less_than)(const void**) =
        direction == SortDirection::kAscending ? &LessThanWithNaN<Key>
                                               : &GreaterThanWithNaN<Key>;
    Sort2DUnsignedKeyValue<UIntKey, Value>(
        sort_dims, start_slice, end_slice, ukeys, values, is_stable, dir_xor,
        FloatExpMask<Key>(), fallback_less_than);
  }
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
