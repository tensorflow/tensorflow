/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This file provides general C++ utility functions in TFLite.
// For example: Converting between `TfLiteIntArray`, `std::vector` and
// Flatbuffer vectors. These functions can't live in `context.h` since it's pure
// C.

#ifndef TENSORFLOW_LITE_UTIL_H_
#define TENSORFLOW_LITE_UTIL_H_

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {

// Memory allocation parameter used by ArenaPlanner.
// Clients (such as delegates) might look at this to ensure interop between
// TFLite memory & hardware buffers.
// NOTE: This only holds for tensors allocated on the arena.
constexpr int kDefaultTensorAlignment = 64;

// The prefix of Flex op custom code.
// This will be matched agains the `custom_code` field in `OperatorCode`
// Flatbuffer Table.
// WARNING: This is an experimental API and subject to change.
constexpr char kFlexCustomCodePrefix[] = "Flex";

// Checks whether the prefix of the custom name indicates the operation is an
// Flex operation.
bool IsFlexOp(const char* custom_name);

// Converts a `std::vector` to a `TfLiteIntArray`. The caller takes ownership
// of the returned pointer.
TfLiteIntArray* ConvertVectorToTfLiteIntArray(const std::vector<int>& input);

// Converts an array (of the given size) to a `TfLiteIntArray`. The caller
// takes ownership of the returned pointer, and must make sure 'dims' has at
// least 'ndims' elements.
TfLiteIntArray* ConvertArrayToTfLiteIntArray(int ndims, const int* dims);

// Checks whether a `TfLiteIntArray` and an int array have matching elements.
// The caller must guarantee that 'b' has at least 'b_size' elements.
bool EqualArrayAndTfLiteIntArray(const TfLiteIntArray* a, int b_size,
                                 const int* b);

size_t CombineHashes(std::initializer_list<size_t> hashes);

// Populates the size in bytes of a type into `bytes`. Returns kTfLiteOk for
// valid types, and kTfLiteError otherwise.
TfLiteStatus GetSizeOfType(TfLiteContext* context, const TfLiteType type,
                           size_t* bytes);

// Creates a stub TfLiteRegistration instance with the provided
// `custom_op_name`. The op will fail if invoked, and is useful as a
// placeholder to defer op resolution.
// Note that `custom_op_name` must remain valid for the returned op's lifetime..
TfLiteRegistration CreateUnresolvedCustomOp(const char* custom_op_name);

// Checks whether the provided op is an unresolved custom op.
bool IsUnresolvedCustomOp(const TfLiteRegistration& registration);

// Returns a descriptive name with the given op TfLiteRegistration.
std::string GetOpNameByRegistration(const TfLiteRegistration& registration);

// The prefix of a validation subgraph name.
// WARNING: This is an experimental API and subject to change.
constexpr char kValidationSubgraphNamePrefix[] = "VALIDATION:";

// Checks whether the prefix of the subgraph name indicates the subgraph is a
// validation subgraph.
bool IsValidationSubgraph(const char* name);

// Multiply two sizes and return true if overflow occurred;
// This is based off tensorflow/overflow.h but is simpler as we already
// have unsigned numbers. It is also generalized to work where sizeof(size_t)
// is not 8.
TfLiteStatus MultiplyAndCheckOverflow(size_t a, size_t b, size_t* product);

// Returns whether the TfLiteTensor is a resource or variant tensor.
inline bool IsResourceOrVariant(const TfLiteTensor* tensor) {
  return tensor->type == kTfLiteResource || tensor->type == kTfLiteVariant;
}

// Compute the number of bytes required to represent a tensor with dimensions
// specified by the array dims (of length dims_size). Returns the status code
// and bytes.
TfLiteStatus BytesRequired(TfLiteType type, const int* dims, size_t dims_size,
                           size_t* bytes, TfLiteContext* context);

// `unique_ptr` wrapper for `TfLiteTensor`s.
struct TfLiteTensorDeleter {
  void operator()(TfLiteTensor* t) {
    if (t) {
      TfLiteTensorFree(t);
    }
    free(t);
  }
};

using TensorUniquePtr = std::unique_ptr<TfLiteTensor, TfLiteTensorDeleter>;
TensorUniquePtr BuildTfLiteTensor();
TensorUniquePtr BuildTfLiteTensor(TfLiteType type, const std::vector<int>& dims,
                                  TfLiteAllocationType allocation_type);
TensorUniquePtr BuildTfLiteTensor(TfLiteType type, IntArrayUniquePtr dims,
                                  TfLiteAllocationType allocation_type);

int GetBuiltinDataSize(BuiltinOperator op);

#if !(defined(TFLITE_HAS_OVERFLOW_BUILTINS)) && \
    (defined(__GNUC__) || defined(__clang__))
#define TFLITE_HAS_OVERFLOW_BUILTINS 1
#endif

template <class I>
struct Widen;

#define TFLITE_DEFINE_WIDEN_TRAIT(BASE, WIDE) \
  template <>                                 \
  struct Widen<BASE> {                        \
    using type = WIDE;                        \
  }

TFLITE_DEFINE_WIDEN_TRAIT(int8_t, int16_t);
TFLITE_DEFINE_WIDEN_TRAIT(int16_t, int32_t);
TFLITE_DEFINE_WIDEN_TRAIT(int32_t, int64_t);

TFLITE_DEFINE_WIDEN_TRAIT(uint8_t, uint16_t);
TFLITE_DEFINE_WIDEN_TRAIT(uint16_t, uint32_t);
TFLITE_DEFINE_WIDEN_TRAIT(uint32_t, uint64_t);

#undef TFLITE_DEFINE_WIDEN_TRAIT

template <class T, class = void>
struct CanWiden : std::false_type {};

template <class T>
struct CanWiden<T, std::void_t<decltype(Widen<T>())>> : std::true_type {};

// Wraps an integer value and allows checking if standard arithmetic operations
// used to generate that value have over/underflowed.
template <class T>
class CheckedInt {
 public:
  using type = T;

  static_assert(std::is_integral_v<T>, "T must be an integral value.");

  CheckedInt() = default;
  CheckedInt(const CheckedInt&) = default;
  CheckedInt(CheckedInt&&) = default;
  CheckedInt& operator=(const CheckedInt&) = default;
  CheckedInt& operator=(CheckedInt&&) = default;

  template <class U>
  // NOLINTNEXTLINE(*-explicit-constructor)
  CheckedInt(U val) : value_(static_cast<T>(val)), overflow_(false) {
    if constexpr (std::is_same_v<T, U>) {
      // Don't do anything, the types are the same.
    } else if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {
      overflow_ = val < std::numeric_limits<T>::lowest() ||
                  val > std::numeric_limits<T>::max();
    } else if constexpr (std::is_unsigned_v<U> && std::is_signed_v<T>) {
      overflow_ = sizeof(T) <= sizeof(U) &&
                  val > static_cast<U>(std::numeric_limits<T>::max());
    } else {  // is_signed_v<U> && is_unsigned_v<T>.
      overflow_ = val < 0 || static_cast<std::make_unsigned_t<U>>(val) >
                                 std::numeric_limits<T>::max();
    }
  }

  template <class U>
  explicit CheckedInt(const CheckedInt<U>& other)
      : CheckedInt<T>(other.value_) {
    overflow_ |= other.overflow_;
  }

  template <class U>
  CheckedInt& operator=(const CheckedInt<U>& other) {
    *this = CheckedInt<T>(other.value_);
    overflow_ |= other.overflow_;
    return *this;
  }

  T Value() const noexcept { return value_; }

  bool Overflow() const noexcept { return overflow_; }

  TfLiteStatus Status() const noexcept {
    return overflow_ ? kTfLiteError : kTfLiteOk;
  }

  template <class U>
  CheckedInt& operator+=(const CheckedInt<U>& b) noexcept {
    return *this = *this + b;
  }

  template <class U>
  CheckedInt& operator-=(const CheckedInt<U>& b) noexcept {
    return *this = *this - b;
  }

  template <class U>
  CheckedInt& operator*=(const CheckedInt<U>& b) noexcept {
    return *this = *this * b;
  }

  template <class U>
  CheckedInt& operator/=(const CheckedInt<U>& b) noexcept {
    return *this = *this / b;
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator+=(U b) noexcept {
    return *this += CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator-=(U b) noexcept {
    return *this -= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator*=(U b) noexcept {
    return *this *= CheckedInt<U>(b);
  }

  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>
  CheckedInt& operator/=(U b) noexcept {
    return *this /= CheckedInt<U>(b);
  }

 private:
  // Helper constructor for operators.
  CheckedInt(T val, bool overflow) : value_(val), overflow_(overflow) {}

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator+(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator-(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator*(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class V, class U>
  friend CheckedInt<std::common_type_t<V, U>> operator/(
      const CheckedInt<V>& a, const CheckedInt<U>& b) noexcept;

  template <class U>
  friend class CheckedInt;

  template <class U>
  friend bool operator==(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a == b.Value();
  }

  template <class U>
  friend bool operator!=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return !(a == b);
  }

  template <class U>
  friend bool operator<(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return a < b.Value();
  }

  template <class U>
  friend bool operator<=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return a <= b.Value();
  }

  template <class U>
  friend bool operator>(const CheckedInt<T>& a,
                        const CheckedInt<U>& b) noexcept {
    return b < a.Value();
  }

  template <class U>
  friend bool operator>=(const CheckedInt<T>& a,
                         const CheckedInt<U>& b) noexcept {
    return b <= a.Value();
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(OP)                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(const CheckedInt<T>& a, U b) noexcept {        \
    return a OP CheckedInt<U>(b);                                        \
  }                                                                      \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>> \
  friend auto operator OP(U a, const CheckedInt<T>& b) noexcept {        \
    return CheckedInt<U>(a) OP b;                                        \
  }

#define TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(OP)                            \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(const CheckedInt<T>& a, U b) noexcept {             \
    if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {               \
      return a.Value() OP b;                                                  \
    } else if constexpr (std::is_signed_v<T>) {                               \
      return a.Value() >= 0 ? static_cast<std::make_unsigned_t<T>>(a.Value()) \
                                  OP b                                        \
                            : (0 OP 1);                                       \
    } else {                                                                  \
      return b >= 0 ? a.Value() OP static_cast<std::make_unsigned_t<U>>(b)    \
                    : (1 OP 0);                                               \
    }                                                                         \
  }                                                                           \
  template <class U, typename = std::enable_if_t<std::is_integral_v<U>>>      \
  friend bool operator OP(U a, const CheckedInt<T>& b) noexcept {             \
    if constexpr (std::is_signed_v<U> == std::is_signed_v<T>) {               \
      return a OP b.Value();                                                  \
    } else if constexpr (std::is_signed_v<U>) {                               \
      return a >= 0 ? static_cast<std::make_unsigned_t<U>>(a) OP b.Value()    \
                    : (0 OP 1);                                               \
    } else {                                                                  \
      return b.Value() >= 0                                                   \
                 ? a OP static_cast<std::make_unsigned_t<T>>(b.Value())       \
                 : (1 OP 0);                                                  \
    }                                                                         \
  }

  // NOLINTBEGIN(whitespace/operators)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(+)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(-)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(*)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_OP(/)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(==)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(!=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(<=)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>)
  TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP(>=)
  // NOLINTEND(whitespace/operators)
  //
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_OP
#undef TFLITE_OVERFLOW_AWARE_INT_MIXED_CMP_OP

 private:
  T value_{};
  bool overflow_ = false;
};

template <class T>
CheckedInt(T) -> CheckedInt<T>;

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator+(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
#if TFLITE_HAS_OVERFLOW_BUILTINS
  res.overflow_ = __builtin_add_overflow(a.value_, b.value_, &res.value_) ||
                  a.overflow_ || b.overflow_;
#else
  if constexpr (std::is_same_v<T, U>) {
    if constexpr (std::is_unsigned_v<T>) {
      res.value_ = a.value_ + b.value_;
      res.overflow_ = a.overflow_ || b.overflow_ || a.value_ > res.value_;
    } else {  // is_signed_v<T>
      // Signed overflow is undefined behaviour. We can only have an overflow
      // if the signs are the same. Because two's-complement arithmetic works
      // the same for signed and unsigned, we compute as unsigned and check if
      // the sign bit has changed.
      using Unsigned = std::make_unsigned_t<T>;
      const Unsigned ua = static_cast<Unsigned>(a.value_);
      const Unsigned ub = static_cast<Unsigned>(b.value_);
      const Unsigned tmp = ua + ub;
      constexpr Unsigned mask = static_cast<Unsigned>(1)
                                << (sizeof(Unsigned) * 8 - 1);
      const bool same_sign = ~(ua ^ ub) & mask;
      const bool sign_changed = (tmp ^ ua) & mask;
      res.value_ = static_cast<T>(tmp);
      res.overflow_ = a.overflow_ || b.overflow_ || (same_sign && sign_changed);
    }
  } else {
    // Convert elements to the common type first to check for implicit
    // conversion overflows.
    return CheckedInt<std::common_type_t<T, U>>(a) +
           CheckedInt<std::common_type_t<T, U>>(b);
  }
#endif
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator-(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
#if TFLITE_HAS_OVERFLOW_BUILTINS
  res.overflow_ = __builtin_sub_overflow(a.value_, b.value_, &res.value_) ||
                  a.overflow_ || b.overflow_;
#else
  if constexpr (std::is_same_v<T, U>) {
    if constexpr (std::is_unsigned_v<T>) {
      res.value_ = a.value_ - b.value_;
      res.overflow_ = a.overflow_ || b.overflow_ || a.value_ < res.value_;
    } else {  // is_signed_v<T>
      // Signed overflow is undefined behaviour. We can only have an overflow
      // if the sign opposite of b is the same as the sign of a. Because
      // two's-complement arithmetic works the same for signed and unsigned,
      // we compute as unsigned and check if the sign bit has changed.
      using Unsigned = std::make_unsigned_t<T>;
      const Unsigned ua = static_cast<Unsigned>(a.value_);
      const Unsigned ub = static_cast<Unsigned>(b.value_);
      const Unsigned tmp = ua - ub;
      constexpr Unsigned mask = static_cast<Unsigned>(1)
                                << (sizeof(Unsigned) * 8 - 1);
      const bool same_sign = ~(ua ^ ~ub) & mask;
      const bool sign_changed = (tmp ^ ua) & mask;
      res.value_ = static_cast<T>(tmp);
      res.overflow_ = a.overflow_ || b.overflow_ || (same_sign && sign_changed);
    }
  } else {
    // Convert elements to the common type first to check for implicit
    // conversion overflows.
    return CheckedInt<std::common_type_t<T, U>>(a) -
           CheckedInt<std::common_type_t<T, U>>(b);
  }
#endif
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator*(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  CheckedInt<std::common_type_t<T, U>> res;
#if TFLITE_HAS_OVERFLOW_BUILTINS
  res.overflow_ = __builtin_mul_overflow(a.value_, b.value_, &res.value_) ||
                  a.overflow_ || b.overflow_;
#else
  if constexpr (std::is_same_v<T, U>) {
    using C = decltype(res.value_);
    if constexpr (CanWiden<C>::value) {
      using W = typename Widen<C>::type;
      const W wa = static_cast<W>(a.value_);
      const W wb = static_cast<W>(b.value_);
      const W tmp = wa * wb;
      res = tmp;
      res.overflow_ |= a.overflow_ || b.overflow_;
    } else {
      static_assert(sizeof(C) == sizeof(uint64_t));
      const uint64_t ua = static_cast<uint64_t>(a.value_);
      const uint64_t ub = static_cast<uint64_t>(b.value_);
#define hi(x) (x >> 32)
#define lo(x) (x & 0xffffffff)
      const uint64_t hia = hi(ua);
      const uint64_t loa = lo(ua);
      const uint64_t hib = hi(ub);
      const uint64_t lob = lo(ub);

      const uint64_t lo_lo = loa * lob;
      const uint64_t hi_lo = hia * lob;
      const uint64_t lo_hi = loa * hib;
      const uint64_t hi_hi = hia * hib;

      const uint64_t cross = hi(lo_lo) + lo(hi_lo) + lo_hi;
      uint64_t upper_64 = hi_hi + hi(hi_lo) + hi(cross);
#undef hi
#undef lo
      if constexpr (std::is_signed_v<C>) {
        // It took a while to understand this.
        //
        // If a < 0, then ua = a + 2^64.
        // So ua * ub = (a + 2^64) * ub
        //            = a * ub + ub * 2^64
        //                       ~~~~~~~~~
        // This means that the upper_64 that we compute above has an extra ub
        // added to it that we need to remove.
        //
        // The same is applied to b below.
        if (a.value_ < 0) {
          upper_64 -= ub;
        }
        if (b.value_ < 0) {
          upper_64 -= ua;
        }
        const uint64_t lower_64 = ua * ub;
        const uint64_t sign_ext =
            static_cast<uint64_t>(static_cast<int64_t>(lower_64) >> 63);
        res.overflow_ = a.overflow_ || b.overflow_ || (upper_64 != sign_ext);
      } else {
        res.overflow_ = a.overflow_ || b.overflow_ || (upper_64 != 0);
      }
      res.value_ = a.value_ * b.value_;
    }
  } else {
    return CheckedInt<std::common_type_t<T, U>>(a) *
           CheckedInt<std::common_type_t<T, U>>(b);
  }
#endif
  return res;
}

template <class T, class U>
CheckedInt<std::common_type_t<T, U>> operator/(
    const CheckedInt<T>& a, const CheckedInt<U>& b) noexcept {
  using C = std::common_type_t<T, U>;
  using limits = std::numeric_limits<C>;
  if constexpr (std::is_same_v<T, U>) {
    if constexpr (std::is_signed_v<C>) {
      if (a.value_ == limits::lowest() && b.value_ == -1) {
        return {/*val=*/limits::max(), /*overflow=*/true};
      }
    }
    return {/*val=*/b.value_ != 0 ? static_cast<C>(a.value_ / b.value_)
                                  : limits::max(),
            /*overflow=*/b.value_ == 0 || a.overflow_ || b.overflow_};
  } else {
    return CheckedInt<std::common_type_t<T, U>>(a) /
           CheckedInt<std::common_type_t<T, U>>(b);
  }
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_UTIL_H_
