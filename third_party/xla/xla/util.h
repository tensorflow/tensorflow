/* Copyright 2017 The OpenXLA Authors.

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

// Generally useful utility functions that are common to (not specific to any
// given part of) the XLA code base.

#ifndef XLA_UTIL_H_
#define XLA_UTIL_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/log_severity.h"
#include "absl/base/macros.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "xla/status_macros.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/bfloat16.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"  // IWYU pragma: keep
#include "tsl/platform/logging.h"
#include "tsl/platform/ml_dtypes.h"

namespace xla {

// Converts the unsigned integer n into a mixed-radix representation with the
// given bounds (radices). More precisely, if there are K radices, then the
// returned vector digits has K entries and satisfies
//
//   0 <= digits[i] < bounds[i],  for i = 0, ..., K - 1
//
// and FromMixedRadix(digits) == n. The mixed radix representation is unique
// modulo the product of the entries of bounds.
std::vector<int64_t> ToMixedRadix(int64_t n, absl::Span<const int64_t> bounds);

// Logs the provided status message with a backtrace.
//
// For use by absl::Status-factories, logs a backtrace at the point where the
// status is created, such that we can use --vmodule=util=1 to see all status
// creation backtraces.
absl::Status WithLogBacktrace(const absl::Status& status);

// Ranks greater than 6 are very rare, so use InlinedVector<int64_t, 6> to store
// the bounds and indices. And for the rare cases of ranks greater than 6,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
inline constexpr int InlineRank() { return 6; }
using DimensionVector = absl::InlinedVector<int64_t, InlineRank()>;
using DimLevelTypeVector = absl::InlinedVector<DimLevelType, InlineRank()>;

// RAII timer that logs with a given label the wall clock time duration in human
// readable form. This differs from base's ElapsedTimer primarily in that it
// spits out the human-readable duration form.
//
// Keeps track of global maximum and cumulative times across all invocations.
//
// By default, the timing traces are only printed at VLOG(1) and above:
//
//   XLA_SCOPED_LOGGING_TIMER("fooing bar");  // nop if !VLOG_IS_ON(1).
//
// but you can control this via:
//
//   XLA_SCOPED_LOGGING_TIMER_LEVEL("fooing bar", 2);  // nop if !VLOG_IS_ON(2)
//
#define XLA_SCOPED_LOGGING_TIMER(label) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, 1, __COUNTER__, /*condition=*/true)
#define XLA_SCOPED_LOGGING_TIMER_LEVEL(label, level) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, __COUNTER__, /*condition=*/true)
// The timer trace is only printed if the condition is true.
#define XLA_SCOPED_LOGGING_TIMER_IF(label, condition) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, 1, __COUNTER__, (condition))

// Helper for implementing macros above.  Do not use directly.
//
// Forces the evaluation of "counter", which we expect is equal to __COUNTER__.
#define XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, counter, condition) \
  XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter, (condition))

// Helper for macros above.  Don't use directly.
#define XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter, condition)     \
  static ::xla::TimerStats XLA_TimerStats##counter;                            \
  ::xla::ScopedLoggingTimer XLA_ScopedLoggingTimerInstance##counter(           \
      label, /*enabled=*/VLOG_IS_ON(level) && (condition), __FILE__, __LINE__, \
      &XLA_TimerStats##counter);

struct TimerStats {
  absl::Mutex stats_mutex;
  double cumulative_secs ABSL_GUARDED_BY(stats_mutex) = 0;
  double max_secs ABSL_GUARDED_BY(stats_mutex) = 0;
  uint64_t times_called ABSL_GUARDED_BY(stats_mutex) = 0;
};

// RAII timer for XLA_SCOPED_LOGGING_TIMER and XLA_SCOPED_LOGGING_TIMER_LEVEL
// macros above.  Recommended usage is via the macros so you don't have to give
// the timer a name or worry about calling VLOG_IS_ON yourself.
class ScopedLoggingTimer {
 public:
  // label: Label to display for logging.
  // enabled: Whether this timer should do anything at all.
  // file: Filename to display in logging.
  // line: Line number to display in logging.
  // `timer_stats`: unowned non-null pointer which is used to populate the
  // global timer statistics.
  ScopedLoggingTimer(absl::string_view label, bool enabled, const char* file,
                     int line, TimerStats* timer_stats);

  // Stop the timer and log the tracked time. Timer is disabled after this
  // function is called.
  void StopAndLog();

  ~ScopedLoggingTimer();

 private:
  const std::string label_;
  const char* const file_;
  const int line_;
  TimerStats* const timer_stats_;
  uint64_t start_micros_;
  bool enabled_;
};

// Turns an immutable slice of type T into an immutable slice of bytes with the
// same byte size.
template <typename T>
absl::Span<const uint8_t> CastToByteSlice(absl::Span<const T> slice) {
  return absl::Span<const uint8_t>(
      reinterpret_cast<const uint8_t*>(slice.data()), slice.size() * sizeof(T));
}

// Casts a byte slice to a non-byte type T, checking that the original slice
// length is a multiple of sizeof(T).
template <typename T>
absl::Span<const T> CastByteSlice(absl::Span<const uint8_t> slice) {
  CHECK_EQ(0, slice.size() % sizeof(T));
  return absl::Span<const T>(reinterpret_cast<const T*>(slice.data()),
                             slice.size() / sizeof(T));
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using their
// operator==. Like std::equal, but forces size equality.
template <typename Container1T,
          typename ElementType = typename Container1T::value_type>
bool ContainersEqual(const Container1T& c1,
                     std::initializer_list<ElementType> il) {
  absl::Span<const ElementType> c2{il};
  return absl::c_equal(c1, c2);
}

#if defined(__cpp_lib_to_underlying) && __cpp_lib_to_underlying >= 202102L
using to_underlying = std::to_underlying;
#else
// Helper function which implements C++23's std::to_underlying.
template <typename T>
constexpr std::underlying_type_t<T> to_underlying(T value) noexcept {
  return static_cast<std::underlying_type_t<T>>(value);
}
#endif

// Performs a copy of count values from src to dest, using different strides for
// source and destination. The source starting index is src_base, while the
// destination one is dest_base.
template <typename D, typename S>
void StridedCopy(D* dest, int64_t dest_stride, const S* src, int64_t src_stride,
                 int64_t count) {
  const S* src_end = src + count * src_stride;
  DCHECK_LT(src, src_end);
  for (; src < src_end; dest += dest_stride, src += src_stride) {
    *dest = static_cast<D>(*src);
  }
}

// Adds some context information to the error message in a
// absl::Status.  This is useful as absl::Statuses are
// propagated upwards.
absl::Status AddStatus(absl::Status prior, absl::string_view context);
absl::Status AppendStatus(absl::Status prior, absl::string_view context);

// The following three macros define a common set of code for creating
// absl::Status errors with the given error_type, with the addition of adding
// absl::SourceLocation if it's available (PLATFORM_GOOGLE).  They're a
// complicated by the need to use #ifdefs within the code.  This would be the
// equivalent code for ResourceExhausted if a #define macro could have embedded
// #ifdef directives:
//
// template <typename... Args>
// struct ResourceExhausted {
//   absl::Status status;
// #if defined(PLATFORM_GOOGLE)
//   // NOLINTNEXTLINE(google-explicit-constructor)
//   ResourceExhausted(const absl::FormatSpec<Args...>& format, Args&&... args,
//                     absl::SourceLocation loc =
//                     absl::SourceLocation::current())
//       : status(WithLogBacktrace(
//             absl::ResourceExhaustedError(absl::StrFormat(format, args...))
//                 .WithSourceLocation(loc))) {}
// #else
//   ResourceExhaustedStrCat(Args&&... concat)
//       : status(WithLogBacktrace(
//             absl::ResourceExhaustedError(absl::StrFormat(format, args...)))
//             {}
// #endif
//
//   // NOLINTNEXTLINE(google-explicit-constructor)
//   operator absl::Status() const { return status; }
// };
//
#define XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_PREFIX(error_type) \
  template <typename... Args>                                     \
  struct error_type {                                             \
    absl::Status status;
#define XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_SUFFIX(error_type)        \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                      \
  operator absl::Status() const { return status; }                       \
  }                                                                      \
  ;                                                                      \
  /*Deduction guide to make variadic arguments play nice with default */ \
  /* absl::SourceLocation argument. */                                   \
  template <typename... Args>                                            \
  error_type(const absl::FormatSpec<Args...>& format,                    \
             Args&&...) -> error_type<Args...>;

#if defined(PLATFORM_GOOGLE)
#define XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(error_type)               \
  XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_PREFIX(error_type)              \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                      \
  error_type(const absl::FormatSpec<Args...>& format, Args&&... args,    \
             absl::SourceLocation loc = absl::SourceLocation::current()) \
      : status(WithLogBacktrace(                                         \
            absl::error_type##Error(absl::StrFormat(format, args...))    \
                .WithSourceLocation(loc))) {}                            \
  XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_SUFFIX(error_type)
#else
#define XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(error_type)          \
  template <typename... Args>                                       \
  absl::Status error_type(const absl::FormatSpec<Args...>& format,  \
                          const Args&... args) {                    \
    return WithLogBacktrace(                                        \
        absl::error_type##Error(absl::StrFormat(format, args...))); \
  }
#endif

XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(Cancelled);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(FailedPrecondition);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(Internal);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(InvalidArgument);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(NotFound);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(ResourceExhausted);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(Unavailable);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(Unimplemented);
XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE(Unknown);

#undef XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE
#undef XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_PREFIX
#undef XLA_ERROR_WITH_STRFORMAT_AND_BACKTRACE_SUFFIX

// The following three macros define a common set of code for creating
// absl::Status errors with the given error_type, with the addition of adding
// absl::SourceLocation if it's available (PLATFORM_GOOGLE).  They're a
// complicated by the need to use #ifdefs within the code.  This would be the
// equivalent code for ResourceExhausted if a #define macro could have embedded
// #ifdef directives:
//
// template <typename... Args>
// struct ResourceExhaustedStrCat {
//   absl::Status status;
// #if defined(PLATFORM_GOOGLE)
//   // NOLINTNEXTLINE(google-explicit-constructor)
//   ResourceExhaustedStrCat(Args&&... concat, absl::SourceLocation loc =
//                                             absl::SourceLocation::current())
//       : status(WithLogBacktrace(
//             absl::ResourceExhaustedError(absl::StrCat(
//                                          std::forward<Args>(concat)...))
//                 .WithSourceLocation(loc))) {}
// #else
//   ResourceExhaustedStrCat(Args&&... concat)
//       : status(WithLogBacktrace(
//             absl::ResourceExhaustedError(absl::StrCat(
//                                          std::forward<Args>(concat)...))))
//             {}
// #endif
//
//   // NOLINTNEXTLINE(google-explicit-constructor)
//   operator absl::Status() const { return status; }
// };
//
#define XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_PREFIX(error_type) \
  template <typename... Args>                                  \
  struct error_type##StrCat {                                  \
    absl::Status status;                                       \
    /* NOLINTNEXTLINE(google-explicit-constructor) */
#define XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_SUFFIX(error_type)           \
  /* NOLINTNEXTLINE(google-explicit-constructor) */                      \
  operator absl::Status() const { return status; }                       \
  }                                                                      \
  ;                                                                      \
  /*Deduction guide to make variadic arguments play nice with default */ \
  /* absl::SourceLocation argument. */                                   \
  template <typename... Args>                                            \
  error_type##StrCat(Args&&...)->error_type##StrCat<Args...>;

#if defined(PLATFORM_GOOGLE)
#define XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(error_type)                       \
  XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_PREFIX(error_type)                      \
  error_type##StrCat(Args&&... concat, absl::SourceLocation loc =             \
                                           absl::SourceLocation::current())   \
      : status(                                                               \
            WithLogBacktrace(absl::error_type##Error(                         \
                                 absl::StrCat(std::forward<Args>(concat)...)) \
                                 .WithSourceLocation(loc))) {}                \
  XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_SUFFIX(error_type)
#else
#define XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(error_type)       \
  XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_PREFIX(error_type)      \
  error_type##StrCat(Args&&... concat)                        \
      : status(WithLogBacktrace(absl::error_type##Error(      \
            absl::StrCat(std::forward<Args>(concat)...)))) {} \
  XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_SUFFIX(error_type)
#endif

XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(ResourceExhausted);
XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(InvalidArgument);
XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(Unimplemented);
XLA_ERROR_WITH_STRCAT_AND_BACKTRACE(Internal);

#undef XLA_ERROR_WITH_STRCAT_AND_BACKTRACE
#undef XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_PREFIX
#undef XLA_ERROR_WITH_STRCAT_AND_BACKTRACE_SUFFIX

// Splits the lines of the original, replaces leading whitespace with the prefix
// given by "indentation", and returns the string joined by newlines again. As a
// side effect, any additional trailing whitespace is removed.
//
// Note: even different amounts of leading whitespace on different lines will be
// uniformly replaced with "indentation".
std::string Reindent(absl::string_view original, absl::string_view indentation);

template <typename Container>
int64_t PositionInContainer(const Container& container, int64_t value) {
  return std::distance(container.begin(), absl::c_find(container, value));
}

// Formats the container as a comma-separated string. StrAppend must support
// appending the elements of the container. Prefix is prepended and suffix is
// appended to the returned string.
template <typename Container>
std::string CommaSeparatedString(const Container& c, const char* prefix = "",
                                 const char* suffix = "") {
  // Not using Join() since the implementation here is simple anyway and this
  // avoids copying the string to append prefix.
  std::string comma_separated = prefix;
  const char* separator = "";
  for (const auto& entry : c) {
    absl::StrAppend(&comma_separated, separator, entry);
    separator = ", ";
  }
  comma_separated += suffix;
  return comma_separated;
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
std::string CommaSeparatedString(const std::initializer_list<T>& c,
                                 const char* prefix = "",
                                 const char* suffix = "") {
  return CommaSeparatedString<std::initializer_list<T>>(c, prefix, suffix);
}

// Formats the container in the mathematical notation for a vector, e.g. (1, 3,
// 7). StrAppend must support appending the elements of c.
template <typename Container>
std::string VectorString(const Container& c) {
  return CommaSeparatedString(c, "(", ")");
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
std::string VectorString(const std::initializer_list<T>& c) {
  return VectorString<std::initializer_list<T>>(c);
}

// Returns a string which can losslessly round trip to a float8 E5M2.
std::string RoundTripFpToString(tsl::float8_e5m2 value);

// Returns a string which can losslessly round trip to a float8 E4M3.
std::string RoundTripFpToString(tsl::float8_e4m3fn value);

// Returns a string which can losslessly round trip to a float8 E4M3B11.
std::string RoundTripFpToString(tsl::float8_e4m3b11fnuz value);

// Returns a string which can losslessly round trip to a float8 E5M2FNUZ.
std::string RoundTripFpToString(tsl::float8_e5m2fnuz value);

// Returns a string which can losslessly round trip to a float8 E4M3FNUZ.
std::string RoundTripFpToString(tsl::float8_e4m3fnuz value);

// Returns a string which can losslessly round trip to a bfloat.
std::string RoundTripFpToString(tsl::bfloat16 value);

// Returns a string which can losslessly round trip to a fp16.
std::string RoundTripFpToString(Eigen::half value);

// Returns a string which can losslessly round trip to a float.
std::string RoundTripFpToString(float value);

// Returns a string which can losslessly round trip to a double.
std::string RoundTripFpToString(double value);

// Returns a PaddingConfig object that represents no padding for the given rank.
PaddingConfig MakeNoPaddingConfig(int64_t rank);

// Returns a PaddingConfig object where 'padding' contains
// (low edge padding, high edge padding) pairs for each dimension.
PaddingConfig MakeEdgePaddingConfig(
    absl::Span<const std::pair<int64_t, int64_t>> padding);

// Returns true if the padding configuration has at least one dimension with
// non-zero interior padding.
bool HasInteriorPadding(const PaddingConfig& config);

// Imports the templated FloorOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T FloorOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::FloorOfRatio<T>(dividend, divisor);
}

// Imports the templated CeilOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
constexpr T CeilOfRatio(T dividend, T divisor) {
  return tsl::MathUtil::CeilOfRatio<T>(dividend, divisor);
}

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpTo(13, 8) => 16
template <typename T>
constexpr T RoundUpTo(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

// Rounds the value down to a multiple of the divisor by first calling
// FloorOfRatio then multiplying by the divisor. For example:
// RoundDownTo(13, 8) => 8
template <typename T>
constexpr T RoundDownTo(T value, T divisor) {
  return FloorOfRatio(value, divisor) * divisor;
}

template <typename T>
struct DivMod {
  T quotient;
  T modulo;
};

// Divide `dividend` by `divisor` such that the quotient is rounded towards
// negative infinity. The remainder will have the same sign as `divisor`.
template <typename T>
constexpr DivMod<T> FloorDivMod(T dividend, T divisor) {
  DivMod<T> div_mod;
  div_mod.quotient = FloorOfRatio(dividend, divisor);
  div_mod.modulo = dividend - div_mod.quotient * divisor;
  return div_mod;
}

// Given a number of flops executed in an amount of time, produces a string that
// represents the throughput;
// e.g. HumanReadableNumFlops(1e9, 1e9) => 1.00GFLOP/s.
std::string HumanReadableNumFlops(double flops, double nanoseconds);

// Given a number of transcendental ops executed in an amount of time, produces
// a string that represents the throughput;
// e.g. HumanReadableNumTranscendentalOps(1e9, 1e9) => 1.00GTROP/s.
std::string HumanReadableNumTranscendentalOps(double trops, double nanoseconds);

// Split the text into multiple lines and log each line with the given
// severity, filename, and line number.
void LogLines(absl::LogSeverity sev, absl::string_view text, const char* fname,
              int lineno);
inline void LogLinesINFO(absl::string_view text, const char* fname,
                         int lineno) {
  return LogLines(absl::LogSeverity::kInfo, text, fname, lineno);
}
inline void LogLinesWARNING(absl::string_view text, const char* fname,
                            int lineno) {
  return LogLines(absl::LogSeverity::kWarning, text, fname, lineno);
}
inline void LogLinesERROR(absl::string_view text, const char* fname,
                          int lineno) {
  return LogLines(absl::LogSeverity::kError, text, fname, lineno);
}
inline void LogLinesFATAL(absl::string_view text, const char* fname,
                          int lineno) {
  return LogLines(absl::LogSeverity::kFatal, text, fname, lineno);
}

// Returns a mask with "width" number of least significant bits set.
template <typename T>
constexpr inline T LsbMask(int width) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  ABSL_ASSERT(width >= 0);
  ABSL_ASSERT(width <= std::numeric_limits<T>::digits);
  return width == 0
             ? 0
             : static_cast<T>(-1) >> (std::numeric_limits<T>::digits - width);
}

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Floor(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return absl::bit_width(x) - 1;
}

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
template <typename T>
constexpr inline int Log2Ceiling(T x) {
  static_assert(std::is_unsigned<T>::value,
                "T should be an unsigned integer type");
  return x == 0 ? -1 : absl::bit_width(x - 1);
}

// Return the number of sign bits (i.e. the number of leading ones for negative
// numbers and the number of leading zeros for non-negative numbers).
template <typename T>
constexpr inline int CountLeadingSignBits(T x) {
  static_assert(std::is_signed<T>::value, "T should be a signed integer type");
  using UnsignedType = std::make_unsigned_t<T>;
  return x < T{0} ? absl::countl_one<UnsignedType>(x)
                  : absl::countl_zero<UnsignedType>(x);
}

// Returns `value` with the low `width` bits set and the remaining bits set to
// zero.
template <typename T>
constexpr inline T KeepLowerBits(T value, int width) {
  return value & LsbMask<T>(width);
}

// Returns `base` multiplied by itself `exponent` number of times.
//
// Note: returns 1 when `exponent` is zero.
// Precondition: `exponent` is non-negative for integral `T`.
template <typename T, typename ExpType>
constexpr T IPow(T base, ExpType exponent) {
  static_assert(std::numeric_limits<ExpType>::is_integer);
  if constexpr (std::numeric_limits<T>::is_integer) {
    // A negative `exponent` is indicative of a logic bug for integral `base`.
    // We disallow it for floating-point types for symmetry.
    ABSL_ASSERT(exponent >= 0);
  }
  const bool take_reciprocal = exponent < 0;
  // We use the right-to-left binary exponentiation algorithm.
  T result(1);
  for (;;) {
    if ((exponent & 1) != 0) {
      result *= base;
    }
    exponent /= 2;
    if (exponent == 0) {
      break;
    }
    base *= base;
  }
  if constexpr (std::numeric_limits<ExpType>::is_signed) {
    if (take_reciprocal) {
      return T(1) / result;
    }
  }
  return result;
}

// UnsignedIntegerTypeForSize<N> gets an unsigned integer with the given size in
// bytes.
template <size_t>
struct UnsignedIntegerTypeForSize;

template <>
struct UnsignedIntegerTypeForSize<1> {
  using type = uint8_t;
};

template <>
struct UnsignedIntegerTypeForSize<2> {
  using type = uint16_t;
};

template <>
struct UnsignedIntegerTypeForSize<4> {
  using type = uint32_t;
};

template <>
struct UnsignedIntegerTypeForSize<8> {
  using type = uint64_t;
};

template <size_t kBytes>
using UnsignedIntegerTypeForSizeType =
    typename UnsignedIntegerTypeForSize<kBytes>::type;

template <size_t kBytes>
using SignedIntegerTypeForSizeType =
    std::make_signed_t<UnsignedIntegerTypeForSizeType<kBytes>>;

template <typename T>
auto SignAndMagnitude(T x) {
  using BitType = UnsignedIntegerTypeForSizeType<sizeof(T)>;
  BitType x_abs_bits = Eigen::numext::bit_cast<BitType>(Eigen::numext::abs(x));
  const BitType x_bits = Eigen::numext::bit_cast<BitType>(x);
  const BitType x_sign = x_bits ^ x_abs_bits;
  if constexpr (!has_negative_zero_v<T>) {
    //  f8e4m3b11, f8e4m3fnuz, and f8e5m2fnuz don't support -0, adjust negative
    //  numbers to fill in the gap.
    if (x_sign) {
      x_abs_bits -= 1;
    }
  }
  return std::make_pair(x_sign, x_abs_bits);
}

template <typename T>
auto SignAndMagnitudeToTwosComplement(T sign, T magnitude) {
  static_assert(!std::numeric_limits<T>::is_signed);
  using SignedType = std::make_signed_t<T>;
  return static_cast<SignedType>(magnitude) ^
         (static_cast<SignedType>(sign) < 0 ? SignedType{-1} : SignedType{0});
}

// Returns the signed magnitude of T.
template <typename T>
auto ToSignMagnitude(T input) {
  auto [sign, magnitude] = SignAndMagnitude(input);
  return SignAndMagnitudeToTwosComplement(sign, magnitude);
}

template <typename T>
constexpr int NanPayloadBits() {
  // Floating point types with signaling NaNs have payloads.
  if constexpr (!std::numeric_limits<T>::has_signaling_NaN) {
    return 0;
  }
  return std::numeric_limits<T>::digits - 1;
}

template <typename T>
constexpr uint64_t QuietNanWithoutPayload() {
  constexpr int bits = NanPayloadBits<T>();
  if constexpr (bits > 0) {
    return uint64_t{1} << (bits - 1);
  }
  return 0;
}

template <typename T>
constexpr uint64_t NanPayloadBitMask() {
  constexpr int bits = NanPayloadBits<T>();
  if constexpr (bits > 0) {
    return LsbMask<uint64_t>(bits);
  }
  return 0;
}

template <typename T>
T NanWithSignAndPayload(bool sign, uint64_t nan_payload) {
  static_assert(NanPayloadBits<T>() > 0);
  using RepT = UnsignedIntegerTypeForSizeType<sizeof(T)>;
  // Clear the sign bit.
  T val = Eigen::numext::abs(std::numeric_limits<T>::quiet_NaN());
  // Conditionally set the sign bit.
  if (sign) {
    val = -val;
  }
  auto rep = absl::bit_cast<RepT>(val);
  rep |= uint64_t{sign} << (std::numeric_limits<RepT>::digits - 1);
  constexpr int kPayloadBits = NanPayloadBits<T>();
  if (kPayloadBits > 0) {
    // Clear rep's NaN payload.
    rep &= ~NanPayloadBitMask<T>();
    CHECK_NE(nan_payload, 0);
    rep |= nan_payload;
  }
  return absl::bit_cast<T>(rep);
}

// Utility for performing a down_cast<> on a std::unique_ptr<>.
template <typename Derived, typename Base>
std::unique_ptr<Derived> unique_ptr_down_cast(std::unique_ptr<Base> ptr) {
  return absl::WrapUnique(tensorflow::down_cast<Derived*>(ptr.release()));
}

int64_t Product(absl::Span<const int64_t> xs);

// Returns an array of results after performing elementwise product of a and b.
std::vector<int64_t> ElemwiseProduct(absl::Span<const int64_t> a,
                                     absl::Span<const int64_t> b);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < CommonFactors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `CommonFactors(a, b)[CommonFactors(a, b).size - 1] = (a.size, b.size)`
//
// If input and output are the same, return {(0, 0), {1, 1}, ... {a.size,
// b.size}}, otherwise if the given shapes have non-zero size, returns the
// bounds of the shortest possible such subsequences; else, returns `{(0, 0),
// (a.size, b.size)}`.
absl::InlinedVector<std::pair<int64_t, int64_t>, 8> CommonFactors(
    absl::Span<const int64_t> a, absl::Span<const int64_t> b);

struct ConvertedDimensionNumbers {
  DimensionVector transformed_from_dimensions;
  DimensionVector untransformed_from_dimensions;
  DimensionVector to_dimensions;
  DimensionVector split_from_dimensions;
  DimensionVector split_from_sizes;
  DimensionVector split_to_dimensions;
};

// Convert and unsorted list of dimensions from one shapes dimension sizes to
// another shapes dimensions sizes.
ConvertedDimensionNumbers ConvertDimensionNumbers(
    absl::Span<const int64_t> from_dimensions,
    absl::Span<const int64_t> from_sizes, absl::Span<const int64_t> to_sizes);

// Removes illegal characters from filenames.
std::string SanitizeFileName(std::string file_name);

// Check that a sequence of distinct numbers can form a continuous interval.
bool DistinctNumbersAreConsecutiveIfSorted(absl::Span<const int64_t>);

template <typename C, typename Value>
int64_t FindIndex(const C& c, Value&& value) {
  auto it = absl::c_find(c, std::forward<Value>(value));
  return std::distance(c.begin(), it);
}

template <typename C, typename Value>
void InsertAt(C* c, int64_t index, Value&& value) {
  c->insert(c->begin() + index, std::forward<Value>(value));
}

template <typename C>
void EraseAt(C* c, int64_t index) {
  c->erase(c->begin() + index);
}

template <typename T>
std::vector<T> SpanToVector(absl::Span<const T> slice) {
  return std::vector<T>(slice.begin(), slice.end());
}

template <typename T, size_t N>
std::vector<T> InlinedVectorToVector(
    const absl::InlinedVector<T, N>& inlined_vector) {
  return std::vector<T>(inlined_vector.begin(), inlined_vector.end());
}

// Returns true if `x` fits in 32-bits.
template <typename T>
bool IsInt32(T x) {
  // Following conversion rules: "the value is unchanged if it can be
  // represented in the destination type (and bit-field width); otherwise, the
  // value is implementation-defined."
  return static_cast<int32_t>(x) == x;
}

template <typename T>
absl::Status EraseElementFromVector(std::vector<T>* container, const T& value) {
  // absl::c_find returns a const_iterator which does not seem to work on
  // gcc 4.8.4, and this breaks the ubuntu/xla_gpu build bot.
  auto it = std::find(container->begin(), container->end(), value);
  TF_RET_CHECK(it != container->end());
  container->erase(it);
  return absl::OkStatus();
}

// Takes a sequence of unpacked n-bit values, such that every byte stores one
// value in the low-order bits, and packs them so every byte stores as many
// which will fit. `output` should have ceil((input.size()*kBitsPerElement)/8)
// bytes. The high-order bits of each byte in `input` are ignored.
template <size_t kBitsPerElement>
void PackIntN(absl::Span<const char> input, absl::Span<char> output) {
  constexpr auto kElementsPerByte = 8 / kBitsPerElement;
  const size_t aligned_inputs = input.size() / kElementsPerByte;
  for (size_t i = 0; i < aligned_inputs; ++i) {
    char byte = 0;
    for (size_t j = 0; j < kElementsPerByte; ++j) {
      byte |=
          (input[i * kElementsPerByte + j] & LsbMask<uint8_t>(kBitsPerElement))
          << (kBitsPerElement * (kElementsPerByte - j - 1));
    }
    output[i] = byte;
  }
  if (size_t remainder = input.size() % kElementsPerByte; remainder != 0) {
    char byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (input[aligned_inputs * kElementsPerByte + j] &
               LsbMask<uint8_t>(kBitsPerElement))
              << (kBitsPerElement * (kElementsPerByte - j - 1));
    }
    output[aligned_inputs] = byte;
  }
}

inline void PackIntN(int bits_per_element, absl::Span<const char> input,
                     absl::Span<char> output) {
  if (bits_per_element == 2) {
    PackIntN<2>(input, output);
  } else if (bits_per_element == 4) {
    PackIntN<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

// Takes a sequence of packed values, such that every byte stores multiple
// values, and unpacks them so every byte stores one value in the low-order
// bits. `input` should have
// ceil(output.size()*8/kBitsPerElement) bytes. The high-order bits in each
// output are zero.
template <size_t kBitsPerElement>
void UnpackIntN(absl::Span<const char> input, absl::Span<char> output) {
  constexpr auto kElementsPerByte = 8 / kBitsPerElement;
  const size_t aligned_outputs = output.size() / kElementsPerByte;
  for (size_t i = 0; i < aligned_outputs; ++i) {
    const char byte = input[i];
    for (int j = 0; j < kElementsPerByte; ++j) {
      output[i * kElementsPerByte + j] =
          (byte >> (kBitsPerElement * (kElementsPerByte - j - 1))) &
          LsbMask<uint8_t>(kBitsPerElement);
    }
  }
  if (size_t remainder = output.size() % kElementsPerByte; remainder != 0) {
    const char byte = input[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      output[aligned_outputs * kElementsPerByte + j] =
          (byte >> (kBitsPerElement * (kElementsPerByte - j - 1))) &
          LsbMask<uint8_t>(kBitsPerElement);
    }
  }
}

inline void UnpackIntN(int bits_per_element, absl::Span<const char> input,
                       absl::Span<char> output) {
  if (bits_per_element == 2) {
    UnpackIntN<2>(input, output);
  } else if (bits_per_element == 4) {
    UnpackIntN<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

// Returns a container with `sorted_ids_to_remove` elements removed.
template <typename T>
static T RemoveElements(absl::Span<int64_t const> sorted_ids_to_remove,
                        const T& container) {
  T result;
  auto id_to_remove = sorted_ids_to_remove.begin();
  for (size_t i = 0; i < container.size(); ++i) {
    if (id_to_remove != sorted_ids_to_remove.end() && *id_to_remove == i) {
      ++id_to_remove;
      continue;
    }
    result.push_back(container[i]);
  }
  return result;
}

class HloInstruction;
class HloModule;

// A predicate over HLO instruction.
using HloPredicate = std::function<bool(const HloInstruction*)>;
using HloModulePredicate = std::function<bool(const HloModule*)>;

inline bool HloPredicateTrue(const HloInstruction*) { return true; }
inline bool HloPredicateFalse(const HloInstruction*) { return false; }

using Vector2 = std::array<int64_t, 2>;
using Vector3 = std::array<int64_t, 3>;

}  // namespace xla

// Note that STRING is evaluated regardless of whether it will be logged.
#define XLA_LOG_LINES(SEV, STRING) \
  ::xla::LogLines##SEV(STRING, __FILE__, __LINE__)

// Like LOG_LINES, but only logs if VLOG is enabled for the given level.
// STRING is evaluated only if it will be logged.
#define XLA_VLOG_LINES(LEVEL, STRING)                   \
  do {                                                  \
    if (VLOG_IS_ON(LEVEL)) XLA_LOG_LINES(INFO, STRING); \
  } while (false)

#endif  // XLA_UTIL_H_
