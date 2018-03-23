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

// Generally useful utility functions that are common to (not specific to any
// given part of) the XLA code base.

#ifndef TENSORFLOW_COMPILER_XLA_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_UTIL_H_

#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Logs the provided status message with a backtrace.
//
// For use by Status-factories, logs a backtrace at the point where the status
// is created, such that we can use --vmodule=util=1 to see all status
// creation backtraces.
Status WithLogBacktrace(const Status& status);

// Ranks greater than 8 are very rare, so use InlinedVector<int64, 8> to store
// the bounds and indices. And for the rare cases of ranks greater than 8,
// the InlinedVector will just behave like an std::vector<> and allocate the
// memory to store its values.
static constexpr int kInlineRank = 8;
using DimensionVector = tensorflow::gtl::InlinedVector<int64, kInlineRank>;

// RAII timer that logs with a given label the wall clock time duration in human
// readable form. This differs from base's ElapsedTimer primarily in that it
// spits out the human-readable duration form.
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
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, 1, __COUNTER__)
#define XLA_SCOPED_LOGGING_TIMER_LEVEL(label, level) \
  XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, __COUNTER__)

// Helper for implementing macros above.  Do not use directly.
//
// Forces the evaluation of "counter", which we expect is equal to __COUNTER__.
#define XLA_SCOPED_LOGGING_TIMER_HELPER(label, level, counter) \
  XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter)

// Helper for macros above.  Don't use directly.
#define XLA_SCOPED_LOGGING_TIMER_HELPER2(label, level, counter)      \
  ::xla::ScopedLoggingTimer XLA_ScopedLoggingTimerInstance##counter( \
      label, VLOG_IS_ON(level))

// RAII timer for XLA_SCOPED_LOGGING_TIMER and XLA_SCOPED_LOGGING_TIMER_LEVEL
// macros above.  Recommended usage is via the macros so you don't have to give
// the timer a name or worry about calling VLOG_IS_ON yourself.
struct ScopedLoggingTimer {
  // The timer does nothing if enabled is false.  This lets you pass in your
  // file's VLOG_IS_ON value.
  ScopedLoggingTimer(const string& label, bool enabled);
  ~ScopedLoggingTimer();

  bool enabled;
  string label;
  uint64 start_micros;
};

// Given a vector<T>, returns a MutableArraySlice<char> that points at its
// internals.
//
// Warning: if the vector is updated its storage pointer may change, so use this
// with caution (ideally in limited scopes with temporary lifetimes).
template <typename T>
tensorflow::gtl::MutableArraySlice<uint8> MutableByteSlice(std::vector<T>* v) {
  return tensorflow::gtl::MutableArraySlice<uint8>(
      reinterpret_cast<uint8*>(v->data()), v->size() * sizeof(T));
}

// Turns an immutable slice of type T into an immutable slice of bytes with the
// same byte size.
template <typename T>
tensorflow::gtl::ArraySlice<uint8> CastToByteSlice(
    tensorflow::gtl::ArraySlice<T> slice) {
  return tensorflow::gtl::ArraySlice<uint8>(
      reinterpret_cast<const uint8*>(slice.data()), slice.size() * sizeof(T));
}

// Casts a byte slice to a non-byte type T, checking that the original slice
// length is a multiple of sizeof(T).
template <typename T>
tensorflow::gtl::ArraySlice<T> CastByteSlice(
    tensorflow::gtl::ArraySlice<uint8> slice) {
  CHECK_EQ(0, slice.size() % sizeof(T));
  return tensorflow::gtl::ArraySlice<T>(
      reinterpret_cast<const T*>(slice.data()), slice.size() / sizeof(T));
}

// Convenience function to force a vector to convert to an immutable slice.
template <typename T>
tensorflow::gtl::ArraySlice<T> AsSlice(const std::vector<T>& v) {
  return tensorflow::gtl::ArraySlice<T>(v);
}

// Converts a mutable vector pointer into a MutableArraySlice of the same
// type.
template <typename T>
tensorflow::gtl::MutableArraySlice<T> AsMutableSlice(std::vector<T>* v) {
  return tensorflow::gtl::MutableArraySlice<T>(v->data(), v->size());
}

// xla::int64 is not the same type as tensorflow::protobuf_int64 in open-source.
// Wrapper function that gives an int64 array slice view of a repeated int64
// protobuf field.
static inline tensorflow::gtl::ArraySlice<int64> AsInt64Slice(
    const tensorflow::protobuf::RepeatedField<tensorflow::protobuf_int64>& v) {
  tensorflow::gtl::ArraySlice<tensorflow::protobuf_int64> slice(v);
  return tensorflow::gtl::ArraySlice<int64>(
      reinterpret_cast<const int64*>(slice.data()), slice.size());
}

// As above, but for uint64 types.
static inline tensorflow::gtl::ArraySlice<uint64> AsUInt64Slice(
    const tensorflow::protobuf::RepeatedField<tensorflow::protobuf_uint64>& v) {
  tensorflow::gtl::ArraySlice<tensorflow::protobuf_uint64> slice(v);
  return tensorflow::gtl::ArraySlice<uint64>(
      reinterpret_cast<const uint64*>(slice.data()), slice.size());
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using their
// operator==. Like std::equal, but forces size equality.
template <typename Container1T, typename Container2T>
bool ContainersEqual(const Container1T& c1, const Container2T& c2) {
  return ((c1.size() == c2.size()) &&
          std::equal(std::begin(c1), std::end(c1), std::begin(c2)));
}

template <typename Container1T,
          typename ElementType = typename Container1T::value_type>
bool ContainersEqual(const Container1T& c1,
                     std::initializer_list<ElementType> il) {
  tensorflow::gtl::ArraySlice<ElementType> c2{il};
  return ContainersEqual(c1, c2);
}

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using the predicate
// p. Like std::equal, but forces size equality.
template <typename Container1T, typename Container2T, class PredicateT>
bool ContainersEqual(const Container1T& c1, const Container2T& c2,
                     PredicateT p) {
  return ((c1.size() == c2.size()) &&
          std::equal(std::begin(c1), std::end(c1), std::begin(c2), p));
}

// Performs a copy of count values from src to dest, using different strides for
// source and destination. The source starting index is src_base, while the
// destination one is dest_base.
template <typename D, typename S>
void StridedCopy(tensorflow::gtl::MutableArraySlice<D> dest, int64 dest_base,
                 int64 dest_stride, tensorflow::gtl::ArraySlice<S> src,
                 int64 src_base, int64 src_stride, int64 count) {
  for (; count > 0; --count, dest_base += dest_stride, src_base += src_stride) {
    dest[dest_base] = static_cast<D>(src[src_base]);
  }
}

// Adds some context information to the error message in a
// Status.  This is useful as Statuses are
// propagated upwards.
Status AddStatus(Status prior, tensorflow::StringPiece context);
Status AppendStatus(Status prior, tensorflow::StringPiece context);

// Status error shorthands -- printfs the arguments to be
// used as an error message and returns a status in the canonical
// error space.
Status InvalidArgument(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status Unimplemented(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status InternalError(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status FailedPrecondition(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status Cancelled(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status ResourceExhausted(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status NotFound(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status Unavailable(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);

// Passed-varargs variant of the InvalidArgument factory above.
Status InvalidArgumentV(const char* format, va_list args);

template <typename... Args>
Status UnimplementedStrCat(Args&&... concat) {
  return Unimplemented(
      "%s", tensorflow::strings::StrCat(std::forward<Args>(concat)...).c_str());
}

template <typename... Args>
Status InternalErrorStrCat(Args&&... concat) {
  return InternalError(
      "%s", tensorflow::strings::StrCat(std::forward<Args>(concat)...).c_str());
}

template <typename... Args>
Status ResourceExhaustedStrCat(Args&&... concat) {
  return ResourceExhausted(
      "%s", tensorflow::strings::StrCat(std::forward<Args>(concat)...).c_str());
}

// Splits the lines of the original, replaces leading whitespace with the prefix
// given by "indentation", and returns the string joined by newlines again. As a
// side effect, any additional trailing whitespace is removed.
//
// Note: even different amounts of leading whitespace on different lines will be
// uniformly replaced with "indentation".
string Reindent(tensorflow::StringPiece original,
                tensorflow::StringPiece indentation);

// Checks whether permutation is a permutation of the [0, rank) integer range.
bool IsPermutation(tensorflow::gtl::ArraySlice<int64> permutation, int64 rank);

// Applies `permutation` on `input` and returns the permuted array.
// For each i, output[permutation[i]] = input[i].
//
// Precondition:
// 1. `permutation` is a permutation of 0..permutation.size()-1.
// 2. permutation.size() == input.size().
template <template <typename...> class C, typename T>
std::vector<T> Permute(tensorflow::gtl::ArraySlice<int64> permutation,
                       C<T> input) {
  tensorflow::gtl::ArraySlice<T> data(input);
  CHECK(IsPermutation(permutation, data.size()));
  std::vector<T> output(data.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = data[i];
  }
  return output;
}

// Override of the above that works around compile failures with gcc 7.1.1.
// For details see https://github.com/tensorflow/tensorflow/issues/10843
// Hide this workaround from MSVC as it causes ambiguous error.
#ifndef _MSC_VER
template <typename T>
std::vector<T> Permute(tensorflow::gtl::ArraySlice<int64> permutation,
                       const std::vector<T>& input) {
  return Permute<std::vector, T>(permutation, input);
}
#endif

// Inverts a permutation, i.e., output_permutation[input_permutation[i]] = i.
std::vector<int64> InversePermutation(
    tensorflow::gtl::ArraySlice<int64> input_permutation);

// Composes two permutations: output[i] = p1[p2[i]].
std::vector<int64> ComposePermutations(tensorflow::gtl::ArraySlice<int64> p1,
                                       tensorflow::gtl::ArraySlice<int64> p2);

// Returns true iff permutation == {0, 1, 2, ...}.
bool IsIdentityPermutation(tensorflow::gtl::ArraySlice<int64> permutation);

template <typename Container>
int64 PositionInContainer(const Container& container, int64 value) {
  return std::distance(container.begin(),
                       std::find(container.begin(), container.end(), value));
}

// Formats the container as a comma-separated string. StrAppend must support
// appending the elements of the container. Prefix is prepended and suffix is
// appended to the returned string.
template <typename Container>
string CommaSeparatedString(const Container& c, const char* prefix = "",
                            const char* suffix = "") {
  // Not using Join() since the implementation here is simple anyway and this
  // avoids copying the string to append prefix.
  string comma_separated = prefix;
  const char* separator = "";
  for (const auto& entry : c) {
    tensorflow::strings::StrAppend(&comma_separated, separator, entry);
    separator = ", ";
  }
  comma_separated += suffix;
  return comma_separated;
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
string CommaSeparatedString(const std::initializer_list<T>& c,
                            const char* prefix = "", const char* suffix = "") {
  return CommaSeparatedString<std::initializer_list<T>>(c, prefix, suffix);
}

// Formats the container in the mathematical notation for a vector, e.g. (1, 3,
// 7). StrAppend must support appending the elements of c.
template <typename Container>
string VectorString(const Container& c) {
  return CommaSeparatedString(c, "(", ")");
}

// Overload needed to allow the container to be an initializer list. The default
// type for T makes an empty initializer list work as well.
template <typename T = int>
string VectorString(const std::initializer_list<T>& c) {
  return VectorString<std::initializer_list<T>>(c);
}

// Returns a PaddingConfig object that represents no padding for the given rank.
PaddingConfig MakeNoPaddingConfig(int64 rank);

// Returns a PaddingConfig object where 'padding' contains
// (low edge padding, high edge padding) pairs for each dimension.
PaddingConfig MakeEdgePaddingConfig(
    tensorflow::gtl::ArraySlice<std::pair<int64, int64>> padding);

// Returns true if the padding configuration has at least one dimension with
// non-zero interior padding.
bool HasInteriorPadding(const PaddingConfig& config);

// Imports the templated FloorOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
T FloorOfRatio(T dividend, T divisor) {
  return tensorflow::MathUtil::FloorOfRatio<T>(dividend, divisor);
}

// Imports the templated CeilOfRatio math function from the TensorFlow
// namespace, as it is very commonly used.
template <typename T>
T CeilOfRatio(T dividend, T divisor) {
  return tensorflow::MathUtil::CeilOfRatio<T>(dividend, divisor);
}

// Rounds the value up to a multiple of the divisor by first calling CeilOfRatio
// then multiplying by the divisor. For example: RoundUpToNearest(13, 8) => 16
template <typename T>
T RoundUpToNearest(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

// Rounds the value down to a multiple of the divisor by first calling
// FloorOfRatio then multiplying by the divisor. For example:
// RoundDownToNearest(13, 8) => 8
template <typename T>
T RoundDownToNearest(T value, T divisor) {
  return FloorOfRatio(value, divisor) * divisor;
}

// Given a number of flops executed in an amount of time, produces a string that
// represents the throughput;
// e.g. HumanReadableNumFlops(1e9, 1e9) => 1.00GFLOP/s.
string HumanReadableNumFlops(double flops, double nanoseconds);

// Given a number of transcendental ops executed in an amount of time, produces
// a string that represents the throughput;
// e.g. HumanReadableNumTranscendentalOps(1e9, 1e9) => 1.00GTROP/s.
string HumanReadableNumTranscendentalOps(double trops, double nanoseconds);

// Split the text into multiple lines and log each line with the given
// severity, filename, and line number.
void LogLines(int sev, tensorflow::StringPiece text, const char* fname,
              int lineno);

template <typename T>
inline bool IsPowerOfTwo(T x) {
  static_assert(!std::numeric_limits<T>::is_signed, "unsigned types only");
  return x != 0 && (x & (x - 1)) == 0;
}

// Returns a mask with "bits" number of least significant bits set.
inline uint32 LsbMaskU32(int bits) {
  CHECK_GE(bits, 0);
  return (1U << bits) - 1;
}

// Utility for performing a static_cast<> on a std::unique_ptr<>.
template <typename Derived, typename Base>
std::unique_ptr<Derived> unique_ptr_static_cast(std::unique_ptr<Base> ptr) {
  return std::unique_ptr<Derived>(static_cast<Derived*>(ptr.release()));
}

int64 Product(tensorflow::gtl::ArraySlice<int64> xs);

// Returns the start indices of consecutive non-overlapping subsequences of `a`
// and `b` with the same product, i.e. `(i, j)` so
// • a = {a[0 = i_0], ..., a[i_1 - 1], a[i_1], ... , a[i_2 - 1], ...}
// • b = {b[0 = j_0], ..., b[j_1 - 1], b[j_1], ... , b[j_2 - 1], ...}
// • ∀ k . 0 <= k < CommonFactors(a, b).size - 1 =>
//         a[i_k] × a[i_k + 1] × ... × a[i_(k+1) - 1] =
//         b[j_k] × b[j_k + 1] × ... × b[j_(k+1) - 1]
// where `CommonFactors(a, b)[CommonFactors(a, b).size - 1] = (a.size, b.size)`
//
// If the given shapes have non-zero size, returns the bounds of the shortest
// possible such subsequences; else, returns `{(0, 0), (a.size, b.size)}`.
std::vector<std::pair<int64, int64>> CommonFactors(
    tensorflow::gtl::ArraySlice<int64> a, tensorflow::gtl::ArraySlice<int64> b);

// Removes illegal characters from filenames.
string SanitizeFileName(string file_name);

template <typename Container, typename Predicate>
bool c_all_of(const Container& container, Predicate&& predicate) {
  return std::all_of(std::begin(container), std::end(container),
                     std::forward<Predicate>(predicate));
}

template <typename Container, typename Predicate>
bool c_any_of(const Container& container, Predicate&& predicate) {
  return std::any_of(std::begin(container), std::end(container),
                     std::forward<Predicate>(predicate));
}

template <typename InputContainer, typename OutputIterator,
          typename UnaryOperation>
OutputIterator c_transform(const InputContainer& input_container,
                           OutputIterator output_iterator,
                           UnaryOperation&& unary_op) {
  return std::transform(std::begin(input_container), std::end(input_container),
                        output_iterator,
                        std::forward<UnaryOperation>(unary_op));
}

template <class InputContainer, class OutputIterator, class UnaryPredicate>
OutputIterator c_copy_if(const InputContainer& input_container,
                         OutputIterator output_iterator,
                         UnaryPredicate&& predicate) {
  return std::copy_if(std::begin(input_container), std::end(input_container),
                      output_iterator, std::forward<UnaryPredicate>(predicate));
}

template <class InputContainer, class OutputIterator>
OutputIterator c_copy(const InputContainer& input_container,
                      OutputIterator output_iterator) {
  return std::copy(std::begin(input_container), std::end(input_container),
                   output_iterator);
}

template <class InputContainer>
void c_sort(InputContainer& input_container) {
  std::sort(std::begin(input_container), std::end(input_container));
}

template <class InputContainer, class Comparator>
void c_sort(InputContainer& input_container, Comparator&& comparator) {
  std::sort(std::begin(input_container), std::end(input_container),
            std::forward<Comparator>(comparator));
}

template <typename Sequence, typename T>
bool c_binary_search(const Sequence& sequence, T&& value) {
  return std::binary_search(std::begin(sequence), std::end(sequence),
                            std::forward<T>(value));
}

template <typename C>
bool c_is_sorted(const C& c) {
  return std::is_sorted(std::begin(c), std::end(c));
}

template <typename C>
auto c_adjacent_find(const C& c) -> decltype(std::begin(c)) {
  return std::adjacent_find(std::begin(c), std::end(c));
}

template <typename C, typename Pred>
auto c_find_if(const C& c, Pred&& pred) -> decltype(std::begin(c)) {
  return std::find_if(std::begin(c), std::end(c), std::forward<Pred>(pred));
}

template <typename C, typename Value>
auto c_find(const C& c, Value&& value) -> decltype(std::begin(c)) {
  return std::find(std::begin(c), std::end(c), std::forward<Value>(value));
}

template <typename Sequence>
void c_reverse(Sequence& sequence) {
  std::reverse(std::begin(sequence), std::end(sequence));
}

template <typename Sequence, typename T, typename BinaryOp>
typename std::decay<T>::type c_accumulate(const Sequence& sequence, T&& init,
                                          BinaryOp&& binary_op) {
  return std::accumulate(std::begin(sequence), std::end(sequence),
                         std::forward<T>(init),
                         std::forward<BinaryOp>(binary_op));
}

template <typename C, typename Value>
int64 FindIndex(const C& c, Value&& value) {
  auto it = c_find(c, std::forward<Value>(value));
  return std::distance(c.begin(), it);
}

// Returns true if `x` fits in 32-bits.
template <typename T>
bool IsInt32(T x) {
  // Following conversion rules: "the value is unchanged if it can be
  // represented in the destination type (and bit-field width); otherwise, the
  // value is implementation-defined."
  return static_cast<int32>(x) == x;
}
}  // namespace xla

#define XLA_LOG_LINES(SEV, STRING) \
  ::xla::LogLines(SEV, STRING, __FILE__, __LINE__)

#define XLA_VLOG_LINES(LEVEL, STRING)                                 \
  do {                                                                \
    if (VLOG_IS_ON(LEVEL)) XLA_LOG_LINES(::tensorflow::INFO, STRING); \
  } while (false);

// Utility macro that performs the equivalent of what one would expect
// LOG_LINES(FATAL, X) to do but can be used at the end of a function that
// returns a value without getting a compiler warning that no value is returned.
#define XLA_FATAL_LOG(X)                 \
  XLA_LOG_LINES(::tensorflow::ERROR, X); \
  LOG(FATAL) << "Aborting in " << __FUNCTION__ << " due to previous errors.";

#endif  // TENSORFLOW_COMPILER_XLA_UTIL_H_
