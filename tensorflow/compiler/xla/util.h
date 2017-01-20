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
#include <vector>

#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/math/math_util.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// RAII timer that logs with a given label the wall clock time duration in human
// readable form. This differs from base's ElapsedTimer primarily in that it
// spits out the human-readable duration form.
struct ScopedLoggingTimer {
  explicit ScopedLoggingTimer(const string& label, int32 vlog_level = 1);
  ~ScopedLoggingTimer();

  uint64 start_micros;
  string label;
  int32 vlog_level;
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

// Compares two containers for equality. Returns true iff the two containers
// have the same size and all their elements compare equal using the predicate
// p. Like std::equal, but forces size equality.
template <typename Container1T, typename Container2T, class PredicateT>
bool ContainersEqual(const Container1T& c1, const Container2T& c2,
                     PredicateT p) {
  return ((c1.size() == c2.size()) &&
          std::equal(std::begin(c1), std::end(c1), std::begin(c2), p));
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
Status ResourceExhausted(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status NotFound(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);
Status Unavailable(const char* format, ...) TF_PRINTF_ATTRIBUTE(1, 2);

// Splits the lines of the original, replaces leading whitespace with the prefix
// given by "indentation", and returns the string joined by newlines again. As a
// side effect, any additional trailing whitespace is removed.
//
// Note: even different amounts of leading whitespace on different lines will be
// uniformly replaced with "indentation".
string Reindent(tensorflow::StringPiece original,
                tensorflow::StringPiece indentation);

// Applies `permutation` on `input` and returns the permuted array.
// For each i, output[permutation[i]] = input[i].
//
// Precondition:
// 1. `permutation` is a permutation of 0..permutation.size()-1.
// 2. permutation.size() == input.size().
template <template <typename...> class C, typename T>
std::vector<T> Permute(tensorflow::gtl::ArraySlice<int64> permutation,
                       C<T> input_) {
  tensorflow::gtl::ArraySlice<T> input(input_);
  CHECK_EQ(permutation.size(), input.size());
  std::vector<T> output(input.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    output[permutation[i]] = input[i];
  }
  DCHECK(std::is_permutation(input.begin(), input.end(), output.begin()));
  return output;
}

// Inverts a permutation, i.e., output_permutation[input_permutation[i]] = i.
std::vector<int64> InversePermutation(
    tensorflow::gtl::ArraySlice<int64> input_permutation);

// Composes two permutations: output[i] = p1[p2[i]].
std::vector<int64> ComposePermutations(tensorflow::gtl::ArraySlice<int64> p1,
                                       tensorflow::gtl::ArraySlice<int64> p2);

template <typename Container>
int64 PositionInContainer(const Container& container, int64 value) {
  return std::distance(container.begin(),
                       std::find(container.begin(), container.end(), value));
}

// Returns a PaddingConfig object that represents no padding for the given rank.
PaddingConfig MakeNoPaddingConfig(int64 rank);

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
// then multiplying by the divisor. For example: RoundUpToMultiple(13, 8) => 16
template <typename T>
T RoundUpToNearest(T value, T divisor) {
  return CeilOfRatio(value, divisor) * divisor;
}

// Given a number of flops executed in an amount of time, produces a string that
// represents the throughput;
// e.g. HumanReadableNumFlops(1e9, 1e9) => 1.00GFLOP/s.
string HumanReadableNumFlops(double flops, double nanoseconds);

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

}  // namespace xla

#define XLA_LOG_LINES(SEV, STRING) LogLines(SEV, STRING, __FILE__, __LINE__)

#define XLA_VLOG_LINES(LEVEL, STRING)                               \
  do {                                                              \
    if (VLOG_IS_ON(LEVEL)) XLA_LOG_LINES(tensorflow::INFO, STRING); \
  } while (false);

// Utility macro that performs the equivalent of what one would expect
// LOG_LINES(FATAL, X) to do but can be used at the end of a function that
// returns a value without getting a compiler warning that no value is returned.
#define XLA_FATAL_LOG(X)               \
  XLA_LOG_LINES(tensorflow::ERROR, X); \
  LOG(FATAL) << "Aborting in " << __FUNCTION__ << " due to previous errors.";

#endif  // TENSORFLOW_COMPILER_XLA_UTIL_H_
