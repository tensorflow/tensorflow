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

#ifndef XLA_BACKENDS_CPU_RUNTIME_SORT_LIB_H_
#define XLA_BACKENDS_CPU_RUNTIME_SORT_LIB_H_

#include <cstddef>
#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/types/span.h"

namespace xla::cpu::internal {

// Conceptually we have a 3-dimensional shape:
//
//   [outer_dim_size, sort_dim_size, inner_dim_size]
//
// We sort `outer_dim_size * inner_dim_size` vectors of length `sort_dim_size`,
// by iterating over `data` memory and calling `std::sort` (or
// `std::stable_sort`) on each (strided) slice of the buffer.
struct SortDims {
  int64_t outer_dim_size;
  int64_t sort_dim_size;
  int64_t inner_dim_size;
};

// For trivial sort functors (computation with two parameters that are
// compared using `LT` or `GT` direction) we can define sort as a enum. We use
// it for performance optimization to be able to inline the sort function.
enum class SortDirection {
  kAscending,
  kDescending,
};

// Sorts `data` using `less_than` comparator function. Data is sorted in place,
// and sort dimensions are specified in `sort_dims`.
using LessThan = absl::AnyInvocable<bool(const void** data)>;
void SortInplace(const SortDims& sort_dims, absl::Span<std::byte* const> data,
                 absl::Span<const size_t> primitive_sizes, bool is_stable,
                 LessThan* less_than);

// Sorts `data` using the sort `direction` with builtin comparator functions.
// This is more efficient, as the comparator can be inlined.
template <typename T>
void SortInplace(const SortDims& sort_dims, T* data, bool is_stable,
                 SortDirection direction);

// Declare SortInplace for all supported types. Template is instantiated in
// the .cc file.
#define DECLARE_SORT_INPLACE(T) \
  extern template void SortInplace<T>(const SortDims&, T*, bool, SortDirection)

DECLARE_SORT_INPLACE(float);
DECLARE_SORT_INPLACE(double);
DECLARE_SORT_INPLACE(int8_t);
DECLARE_SORT_INPLACE(int16_t);
DECLARE_SORT_INPLACE(int32_t);
DECLARE_SORT_INPLACE(int64_t);
DECLARE_SORT_INPLACE(uint8_t);
DECLARE_SORT_INPLACE(uint16_t);
DECLARE_SORT_INPLACE(uint32_t);
DECLARE_SORT_INPLACE(uint64_t);

#undef DECLARE_SORT_INPLACE

}  // namespace xla::cpu::internal

#endif  // XLA_BACKENDS_CPU_RUNTIME_SORT_LIB_H_
