/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PACKING_H_
#define XLA_PACKING_H_

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"

namespace xla {

// Takes a sequence of unpacked kBitsPerElement-bit values (kBitsPerElement must
// be 1, 2, or 4), such that every byte stores one value in the low-order
// bits, and packs them so every byte stores as many which will fit. `output`
// should have at least ceil((input.size()*kBitsPerElement)/8.0) bytes. The
// high-order bits of each byte in `input` are ignored.
template <size_t kBitsPerElement>
void PackIntNHwy(absl::Span<const char> input, absl::Span<char> output);

template <>
void PackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output);
template <>
void PackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output);
template <>
void PackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output);

// Same as above, but takes the number of bits per element as an argument.
// `bits_per_element` must be 1, 2, or 4, or this function will crash.
void PackIntNHwy(int bits_per_element, absl::Span<const char> input,
                 absl::Span<char> output);

// Takes a sequence of packed values, such that every byte stores multiple
// values, and unpacks them so every byte stores one value in the low-order
// bits. `input` should have
// ceil(output.size()*8.0/kBitsPerElement) bytes. kBitsPerElement must be
// 1, 2, or 4. The high-order bits in each output are zero.
template <size_t kBitsPerElement>
void UnpackIntNHwy(absl::Span<const char> input, absl::Span<char> output);

template <>
void UnpackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output);
template <>
void UnpackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output);
template <>
void UnpackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output);

// Same as above, but takes the number of bits per element as an argument.
// `bits_per_element` must be 1, 2, or 4, or this function will crash.
void UnpackIntNHwy(int bits_per_element, absl::Span<const char> input,
                   absl::Span<char> output);

}  // namespace xla

#endif  // XLA_PACKING_H_
