/*
 * Copyright 2023 The OpenXLA Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef XLA_PYTHON_IFRT_PROXY_COMMON_ARRAY_UTIL_H_
#define XLA_PYTHON_IFRT_PROXY_COMMON_ARRAY_UTIL_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {
namespace proxy {

// Returns the byte-strides corresponding to the compact major-to-minor layout.
absl::StatusOr<std::vector<int64_t>> DefaultByteStrides(DType dtype,
                                                        const Shape& shape);

// Denotes a chunk of contiguous memory that contains all elements of the
// in-host (RAM) representation of an Array.
class ArrayMemRegion {
 public:
  // Nullopt implies compact major-to-minor layout, as returned by
  // `DefaultByteStrides()`.
  using ByteStrides = std::optional<absl::Span<const int64_t>>;

  // Constructs an ArrayMemRegion given `mem_region`, where `mem_region` is
  // minimal, i.e., the lower-most and upper-most addresses of `mem_region` are
  // necessary to retrieve elements from the array.
  static absl::StatusOr<ArrayMemRegion> FromMinimalMemRegion(
      absl::string_view mem_region, DType dtype, const Shape& shape,
      ByteStrides byte_strides);

  // Constructs an ArrayMemRegion given a pointer to the zeroth-element of the
  // (in-host representation of the) Array.
  static absl::StatusOr<ArrayMemRegion> FromZerothElementPointer(
      const void* zeroth_element, DType dtype, const Shape& shape,
      ByteStrides byte_strides);

  // Returns a region of memory whose lower-most and upper-most addresses are
  // necessary to retrieve elements of the (in-host representation of) the
  // array.
  absl::string_view mem_region() const;

  // Returns a pointer to the zeroth-element of the (in-host representation of
  // the) Array.
  void* zeroth_element() const;

  // Returns the number of bytes necessary for the (in-host representation of
  // the) Array.
  size_t nbytes() const;

 private:
  ArrayMemRegion(void* mem_region_start, size_t nbytes)
      : mem_region_start_(mem_region_start), nbytes_(nbytes) {}

  void* const mem_region_start_;
  const size_t nbytes_;
};

// Utilities for serializing and deserializing a host buffer of dtype
// `DType::kString` (represented as arrays of absl::Cords).
absl::StatusOr<std::unique_ptr<std::string>> SerializeStringHostBuffer(
    absl::Span<const absl::Cord> cords);

absl::StatusOr<std::vector<absl::Cord>> DeserializeStringHostBufferFromString(
    const std::string& serialized_string_buffer);

// Callers must ensure that the `preallocated_buffer` consists of `N`
// `absl::Cord` objects, where N is the number of string elements in the
// `serialized_string_buffer`.
absl::Status DeserializeFromCordIntoPreallocatedStringHostBuffer(
    const absl::Cord& serialized_string_buffer,
    absl::Cord* preallocated_buffer);

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_PROXY_COMMON_ARRAY_UTIL_H_
