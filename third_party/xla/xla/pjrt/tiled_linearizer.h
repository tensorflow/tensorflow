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

#ifndef XLA_PJRT_TILED_LINEARIZER_H_
#define XLA_PJRT_TILED_LINEARIZER_H_

#include <cstdint>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/transpose.h"
#include "xla/shape.h"

namespace xla {

class TiledLinearizer {
 public:
  struct ParallelismOptions {
    int max_parallelism = 1;
    absl::AnyInvocable<void(const TransposePlan& plan, const void* a, void* b)
                           const>
        run_plan;
  };

  TiledLinearizer() = delete;
  TiledLinearizer(const TiledLinearizer&) = delete;
  TiledLinearizer& operator=(const TiledLinearizer&) = delete;

  // Linearizes `data` (with shape `host_shape` and `byte_strides`) into the
  // pre-allocated `dest` buffer according to the resolved `device_layout`.
  // `dest` must be 4-byte aligned and sized to match the linearized layout.
  static absl::Status LinearizeUsingTransposeInto(
      absl::Span<uint8_t> dest, const void* data, const Shape& host_shape,
      absl::Span<int64_t const> byte_strides, const Layout& device_layout,
      bool set_padding_to_nan, absl::Span<const uint32_t> dynamic_sizes,
      const ParallelismOptions& parallelism_options);

  // De-linearizes the provided sequence of bytes into a literal with the
  // given shape, where `device_layout` already has its tiles and
  // `element_size_in_bits` populated.
  static absl::Status Delinearize(
      const Shape& literal_shape, absl::Span<const uint8_t> data,
      const Layout& device_layout, MutableLiteralBase* literal,
      const ParallelismOptions& parallelism_options);

  // Converts two literals to a X64 literal, where the low and high 32-bit
  // components are pulled from respective input literals.
  static absl::Status X64FromTuple(const LiteralSlice& low,
                                   const LiteralSlice& high,
                                   MutableBorrowingLiteral literal);

  // Converts two literals to a X128 literal, where the low and high 64-bit
  // components are pulled from respective input literals.
  static absl::Status X128FromTuple(const LiteralSlice& low,
                                    const LiteralSlice& high,
                                    MutableBorrowingLiteral literal);
};

}  // namespace xla

#endif  // XLA_PJRT_TILED_LINEARIZER_H_
