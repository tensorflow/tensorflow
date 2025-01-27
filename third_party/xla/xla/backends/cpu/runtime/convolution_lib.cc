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

#include "xla/backends/cpu/runtime/convolution_lib.h"

#include <cstdint>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"

namespace xla::cpu {

absl::InlinedVector<BufferUse, 4> ConvolutionBufferUses(
    const ConvolutionSlices& slices) {
  return {BufferUse::Read(slices.input_buffer),
          BufferUse::Read(slices.kernel_buffer),
          BufferUse::Write(slices.output_buffer)};
}

ConvolutionCanonicalDims::Dims::Dims(absl::Span<const int64_t> dims)
    : rank(dims.size()), x(dims[0]), y(dims[1]), z(rank == 3 ? dims[2] : 0) {}

}  // namespace xla::cpu
