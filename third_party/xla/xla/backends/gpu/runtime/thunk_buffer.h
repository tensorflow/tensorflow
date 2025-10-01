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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_H_

#include <string>
#include <tuple>

#include "absl/strings/str_format.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

// A buffer used by a `Thunk`.
struct ThunkBuffer {
  // The buffer used by the thunk. May be a host or device buffer, check the
  // `BufferAllocation::Color` of slice's allocation to determine.
  BufferAllocation::Slice slice;
  // True if the slice has defined contents when the thunk starts execution.
  bool is_content_defined_on_input = true;
  // True if the slice has defined contents when the thunk finishes execution.
  // This may mean the buffer contains the thunk's output, or that the thunk is
  // expected to preserve the original value of the input buffer.
  bool is_content_defined_on_output = true;

  bool operator==(const ThunkBuffer& other) const {
    return std::tie(slice, is_content_defined_on_input,
                    is_content_defined_on_output) ==
           std::tie(other.slice, other.is_content_defined_on_input,
                    other.is_content_defined_on_output);
  }
  bool operator!=(const ThunkBuffer& other) const { return !(*this == other); }

  std::string ToString() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ThunkBuffer& buffer) {
    absl::Format(&sink, "%v", buffer.ToString());
  }
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_BUFFER_H_
