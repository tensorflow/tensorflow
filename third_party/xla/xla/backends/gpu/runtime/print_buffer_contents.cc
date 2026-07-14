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

#include "xla/backends/gpu/runtime/print_buffer_contents.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/kernel_args.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

namespace {
void PrintBufferContents(stream_executor::Stream*, int input_idx,
                         stream_executor::TensorMap tensor_map) {
  std::string formatted_contents;
  for (std::byte element : tensor_map.storage) {
    absl::StrAppendFormat(&formatted_contents, "%02x ",
                          static_cast<unsigned>(element));
  }
  LOG(INFO) << "TENSOR_MAP(" << input_idx << ") = " << formatted_contents;
}

void PrintBufferContents(stream_executor::Stream* stream, int input_idx,
                         stream_executor::DeviceAddressBase buf) {
  auto host_buffer = std::make_unique<char[]>(buf.size());
  CHECK_OK(stream->Memcpy(host_buffer.get(), buf, buf.size()));
  CHECK_OK(stream->BlockHostUntilDone());

  std::string buffer_contents;
  for (int i = 0; i < buf.size(); ++i) {
    absl::StrAppendFormat(&buffer_contents, "%02x ",
                          static_cast<unsigned>(host_buffer[i]));
  }
  LOG(INFO) << "BUF(" << input_idx << ") = " << buffer_contents;
}

void PrintBufferContents(stream_executor::Stream*, int input_idx,
                         int64_t int_arg) {
  LOG(INFO) << "INT(" << input_idx
            << ") = " << absl::StrFormat("%#08x", int_arg);
}
}  // namespace

void PrintBufferContents(
    stream_executor::Stream* stream,
    absl::Span<const stream_executor::KernelArg> kernel_args) {
  for (int input_idx = 0; input_idx < kernel_args.size(); ++input_idx) {
    const stream_executor::KernelArg& arg = kernel_args[input_idx];
    std::visit(
        [&](auto const& arg) { PrintBufferContents(stream, input_idx, arg); },
        arg);
  }
}

}  // namespace xla::gpu
