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

#ifndef XLA_TSL_LIB_IO_TSL_TO_PROTOBUF_INPUT_STREAM_ADAPTER_H_
#define XLA_TSL_LIB_IO_TSL_TO_PROTOBUF_INPUT_STREAM_ADAPTER_H_

#include <cstring>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "xla/tsl/lib/io/inputstream_interface.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/tstring.h"

namespace tsl::io {

// Implements a Protobuf CopyingInputStream on top of a TSL
// InputStreamInterface
class TslToProtobufInputStreamAdapter
    : public tsl::protobuf::io::CopyingInputStream {
 public:
  explicit TslToProtobufInputStreamAdapter(InputStreamInterface* input_stream)
      : input_stream_(input_stream) {}

  int Read(void* buffer, int size) override {
    tsl::tstring result;
    auto status = input_stream_->ReadNBytes(size, &result);

    // OutOfRange is not an error, it just means, that less than `size` bytes
    // could be read.
    if (!status.ok() && !absl::IsOutOfRange(status)) {
      return -1;
    }

    CHECK(result.size() <= size);
    // tsl::InputStreamInterface makes us read into a tstring instead of an
    // arbitrary buffer, so we need to have this additional copy here.
    std::memcpy(buffer, result.data(), result.size());

    return result.size();
  }

 private:
  InputStreamInterface* input_stream_;
};
}  // namespace tsl::io

#endif  // XLA_TSL_LIB_IO_TSL_TO_PROTOBUF_INPUT_STREAM_ADAPTER_H_
