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

#ifndef XLA_TSL_LIB_IO_ZSTD_ZSTD_INPUT_STREAM_H_
#define XLA_TSL_LIB_IO_ZSTD_ZSTD_INPUT_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/lib/io/inputstream_interface.h"
#include "tsl/platform/tstring.h"
#include "zstd.h"

namespace tsl::io {

class ZstdInputStream : public InputStreamInterface {
 public:
  static absl::StatusOr<std::unique_ptr<ZstdInputStream>> Create(
      InputStreamInterface* input_stream);
  ~ZstdInputStream() override;

  absl::Status ReadNBytes(int64_t bytes_to_read,
                          tstring* output_buffer) override;

  int64_t Tell() const override;

  absl::Status Reset() override;

 private:
  explicit ZstdInputStream(InputStreamInterface* input_stream,
                           ZSTD_DCtx* context, tstring input_buffer)
      : input_stream_(input_stream),
        context_(context),
        input_buffer_(std::move(input_buffer)),
        input_buffer_descriptor_({input_buffer_.data(), input_buffer_.size(),
                                  input_buffer_.size()}) {}
  ZstdInputStream(ZstdInputStream&& other) = default;
  ZstdInputStream& operator=(ZstdInputStream&& other) = default;

  InputStreamInterface* input_stream_;
  ZSTD_DCtx* context_;

  tstring input_buffer_;

  ZSTD_inBuffer input_buffer_descriptor_;

  size_t bytes_read_ = 0;
};
}  // namespace tsl::io

#endif  // XLA_TSL_LIB_IO_ZSTD_ZSTD_INPUT_STREAM_H_
