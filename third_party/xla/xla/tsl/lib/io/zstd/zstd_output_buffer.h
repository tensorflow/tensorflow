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

#ifndef XLA_TSL_LIB_IO_ZSTD_ZSTD_OUTPUT_BUFFER_H_
#define XLA_TSL_LIB_IO_ZSTD_ZSTD_OUTPUT_BUFFER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"
#include "zstd.h"

namespace tsl::io {

// ZstdOutputBuffer is a WritableFile that compresses all data that is written
// into it and writes the compressed data into `file`.
// The compression level can be adjusted with the compression_level parameter.
// Compression level 3 can be considered the default and is s a good starting
// point for most applications.
class ZstdOutputBuffer : public WritableFile {
 public:
  static absl::StatusOr<std::unique_ptr<ZstdOutputBuffer>> Create(
      WritableFile* file, int compression_level);
  ~ZstdOutputBuffer() override;

  absl::Status Append(absl::string_view data) override;
  absl::Status Flush() override;
  absl::Status Close() override;
  absl::Status Sync() override;

 private:
  ZstdOutputBuffer(ZSTD_CCtx* context, std::vector<char> output_buffer,
                   WritableFile* file)
      : context_(context),
        output_buffer_(std::move(output_buffer)),
        file_(file) {}

  absl::Status WriteInternal(ZSTD_inBuffer* input_buffer,
                             ZSTD_EndDirective end_directive);

  ZSTD_CCtx* context_;
  std::vector<char> output_buffer_;
  WritableFile* file_;
  bool is_closed_ = false;
};
}  // namespace tsl::io

#endif  // XLA_TSL_LIB_IO_ZSTD_ZSTD_OUTPUT_BUFFER_H_
