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

#include "xla/tsl/lib/io/zstd/zstd_input_stream.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/lib/io/inputstream_interface.h"
#include "xla/tsl/platform/errors.h"
#include "tsl/platform/tstring.h"

namespace tsl::io {

namespace {
absl::Status ToStatus(size_t error_code) {
  if (!ZSTD_isError(error_code)) {
    return absl::OkStatus();
  }
  return absl::InternalError(
      absl::StrCat("ZSTD error: ", ZSTD_getErrorName(error_code)));
}
}  // namespace

absl::Status ZstdInputStream::ReadNBytes(int64_t bytes_to_read,
                                         tstring* output_buffer) {
  output_buffer->resize_uninitialized(bytes_to_read);
  ZSTD_outBuffer output_buffer_descriptor = {output_buffer->data(),
                                             output_buffer->size(), 0};

  while (output_buffer_descriptor.pos < output_buffer_descriptor.size) {
    LOG(INFO) << "output_buffer_descriptor.pos: "
              << output_buffer_descriptor.pos;
    LOG(INFO) << "output_buffer_descriptor.size: "
              << output_buffer_descriptor.size;
    LOG(INFO) << "input_buffer_descriptor_.pos: "
              << input_buffer_descriptor_.pos;
    LOG(INFO) << "input_buffer_descriptor_.size: "
              << input_buffer_descriptor_.size;
    // If .pos == .size, we need to read more data from the input stream.
    if (input_buffer_descriptor_.pos == input_buffer_descriptor_.size) {
      absl::Status read_status =
          input_stream_->ReadNBytes(ZSTD_DStreamInSize(), &input_buffer_);
      if (!read_status.ok() && !absl::IsOutOfRange(read_status)) {
        return read_status;
      }
      input_buffer_descriptor_ = {input_buffer_.data(), input_buffer_.size(),
                                  0};
    }
    LOG(INFO) << "output_buffer_descriptor.pos: "
              << output_buffer_descriptor.pos;
    LOG(INFO) << "output_buffer_descriptor.size: "
              << output_buffer_descriptor.size;
    LOG(INFO) << "input_buffer_descriptor_.pos: "
              << input_buffer_descriptor_.pos;
    LOG(INFO) << "input_buffer_descriptor_.size: "
              << input_buffer_descriptor_.size;

    bool had_empty_input_buffer =
        input_buffer_descriptor_.size == input_buffer_descriptor_.pos;

    TF_RETURN_IF_ERROR(ToStatus(ZSTD_decompressStream(
        context_, &output_buffer_descriptor, &input_buffer_descriptor_)));

    bool have_non_full_output_buffer =
        output_buffer_descriptor.pos < output_buffer_descriptor.size;

    if (had_empty_input_buffer && have_non_full_output_buffer) {
      break;
    }
  }

  bytes_read_ += output_buffer_descriptor.pos;
  output_buffer->resize(output_buffer_descriptor.pos);
  if (output_buffer_descriptor.pos < bytes_to_read) {
    return absl::OutOfRangeError(absl::StrFormat(
        "%d bytes were requested, but could only produce %d bytes.",
        bytes_to_read, output_buffer_descriptor.pos));
  }
  return absl::OkStatus();
}

int64_t ZstdInputStream::Tell() const { return bytes_read_; }

absl::Status ZstdInputStream::Reset() {
  return absl::UnimplementedError("Resetting the stream is not supported.");
}

absl::StatusOr<std::unique_ptr<ZstdInputStream>> ZstdInputStream::Create(
    InputStreamInterface* input_stream) {
  auto context = ZSTD_createDCtx();
  if (context == nullptr) {
    return absl::InternalError("ZSTD_createDCtx() failed!");
  }

  tstring input_buffer;
  input_buffer.resize_uninitialized(ZSTD_DStreamInSize());

  return std::unique_ptr<ZstdInputStream>(
      new ZstdInputStream(input_stream, context, std::move(input_buffer)));
}

ZstdInputStream::~ZstdInputStream() { ZSTD_freeDCtx(context_); }

}  // namespace tsl::io
