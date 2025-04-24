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

#include "xla/tsl/lib/io/zstd/zstd_output_buffer.h"

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "zstd.h"

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

ZstdOutputBuffer::~ZstdOutputBuffer() {
  auto result = ZstdOutputBuffer::Close();
  if (!result.ok()) {
    LOG(ERROR) << "Failed to close ZstdOutputBuffer: " << result;
  }
  size_t error_code = ZSTD_freeCCtx(context_);
  if (ZSTD_isError(error_code)) {
    LOG(ERROR) << "ZSTD_freeCCtx() failed: " << ZSTD_getErrorName(error_code);
  }
}

absl::StatusOr<std::unique_ptr<ZstdOutputBuffer>> ZstdOutputBuffer::Create(
    WritableFile* file, int compression_level) {
  ZSTD_CCtx* cctx = ZSTD_createCCtx();
  if (cctx == nullptr) {
    return absl::InternalError("ZSTD_createCCtx() failed!");
  }

  TF_RETURN_IF_ERROR(ToStatus(ZSTD_CCtx_setParameter(
      cctx, ZSTD_c_compressionLevel, compression_level)));
  TF_RETURN_IF_ERROR(
      ToStatus(ZSTD_CCtx_setParameter(cctx, ZSTD_c_checksumFlag, 1)));

  size_t output_buffer_size = ZSTD_CStreamOutSize();
  std::vector<char> output_buffer(output_buffer_size);

  return std::unique_ptr<ZstdOutputBuffer>(
      new ZstdOutputBuffer(cctx, std::move(output_buffer), file));
}

absl::Status ZstdOutputBuffer::WriteInternal(
    ZSTD_inBuffer* input_buffer_descriptor, ZSTD_EndDirective end_directive) {
  // We initialize remaining_bytes to 1 to make sure we call compressStream2 at
  // least once.
  size_t remaining_bytes = 1;

  // There are two cases when we have to call compressStream2 again:
  // 1. We didn't read all the data from the input buffer (.pos < .size).
  // 2. We didn't write all the data to the output buffer (remaining_bytes > 0).
  while (input_buffer_descriptor->pos < input_buffer_descriptor->size ||
         remaining_bytes > 0) {
    ZSTD_outBuffer output_buffer_descriptor = {output_buffer_.data(),
                                               output_buffer_.size(), 0};
    remaining_bytes =
        ZSTD_compressStream2(context_, &output_buffer_descriptor,
                             input_buffer_descriptor, end_directive);
    TF_RETURN_IF_ERROR(ToStatus(remaining_bytes));
    if (output_buffer_descriptor.pos > 0) {
      TF_RETURN_IF_ERROR(file_->Append(absl::string_view(
          absl::bit_cast<const char*>(output_buffer_descriptor.dst),
          output_buffer_descriptor.pos)));
    }
  }
  return absl::OkStatus();
}

absl::Status ZstdOutputBuffer::Append(absl::string_view data) {
  ZSTD_inBuffer input_buffer_descriptor = {absl::bit_cast<void*>(data.data()),
                                           data.size(), 0};
  return WriteInternal(&input_buffer_descriptor, ZSTD_e_continue);
}
absl::Status ZstdOutputBuffer::Flush() {
  ZSTD_inBuffer input_buffer_descriptor = {nullptr, 0, 0};
  TF_RETURN_IF_ERROR(WriteInternal(&input_buffer_descriptor, ZSTD_e_flush));
  return file_->Flush();
}

absl::Status ZstdOutputBuffer::Close() {
  if (is_closed_) {
    return absl::OkStatus();
  }
  ZSTD_inBuffer input_buffer_descriptor = {nullptr, 0, 0};
  TF_RETURN_IF_ERROR(WriteInternal(&input_buffer_descriptor, ZSTD_e_end));
  TF_RETURN_IF_ERROR(file_->Close());
  is_closed_ = true;
  return absl::OkStatus();
}

absl::Status ZstdOutputBuffer::Sync() {
  // We need to flush because ZSTD has some internal buffers that need to be
  // written to the output file first.
  TF_RETURN_IF_ERROR(Flush());
  return file_->Sync();
}

}  // namespace tsl::io
