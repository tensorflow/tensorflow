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

#include "xla/tsl/lib/io/zip_writer.h"

#include <zlib.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/io/zlib_compression_options.h"
#include "xla/tsl/lib/io/zlib_outputbuffer.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/statusor.h"

namespace tsl {
namespace io {
namespace {

class WritableStringFile : public tsl::WritableFile {
 public:
  explicit WritableStringFile(std::string* data) : data_(data) {}
  ~WritableStringFile() override = default;

  absl::Status Append(absl::string_view data) override {
    data_->append(data.data(), data.size());
    return absl::OkStatus();
  }

  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Sync() override { return absl::OkStatus(); }

  absl::Status Tell(int64_t* position) override {
    *position = data_->size();
    return absl::OkStatus();
  }

 private:
  std::string* data_;
};

absl::StatusOr<std::string> RawDeflate(absl::string_view input) {
  std::string compressed;
  WritableStringFile f(&compressed);

  auto opts = tsl::io::ZlibCompressionOptions::RAW();
  tsl::io::ZlibOutputBuffer deflate_file(&f, opts.input_buffer_size,
                                         opts.output_buffer_size, opts);
  RETURN_IF_ERROR(deflate_file.Init());
  RETURN_IF_ERROR(deflate_file.Append(input));
  RETURN_IF_ERROR(deflate_file.Close());

  return compressed;
}

}  // namespace

absl::StatusOr<ZipWriter> ZipWriter::Create(
    std::unique_ptr<WritableFile> file) {
  int64_t initial_offset = 0;
  absl::Status status = file->Tell(&initial_offset);
  if (!status.ok() && !absl::IsUnimplemented(status)) {
    return status;
  }
  return ZipWriter(std::move(file), initial_offset);
}

ZipWriter::ZipWriter(std::unique_ptr<WritableFile> file, int64_t initial_offset)
    : file_(std::move(file)),
      current_offset_(initial_offset),
      finished_(false) {}

ZipWriter::ZipWriter(ZipWriter&& other) noexcept
    : file_(std::move(other.file_)),
      current_offset_(other.current_offset_),
      files_(std::move(other.files_)),
      finished_(other.finished_) {
  other.finished_ = true;
}

ZipWriter& ZipWriter::operator=(ZipWriter&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  file_ = std::move(other.file_);
  current_offset_ = other.current_offset_;
  files_ = std::move(other.files_);
  finished_ = other.finished_;
  other.finished_ = true;
  return *this;
}

ZipWriter::~ZipWriter() {
  if (file_ == nullptr || finished_) {
    return;
  }
  absl::Status status = FinishInternal();
  if (!status.ok()) {
    LOG(ERROR) << "ZipWriter destructor failed to finalize archive: " << status;
  }
}

absl::Status ZipWriter::AppendData(absl::string_view data) {
  if (current_offset_ + data.size() > UINT32_MAX) {
    return absl::ResourceExhaustedError("ZIP archive size exceeds 4GB limit.");
  }
  RETURN_IF_ERROR(file_->Append(data));
  current_offset_ += data.size();
  return absl::OkStatus();
}

absl::Status ZipWriter::Append16(uint16_t val) {
  char buf[2];
  buf[0] = static_cast<char>(val & 0xff);
  buf[1] = static_cast<char>((val >> 8) & 0xff);
  return AppendData(absl::string_view(buf, 2));
}

absl::Status ZipWriter::Append32(uint32_t val) {
  char buf[4];
  buf[0] = static_cast<char>(val & 0xff);
  buf[1] = static_cast<char>((val >> 8) & 0xff);
  buf[2] = static_cast<char>((val >> 16) & 0xff);
  buf[3] = static_cast<char>((val >> 24) & 0xff);
  return AppendData(absl::string_view(buf, 4));
}

absl::Status ZipWriter::AddFile(std::string name, absl::string_view content) {
  if (files_.size() >= 0xFFFF) {
    return absl::ResourceExhaustedError(
        "ZIP archive exceeds maximum of 65535 files.");
  }
  if (name.size() > 0xFFFF) {
    return absl::InvalidArgumentError(
        absl::StrCat("Filename too long: ", name.size(), " bytes (max 65535)"));
  }
  if (content.size() > 0xFFFFFFFF) {
    return absl::InvalidArgumentError(absl::StrCat(
        "File content too large: ", content.size(), " bytes (max 4GB)"));
  }

  uint32_t offset = static_cast<uint32_t>(current_offset_);
  uint32_t crc =
      crc32(0, reinterpret_cast<const Bytef*>(content.data()), content.size());

  ASSIGN_OR_RETURN(std::string compressed, RawDeflate(content));
  if (compressed.size() > 0xFFFFFFFF) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Compressed data too large: ", compressed.size(), " bytes (max 4GB)"));
  }

  RETURN_IF_ERROR(Append32(0x04034b50));  // Local file header signature
  RETURN_IF_ERROR(Append16(20));          // Version needed to extract (2.0)
  RETURN_IF_ERROR(Append16(0));           // General purpose bit flag
  RETURN_IF_ERROR(Append16(8));           // Compression method (8 = DEFLATE)
  RETURN_IF_ERROR(Append16(0));           // Last mod file time
  RETURN_IF_ERROR(Append16(0));           // Last mod file date
  RETURN_IF_ERROR(Append32(crc));
  RETURN_IF_ERROR(Append32(compressed.size()));
  RETURN_IF_ERROR(Append32(content.size()));
  RETURN_IF_ERROR(Append16(name.size()));
  RETURN_IF_ERROR(Append16(0));  // Extra field length
  RETURN_IF_ERROR(AppendData(name));
  RETURN_IF_ERROR(AppendData(compressed));

  files_.push_back({std::move(name), offset, crc,
                    static_cast<uint32_t>(compressed.size()),
                    static_cast<uint32_t>(content.size())});
  return absl::OkStatus();
}

absl::Status ZipWriter::Finish() && { return FinishInternal(); }

absl::Status ZipWriter::FinishInternal() {
  finished_ = true;

  uint32_t cd_offset = static_cast<uint32_t>(current_offset_);
  for (const auto& file : files_) {
    RETURN_IF_ERROR(
        Append32(0x02014b50));      // Central directory header signature
    RETURN_IF_ERROR(Append16(20));  // Version made by
    RETURN_IF_ERROR(Append16(20));  // Version needed to extract
    RETURN_IF_ERROR(Append16(0));   // General purpose bit flag
    RETURN_IF_ERROR(Append16(8));   // Compression method (DEFLATE)
    RETURN_IF_ERROR(Append16(0));   // Last mod file time
    RETURN_IF_ERROR(Append16(0));   // Last mod file date
    RETURN_IF_ERROR(Append32(file.crc));
    RETURN_IF_ERROR(Append32(file.compressed_size));
    RETURN_IF_ERROR(Append32(file.uncompressed_size));
    RETURN_IF_ERROR(Append16(file.name.size()));
    RETURN_IF_ERROR(Append16(0));  // Extra field length
    RETURN_IF_ERROR(Append16(0));  // File comment length
    RETURN_IF_ERROR(Append16(0));  // Disk number start
    RETURN_IF_ERROR(Append16(0));  // Internal file attributes
    RETURN_IF_ERROR(Append32(0));  // External file attributes
    RETURN_IF_ERROR(Append32(file.offset));
    RETURN_IF_ERROR(AppendData(file.name));
  }
  if (current_offset_ - cd_offset > UINT32_MAX) {
    return absl::ResourceExhaustedError("Central directory size exceeds 4GB.");
  }
  uint32_t cd_size = static_cast<uint32_t>(current_offset_ - cd_offset);

  RETURN_IF_ERROR(Append32(0x06054b50));  // End of central directory signature
  RETURN_IF_ERROR(Append16(0));           // Number of this disk
  RETURN_IF_ERROR(Append16(0));           // Disk where central directory starts
  RETURN_IF_ERROR(Append16(files_.size()));  // Number of records on this disk
  RETURN_IF_ERROR(Append16(files_.size()));  // Total number of records
  RETURN_IF_ERROR(Append32(cd_size));
  RETURN_IF_ERROR(Append32(cd_offset));
  RETURN_IF_ERROR(Append16(0));  // Comment length

  return file_->Close();
}

}  // namespace io
}  // namespace tsl
