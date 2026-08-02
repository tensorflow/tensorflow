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

#ifndef XLA_TSL_LIB_IO_ZIP_WRITER_H_
#define XLA_TSL_LIB_IO_ZIP_WRITER_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {
namespace io {

// A simple ZIP archive writer that writes directly to a tsl::WritableFile and
// supports DEFLATE compression.
class ZipWriter {
 public:
  // Static factory method. Takes ownership of the file.
  static absl::StatusOr<ZipWriter> Create(std::unique_ptr<WritableFile> file);

  ZipWriter(ZipWriter&& other) noexcept;
  ZipWriter& operator=(ZipWriter&& other) noexcept;

  ZipWriter(const ZipWriter&) = delete;
  ZipWriter& operator=(const ZipWriter&) = delete;

  // Destructor will call Finish() if it has not been called explicitly.
  // Any error during finalization in the destructor will be logged.
  ~ZipWriter();

  // Adds a file to the ZIP archive. The file content will be compressed
  // using DEFLATE.
  absl::Status AddFile(std::string name, absl::string_view content);

  // Finalizes the ZIP archive by writing the Central Directory.
  // Must be called exactly once after all files have been added.
  // Rvalue-qualified to enforce single-use and ownership transfer.
  absl::Status Finish() &&;

 private:
  explicit ZipWriter(std::unique_ptr<WritableFile> file,
                     int64_t initial_offset);

  struct FileInfo {
    std::string name;
    uint32_t offset;
    uint32_t crc;
    uint32_t compressed_size;
    uint32_t uncompressed_size;
  };

  absl::Status AppendData(absl::string_view data);
  absl::Status Append16(uint16_t val);
  absl::Status Append32(uint32_t val);
  absl::Status FinishInternal();

  std::unique_ptr<WritableFile> file_;
  uint64_t current_offset_;
  std::vector<FileInfo> files_;
  bool finished_;
};

}  // namespace io
}  // namespace tsl

#endif  // XLA_TSL_LIB_IO_ZIP_WRITER_H_
