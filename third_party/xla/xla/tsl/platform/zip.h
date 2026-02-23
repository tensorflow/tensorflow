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

#ifndef XLA_TSL_PLATFORM_ZIP_H_
#define XLA_TSL_PLATFORM_ZIP_H_

#include <memory>
#include <string>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/io/zero_copy_stream.h"

namespace tsl {
class RandomAccessFile;
}  // namespace tsl

namespace tsl {
namespace zip {

class ZipArchive {
 public:
  virtual ~ZipArchive() = default;

  // Returns a list of all files/directories in the archive.
  virtual absl::StatusOr<std::vector<std::string>> GetEntries() = 0;

  // Reads the entire content of a specific entry.
  virtual absl::StatusOr<std::string> GetContents(absl::string_view entry) = 0;

  // Opens a file in the archive for reading.
  virtual absl::StatusOr<std::unique_ptr<tsl::RandomAccessFile>> Open(
      absl::string_view entry) = 0;

  // Returns a ZeroCopyInputStream for reading a specific entry.
  virtual absl::StatusOr<std::unique_ptr<google::protobuf::io::ZeroCopyInputStream>>
  GetZeroCopyInputStream(absl::string_view entry) = 0;
};

// Opens a zip archive from the given path.
absl::StatusOr<std::unique_ptr<ZipArchive>> Open(absl::string_view path);

}  // namespace zip
}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_ZIP_H_
