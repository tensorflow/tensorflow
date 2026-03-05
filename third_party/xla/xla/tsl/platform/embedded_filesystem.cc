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

#include "xla/tsl/platform/embedded_filesystem.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/file_system.h"

namespace tsl {

EmbedRandomAccessFile::EmbedRandomAccessFile(absl::string_view name,
                                             absl::string_view contents)
    : name_(name), contents_(contents) {}
EmbedRandomAccessFile::~EmbedRandomAccessFile() = default;

absl::Status EmbedRandomAccessFile::Name(absl::string_view* result) const {
  *result = name_;
  return absl::OkStatus();
}

absl::Status EmbedRandomAccessFile::Read(uint64_t offset,
                                         absl::string_view& result,
                                         absl::Span<char> scratch) const {
  if (offset >= contents_.size()) {
    result = absl::string_view();
    return absl::OutOfRangeError(absl::StrCat("Offset ", offset,
                                              " is out of range for file size ",
                                              contents_.size()));
  }

  size_t bytes_to_read = std::min(scratch.size(), contents_.size() - offset);
  std::copy(contents_.begin() + offset,
            contents_.begin() + offset + bytes_to_read, scratch.begin());
  result = absl::string_view(scratch.data(), bytes_to_read);

  if (bytes_to_read < scratch.size()) {
    return absl::OutOfRangeError(
        absl::StrCat("EOF reached: only read ", bytes_to_read, " bytes."));
  }

  return absl::OkStatus();
}

EmbedFileSystem::~EmbedFileSystem() = default;

absl::Status EmbedFileSystem::EmbedFile(absl::string_view fname,
                                        absl::string_view contents) {
  absl::MutexLock ml(fs_lock_);
  fs_[fname] = contents;
  return absl::OkStatus();
}

absl::Status EmbedFileSystem::NewRandomAccessFile(
    const std::string& fname, std::unique_ptr<RandomAccessFile>* result) {
  absl::MutexLock ml(fs_lock_);
  const auto it = fs_.find(fname);
  if (it == fs_.end()) {
    return absl::NotFoundError(absl::StrCat(fname, " does not exist."));
  }

  *result = std::make_unique<EmbedRandomAccessFile>(fname, it->second);
  return absl::OkStatus();
}

absl::Status EmbedFileSystem::FileExists(const std::string& fname) {
  absl::MutexLock ml(fs_lock_);
  if (!fs_.contains(fname)) {
    return absl::NotFoundError(absl::StrCat(fname, " does not exist."));
  }

  return absl::OkStatus();
}

absl::Status EmbedFileSystem::GetFileSize(const std::string& fname,
                                          uint64_t* file_size) {
  absl::MutexLock ml(fs_lock_);
  if (const auto it = fs_.find(fname); it != fs_.end()) {
    *file_size = it->second.size();
    return absl::OkStatus();
  }

  return absl::NotFoundError(absl::StrCat(fname, " does not exist."));
}

absl::Status EmbedFileSystem::GetMatchingPaths(
    const std::string& pattern, std::vector<std::string>* results) {
  absl::MutexLock ml(fs_lock_);
  for (const auto& [file, contents] : fs_) {
    if (Match(file, pattern)) {
      results->push_back(std::string(file));
    }
  }

  return absl::OkStatus();
}

}  // namespace tsl
