/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_FILE_SYSTEM_HELPER_H_
#define XLA_TSL_PLATFORM_FILE_SYSTEM_HELPER_H_

#include <algorithm>
#include <vector>

#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

namespace tsl {

class FileSystem;
class Env;

namespace internal {

// Given a pattern, stores in 'results' the set of paths (in the given file
// system) that match that pattern.
//
// This helper may be used by implementations of FileSystem::GetMatchingPaths()
// in order to provide parallel scanning of subdirectories (except on iOS).
//
// Arguments:
//   fs: may not be null and will be used to identify directories and list
//       their contents.
//   env: may not be null and will be used to check if a match has been found.
//   pattern: see FileSystem::GetMatchingPaths() for details.
//   results: will be cleared and may not be null.
//
// Returns an error status if any call to 'fs' failed.
absl::Status GetMatchingPaths(FileSystem* fs, Env* env, const string& pattern,
                              std::vector<string>* results);

// Given a file path, determines whether the file exists. This helper simplifies
// the use of Env::FileExists.
//
// Arguments:
//   env: may not be null.
//   fname: the file path to look up
//
// Returns true if the file exists, false if it does not exist, or an error
// Status.
absl::StatusOr<bool> FileExists(Env* env, const string& fname);

}  // namespace internal

//  A CopyingOutputStream wrapper over tsl::WritableFile. It adapts the
//  CopyingOutputStream interface to the WritableFile's Append method. This
//  allows using a WritableFile with systems expecting a CopyingOutputStream and
//  convert it to a ZeroCopyOutputStream easily using
//  CopyingOutputStreamAdaptor.
class WritableFileCopyingOutputStream
    : public tsl::protobuf::io::CopyingOutputStream {
 public:
  explicit WritableFileCopyingOutputStream(WritableFile* file)
      : tsl::protobuf::io::CopyingOutputStream(), file_(file) {}

  bool Write(const void* buffer, int size) override {
    return file_
        ->Append(absl::string_view(static_cast<const char*>(buffer), size))
        .ok();
  }

 private:
  WritableFile* file_;
};

// A CopyingInputStream wrapper over tsl::RandomAccessFile. It adapts the
// CopyingInputStream interface to the RandomAccessFile's Read method. This
// allows using a RandomAccessFile with systems expecting a CopyingInputStream
// and convert it to a ZeroCopyInputStream easily using
// CopyingInputStreamAdaptor.
class RandomAccessFileCopyingInputStream
    : public protobuf::io::CopyingInputStream {
 public:
  explicit RandomAccessFileCopyingInputStream(RandomAccessFile* file)
      : file_(file), position_(0) {}

  int Read(void* buffer, int size) override {
    if (!file_) {
      return -1;
    }

    absl::string_view result;
    auto status =
        file_->Read(position_, size, &result, static_cast<char*>(buffer));

    if (!status.ok() && status.code() != absl::StatusCode::kOutOfRange) {
      return -1;
    }
    // Documentation of RandomAccessFile::Read warns that *result can also point
    // at something else than buffer. Checking that explicitly and copy the
    // result manually if needed.
    if (result.data() != buffer) {
      // Data was not written directly to the buffer. Copy it ourselves.
      if (result.size() > size) {
        return -1;
      }
      std::copy(result.begin(), result.end(), static_cast<char*>(buffer));
    }

    position_ += result.size();
    return result.size();
  }

 private:
  RandomAccessFile* file_;
  int64_t position_;
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_FILE_SYSTEM_HELPER_H_
