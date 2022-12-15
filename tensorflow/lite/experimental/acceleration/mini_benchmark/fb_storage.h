/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_

#include <errno.h>

#include <cstring>
#include <string>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/base.h"  // from @flatbuffers
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/stderr_reporter.h"
namespace tflite {
namespace acceleration {

// FileStorage wraps storage of data in a file with locking and error handling.
// Locking makes appends and reads atomic, using flock(2).
//
// The locking in this class is not meant for general purpose multiple
// reader/writer support, but primarily for the case where a previous instance
// of a program has not finished and we'd like to not corrupt the file
// unnecessarily.
class FileStorage {
 public:
  FileStorage(absl::string_view path, ErrorReporter* error_reporter);
  // Read contents into buffer_. Returns an error if file exists but cannot be
  // read.
  MinibenchmarkStatus ReadFileIntoBuffer();
  // Append data to file. Resets the in-memory items and returns an error if
  // writing fails in any way.
  //
  // This calls fsync() on the file to guarantee persistence and is hence quite
  // expensive. The assumption is that this is not done often or in a critical
  // path.
  MinibenchmarkStatus AppendDataToFile(absl::string_view data);

 protected:
  std::string path_;
  ErrorReporter* error_reporter_;
  std::string buffer_;
};

// FlatbufferStorage stores several flatbuffer objects in a file. The primary
// usage is for storing mini benchmark results.
//
// Flatbuffers are not designed for easy mutation. This class is append-only.
// The intended usage is to store a log of events like 'start benchmark with
// configuration X', 'benchmark results for X' / 'crash observed with X' that
// are then parsed to make decisions about how to configure TFLite.
//
// The data is stored as consecutive length-prefixed flatbuffers with identifier
// "STO1".
ABSL_CONST_INIT extern const char kFlatbufferStorageIdentifier[];
template <typename T>
class FlatbufferStorage : protected FileStorage {
 public:
  explicit FlatbufferStorage(
      absl::string_view path,
      ErrorReporter* error_reporter = DefaultErrorReporter())
      : FileStorage(path, error_reporter) {}
  // Reads current contents. Returns an error if file is inaccessible or
  // contents are corrupt. The file not existing is not an error.
  MinibenchmarkStatus Read();
  // Get count of objects stored.
  size_t Count() { return contents_.size(); }
  // Get object at index i, i < Count();
  const T* Get(size_t i) { return contents_[i]; }

  // Append a new object to storage and write out to disk. Returns an error if
  // disk write or re-read fails.
  MinibenchmarkStatus Append(flatbuffers::FlatBufferBuilder* fbb,
                             flatbuffers::Offset<T> object);

 private:
  std::vector<const T*> contents_;
};

template <typename T>
MinibenchmarkStatus FlatbufferStorage<T>::Read() {
  contents_.clear();
  MinibenchmarkStatus status = ReadFileIntoBuffer();
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  size_t remaining_size = buffer_.size();
  const uint8_t* current_ptr =
      reinterpret_cast<const uint8_t*>(buffer_.c_str());
  while (remaining_size != 0) {
    if (remaining_size < sizeof(flatbuffers::uoffset_t)) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (remaining size less than "
          "size of uoffset_t)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    flatbuffers::uoffset_t current_size =
        flatbuffers::ReadScalar<flatbuffers::uoffset_t>(current_ptr);
    flatbuffers::Verifier verifier(
        current_ptr, sizeof(flatbuffers::uoffset_t) + current_size);
    if (!verifier.VerifySizePrefixedBuffer<T>(kFlatbufferStorageIdentifier)) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (verifier returned false)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    contents_.push_back(flatbuffers::GetSizePrefixedRoot<T>(current_ptr));
    size_t consumed = sizeof(flatbuffers::uoffset_t) + current_size;
    if (remaining_size < consumed) {
      TF_LITE_REPORT_ERROR(
          error_reporter_,
          "Corrupt size-prefixed flatbuffer file %s (mismatched size "
          "calculation)",
          path_.c_str());
      return kMinibenchmarkCorruptSizePrefixedFlatbufferFile;
    }
    remaining_size -= consumed;
    current_ptr += consumed;
  }
  return kMinibenchmarkSuccess;
}

template <typename T>
MinibenchmarkStatus FlatbufferStorage<T>::Append(
    flatbuffers::FlatBufferBuilder* fbb, flatbuffers::Offset<T> object) {
  contents_.clear();
  fbb->FinishSizePrefixed(object, kFlatbufferStorageIdentifier);
  const char* data = reinterpret_cast<const char*>(fbb->GetBufferPointer());
  size_t size = fbb->GetSize();
  MinibenchmarkStatus status = AppendDataToFile({data, size});
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  return Read();
}

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_FB_STORAGE_H_
