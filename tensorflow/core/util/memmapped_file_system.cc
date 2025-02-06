/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/util/memmapped_file_system.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/memmapped_file_system.pb.h"

namespace tensorflow {

namespace {

uint64 DecodeUint64LittleEndian(const uint8* buffer) {
  uint64 result = 0;
  for (int i = 0; i < static_cast<int>(sizeof(uint64)); ++i) {
    result |= static_cast<uint64>(buffer[i]) << (8 * i);
  }
  return result;
}

}  // namespace

namespace {

class ReadOnlyMemoryRegionFromMemmapped : public ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegionFromMemmapped(const void* data, uint64 length)
      : data_(data), length_(length) {}
  ~ReadOnlyMemoryRegionFromMemmapped() override = default;
  const void* data() override { return data_; }
  uint64 length() override { return length_; }

 private:
  const void* const data_;
  const uint64 length_;
  // intentionally copyable
};

class RandomAccessFileFromMemmapped : public RandomAccessFile {
 public:
  RandomAccessFileFromMemmapped(const void* data, uint64 length)
      : data_(data), length_(length) {}

  ~RandomAccessFileFromMemmapped() override = default;

  absl::Status Name(absl::string_view* result) const override {
    return errors::Unimplemented(
        "RandomAccessFileFromMemmapped does not support Name()");
  }

  absl::Status Read(uint64 offset, size_t to_read, absl::string_view* result,
                    char* scratch) const override {
    if (offset >= length_) {
      *result = absl::string_view(scratch, 0);
      return absl::Status(absl::StatusCode::kOutOfRange, "Read after file end");
    }
    const uint64 region_left =
        std::min(length_ - offset, static_cast<uint64>(to_read));
    *result = absl::string_view(reinterpret_cast<const char*>(data_) + offset,
                                region_left);
    return (region_left == to_read)
               ? absl::OkStatus()
               : absl::Status(absl::StatusCode::kOutOfRange,
                              "Read less bytes than requested");
  }

 private:
  const void* const data_;
  const uint64 length_;
  // intentionally copyable
};

}  // namespace

MemmappedFileSystem::MemmappedFileSystem() = default;

absl::Status MemmappedFileSystem::FileExists(const string& fname,
                                             TransactionToken* token) {
  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(fname);
  if (dir_element != directory_.end()) {
    return absl::OkStatus();
  }
  return errors::NotFound(fname, " not found");
}

absl::Status MemmappedFileSystem::NewRandomAccessFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  *result = std::make_unique<RandomAccessFileFromMemmapped>(
      GetMemoryWithOffset(dir_element->second.offset),
      dir_element->second.length);
  return absl::OkStatus();
}

absl::Status MemmappedFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  *result = std::make_unique<ReadOnlyMemoryRegionFromMemmapped>(
      GetMemoryWithOffset(dir_element->second.offset),
      dir_element->second.length);
  return absl::OkStatus();
}

absl::Status MemmappedFileSystem::GetFileSize(const string& filename,
                                              TransactionToken* token,
                                              uint64* size) {
  if (!mapped_memory_) {
    return errors::FailedPrecondition("MemmappedEnv is not initialized");
  }
  const auto dir_element = directory_.find(filename);
  if (dir_element == directory_.end()) {
    return errors::NotFound("Region ", filename, " is not found");
  }
  *size = dir_element->second.length;
  return absl::OkStatus();
}

absl::Status MemmappedFileSystem::Stat(const string& fname,
                                       TransactionToken* token,
                                       FileStatistics* stat) {
  uint64 size;
  auto status = GetFileSize(fname, token, &size);
  if (status.ok()) {
    stat->length = size;
  }
  return status;
}

absl::Status MemmappedFileSystem::NewWritableFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<WritableFile>* wf) {
  return errors::Unimplemented("memmapped format doesn't support writing");
}

absl::Status MemmappedFileSystem::NewAppendableFile(
    const string& filename, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  return errors::Unimplemented("memmapped format doesn't support writing");
}

absl::Status MemmappedFileSystem::GetChildren(const string& filename,
                                              TransactionToken* token,
                                              std::vector<string>* strings) {
  return errors::Unimplemented("memmapped format doesn't support GetChildren");
}

absl::Status MemmappedFileSystem::GetMatchingPaths(
    const string& pattern, TransactionToken* token,
    std::vector<string>* results) {
  return errors::Unimplemented(
      "memmapped format doesn't support GetMatchingPaths");
}

absl::Status MemmappedFileSystem::DeleteFile(const string& filename,
                                             TransactionToken* token) {
  return errors::Unimplemented("memmapped format doesn't support DeleteFile");
}

absl::Status MemmappedFileSystem::CreateDir(const string& dirname,
                                            TransactionToken* token) {
  return errors::Unimplemented("memmapped format doesn't support CreateDir");
}

absl::Status MemmappedFileSystem::DeleteDir(const string& dirname,
                                            TransactionToken* token) {
  return errors::Unimplemented("memmapped format doesn't support DeleteDir");
}

absl::Status MemmappedFileSystem::RenameFile(const string& filename_from,
                                             const string& filename_to,
                                             TransactionToken* token) {
  return errors::Unimplemented("memmapped format doesn't support RenameFile");
}

const void* MemmappedFileSystem::GetMemoryWithOffset(uint64 offset) const {
  return reinterpret_cast<const uint8*>(mapped_memory_->data()) + offset;
}

constexpr const char MemmappedFileSystem::kMemmappedPackagePrefix[];
constexpr const char MemmappedFileSystem::kMemmappedPackageDefaultGraphDef[];

absl::Status MemmappedFileSystem::InitializeFromFile(Env* env,
                                                     const string& filename) {
  TF_RETURN_IF_ERROR(
      env->NewReadOnlyMemoryRegionFromFile(filename, &mapped_memory_));
  directory_.clear();
  if (mapped_memory_->length() <= sizeof(uint64)) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Invalid package size");
  }
  const auto memory_start =
      reinterpret_cast<const uint8*>(mapped_memory_->data());
  const uint64 directory_offset = DecodeUint64LittleEndian(
      memory_start + mapped_memory_->length() - sizeof(uint64));
  if (directory_offset > mapped_memory_->length() - sizeof(uint64)) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Invalid directory offset");
  }
  MemmappedFileSystemDirectory proto_directory;
  if (!ParseProtoUnlimited(
          &proto_directory, memory_start + directory_offset,
          mapped_memory_->length() - directory_offset - sizeof(uint64))) {
    return errors::DataLoss("Corrupted memmapped model file: ", filename,
                            " Can't parse its internal directory");
  }

  // Iterating in reverse order to get lengths of elements;
  uint64 prev_element_offset = directory_offset;
  for (auto element_iter = proto_directory.element().rbegin();
       element_iter != proto_directory.element().rend(); ++element_iter) {
    // Check that the element offset is in the right range.
    if (element_iter->offset() >= prev_element_offset) {
      return errors::DataLoss("Corrupted memmapped model file: ", filename,
                              " Invalid offset of internal component");
    }
    if (!directory_
             .insert(std::make_pair(
                 element_iter->name(),
                 FileRegion(element_iter->offset(), element_iter->length())))
             .second) {
      return errors::DataLoss("Corrupted memmapped model file: ", filename,
                              " Duplicate name of internal component ",
                              element_iter->name());
    }
    prev_element_offset = element_iter->offset();
  }
  return absl::OkStatus();
}

bool MemmappedFileSystem::IsMemmappedPackageFilename(const string& filename) {
  return absl::StartsWith(filename, kMemmappedPackagePrefix);
}

namespace {
bool IsValidRegionChar(char c) {
  return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
         (c >= '0' && c <= '9') || c == '_' || c == '.';
}
}  // namespace

bool MemmappedFileSystem::IsWellFormedMemmappedPackageFilename(
    const string& filename) {
  if (!IsMemmappedPackageFilename(filename)) {
    return false;
  }
  for (char c :
       filename.substr(strlen(kMemmappedPackagePrefix),
                       filename.length() - strlen(kMemmappedPackagePrefix))) {
    if (!IsValidRegionChar(c)) {
      return false;
    }
  }
  return true;
}

MemmappedEnv::MemmappedEnv(Env* env) : EnvWrapper(env) {}

absl::Status MemmappedEnv::GetFileSystemForFile(const string& fname,
                                                FileSystem** result) {
  if (MemmappedFileSystem::IsMemmappedPackageFilename(fname)) {
    if (!memmapped_file_system_) {
      return errors::FailedPrecondition(
          "MemmappedEnv is not initialized from a file.");
    }
    *result = memmapped_file_system_.get();
    return absl::OkStatus();
  }
  return EnvWrapper::GetFileSystemForFile(fname, result);
}

absl::Status MemmappedEnv::GetRegisteredFileSystemSchemes(
    std::vector<string>* schemes) {
  const auto status = EnvWrapper::GetRegisteredFileSystemSchemes(schemes);
  if (status.ok()) {
    schemes->emplace_back(MemmappedFileSystem::kMemmappedPackagePrefix);
  }
  return status;
}

absl::Status MemmappedEnv::InitializeFromFile(const string& package_filename) {
  std::unique_ptr<MemmappedFileSystem> file_system_ptr(new MemmappedFileSystem);
  const auto status =
      file_system_ptr->InitializeFromFile(target(), package_filename);
  if (status.ok()) {
    memmapped_file_system_ = std::move(file_system_ptr);
  }
  return status;
}

}  // namespace tensorflow
