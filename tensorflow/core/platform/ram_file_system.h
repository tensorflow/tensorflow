/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_

// Implementation of an in-memory TF filesystem for simple prototyping (e.g.
// via Colab). The TPU TF server does not have local filesystem access, which
// makes it difficult to provide Colab tutorials: users must have GCS access
// and sign-in in order to try out an example.
//
// Files are implemented on top of std::string. Directories, as with GCS or S3,
// are implicit based on the existence of child files. Multiple files may
// reference a single FS location, though no thread-safety guarantees are
// provided.

#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#undef CopyFile
#undef TranslateName
#endif

namespace tensorflow {

class RamRandomAccessFile : public RandomAccessFile, public WritableFile {
 public:
  RamRandomAccessFile(std::string name, std::shared_ptr<std::string> cord)
      : name_(name), data_(cord) {}
  ~RamRandomAccessFile() override {}

  Status Name(StringPiece* result) const override {
    *result = name_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    if (offset >= data_->size()) {
      return errors::OutOfRange("");
    }

    uint64 left = std::min(static_cast<uint64>(n), data_->size() - offset);
    auto start = data_->begin() + offset;
    auto end = data_->begin() + offset + left;

    std::copy(start, end, scratch);
    *result = StringPiece(scratch, left);

    // In case of a partial read, we must still fill `result`, but also return
    // OutOfRange.
    if (left < n) {
      return errors::OutOfRange("");
    }
    return Status::OK();
  }

  Status Append(StringPiece data) override {
    data_->append(data.data(), data.size());
    return Status::OK();
  }

#if defined(PLATFORM_GOOGLE)
  Status Append(const absl::Cord& cord) override {
    data_->append(cord.char_begin(), cord.char_end());
    return Status::OK();
  }
#endif

  Status Close() override { return Status::OK(); }
  Status Flush() override { return Status::OK(); }
  Status Sync() override { return Status::OK(); }

  Status Tell(int64* position) override {
    *position = -1;
    return errors::Unimplemented("This filesystem does not support Tell()");
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RamRandomAccessFile);
  string name_;
  std::shared_ptr<std::string> data_;
};

class RamFileSystem : public FileSystem {
 public:
  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    mutex_lock m(mu_);
    if (fs_.find(fname) == fs_.end()) {
      return errors::NotFound("");
    }
    *result = std::unique_ptr<RandomAccessFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    mutex_lock m(mu_);
    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }
  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    mutex_lock m(mu_);
    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return Status::OK();
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented("");
  }

  Status FileExists(const string& fname) override {
    FileStatistics stat;
    return Stat(fname, &stat);
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    mutex_lock m(mu_);
    auto it = fs_.lower_bound(dir);
    while (it != fs_.end() && absl::StartsWith(it->first, dir)) {
      result->push_back(it->first);
      ++it;
    }

    return Status::OK();
  }

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override {
    mutex_lock m(mu_);
    Env* env = Env::Default();
    for (auto it = fs_.begin(); it != fs_.end(); ++it) {
      if (env->MatchPath(it->first, pattern)) {
        results->push_back(it->first);
      }
    }
    return Status::OK();
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    mutex_lock m(mu_);
    auto it = fs_.lower_bound(fname);
    if (it == fs_.end()) {
      return errors::NotFound("");
    }

    if (it->first == fname) {
      stat->is_directory = false;
      stat->length = fs_[fname]->size();
      stat->mtime_nsec = 0;
      return Status::OK();
    }

    stat->is_directory = true;
    stat->length = 0;
    stat->mtime_nsec = 0;
    return Status::OK();
  }

  Status DeleteFile(const string& fname) override {
    mutex_lock m(mu_);
    if (fs_.find(fname) != fs_.end()) {
      fs_.erase(fname);
      return Status::OK();
    }

    return errors::NotFound("");
  }

  Status CreateDir(const string& dirname) override { return Status::OK(); }

  Status RecursivelyCreateDir(const string& dirname) override {
    return Status::OK();
  }

  Status DeleteDir(const string& dirname) override { return Status::OK(); }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    mutex_lock m(mu_);
    if (fs_.find(fname) != fs_.end()) {
      *file_size = fs_[fname]->size();
      return Status::OK();
    }
    return errors::NotFound("");
  }

  Status RenameFile(const string& src, const string& target) override {
    mutex_lock m(mu_);
    if (fs_.find(src) != fs_.end()) {
      fs_[target] = fs_[src];
      fs_.erase(fs_.find(src));
      return Status::OK();
    }
    return errors::NotFound("");
  }

  RamFileSystem() {}
  ~RamFileSystem() override {}

 private:
  mutex mu_;
  std::map<string, std::shared_ptr<std::string>> fs_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_RAM_FILE_SYSTEM_H_
