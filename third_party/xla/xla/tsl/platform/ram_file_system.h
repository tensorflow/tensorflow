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

#ifndef XLA_TSL_PLATFORM_RAM_FILE_SYSTEM_H_
#define XLA_TSL_PLATFORM_RAM_FILE_SYSTEM_H_

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

#include "absl/strings/match.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/file_system.h"
#include "xla/tsl/platform/types.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/stringpiece.h"

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#undef CopyFile
#undef TranslateName
#endif

namespace tsl {

class RamRandomAccessFile : public RandomAccessFile, public WritableFile {
 public:
  RamRandomAccessFile(std::string name, std::shared_ptr<std::string> cord)
      : name_(name), data_(cord) {}
  ~RamRandomAccessFile() override {}

  absl::Status Name(absl::string_view* result) const override {
    *result = name_;
    return absl::OkStatus();
  }

  absl::Status Read(uint64 offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    if (offset >= data_->size()) {
      return errors::OutOfRange("");
    }

    uint64 left = std::min(static_cast<uint64>(n), data_->size() - offset);
    auto start = data_->begin() + offset;
    auto end = data_->begin() + offset + left;

    std::copy(start, end, scratch);
    *result = absl::string_view(scratch, left);

    // In case of a partial read, we must still fill `result`, but also return
    // OutOfRange.
    if (left < n) {
      return errors::OutOfRange("");
    }
    return absl::OkStatus();
  }

  absl::Status Append(absl::string_view data) override {
    data_->append(data.data(), data.size());
    return absl::OkStatus();
  }

#if defined(TF_CORD_SUPPORT)
  absl::Status Append(const absl::Cord& cord) override {
    data_->append(cord.char_begin(), cord.char_end());
    return absl::OkStatus();
  }
#endif

  absl::Status Close() override { return absl::OkStatus(); }
  absl::Status Flush() override { return absl::OkStatus(); }
  absl::Status Sync() override { return absl::OkStatus(); }

  absl::Status Tell(int64_t* position) override {
    *position = -1;
    return errors::Unimplemented("This filesystem does not support Tell()");
  }

 private:
  RamRandomAccessFile(const RamRandomAccessFile&) = delete;
  void operator=(const RamRandomAccessFile&) = delete;
  std::string name_;
  std::shared_ptr<std::string> data_;
};

class RamFileSystem : public FileSystem {
 public:
  TF_USE_FILESYSTEM_METHODS_WITH_NO_TRANSACTION_SUPPORT;

  absl::Status NewRandomAccessFile(
      const std::string& fname_, TransactionToken* token,
      std::unique_ptr<RandomAccessFile>* result) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      return errors::NotFound("");
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<RandomAccessFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return absl::OkStatus();
  }

  absl::Status NewWritableFile(const std::string& fname_,
                               TransactionToken* token,
                               std::unique_ptr<WritableFile>* result) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return absl::OkStatus();
  }

  absl::Status NewAppendableFile(
      const std::string& fname_, TransactionToken* token,
      std::unique_ptr<WritableFile>* result) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) == fs_.end()) {
      fs_[fname] = std::make_shared<std::string>();
    }
    if (fs_[fname] == nullptr) {
      return errors::InvalidArgument(fname_, " is a directory.");
    }
    *result = std::unique_ptr<WritableFile>(
        new RamRandomAccessFile(fname, fs_[fname]));
    return absl::OkStatus();
  }

  absl::Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented("");
  }

  absl::Status FileExists(const std::string& fname_,
                          TransactionToken* token) override {
    FileStatistics stat;
    auto fname = StripRamFsPrefix(fname_);

    return Stat(fname, token, &stat);
  }

  absl::Status GetChildren(const std::string& dir_, TransactionToken* token,
                           std::vector<std::string>* result) override {
    mutex_lock m(mu_);
    auto dir = StripRamFsPrefix(dir_);

    auto it = fs_.lower_bound(dir);
    while (it != fs_.end() && StartsWith(it->first, dir)) {
      auto filename = StripPrefix(StripPrefix(it->first, dir), "/");
      // It is not either (a) the parent directory itself or (b) a subdirectory
      if (!filename.empty() && filename.find("/") == std::string::npos) {
        result->push_back(filename);
      }
      ++it;
    }

    return absl::OkStatus();
  }

  absl::Status GetMatchingPaths(const std::string& pattern_,
                                TransactionToken* token,
                                std::vector<std::string>* results) override {
    mutex_lock m(mu_);
    auto pattern = StripRamFsPrefix(pattern_);

    Env* env = Env::Default();
    for (auto it = fs_.begin(); it != fs_.end(); ++it) {
      if (env->MatchPath(it->first, pattern)) {
        results->push_back("ram://" + it->first);
      }
    }
    return absl::OkStatus();
  }

  absl::Status Stat(const std::string& fname_, TransactionToken* token,
                    FileStatistics* stat) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    auto it = fs_.lower_bound(fname);
    if (it == fs_.end() || !StartsWith(it->first, fname)) {
      return errors::NotFound("");
    }

    if (it->first == fname && it->second != nullptr) {
      stat->is_directory = false;
      stat->length = fs_[fname]->size();
      stat->mtime_nsec = 0;
      return absl::OkStatus();
    }

    stat->is_directory = true;
    stat->length = 0;
    stat->mtime_nsec = 0;
    return absl::OkStatus();
  }

  absl::Status DeleteFile(const std::string& fname_,
                          TransactionToken* token) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) != fs_.end()) {
      fs_.erase(fname);
      return absl::OkStatus();
    }

    return errors::NotFound("");
  }

  absl::Status CreateDir(const std::string& dirname_,
                         TransactionToken* token) override {
    mutex_lock m(mu_);
    auto dirname = StripRamFsPrefix(dirname_);

    auto it = fs_.find(dirname);
    if (it != fs_.end() && it->second != nullptr) {
      return errors::AlreadyExists(
          "cannot create directory with same name as an existing file");
    }

    fs_[dirname] = nullptr;
    return absl::OkStatus();
  }

  absl::Status RecursivelyCreateDir(const std::string& dirname_,
                                    TransactionToken* token) override {
    auto dirname = StripRamFsPrefix(dirname_);

    std::vector<std::string> dirs = StrSplit(dirname, "/");
    absl::Status last_status;
    std::string dir = dirs[0];
    last_status = CreateDir(dir, token);

    for (int i = 1; i < dirs.size(); ++i) {
      dir = dir + "/" + dirs[i];
      last_status = CreateDir(dir, token);
    }
    return last_status;
  }

  absl::Status DeleteDir(const std::string& dirname_,
                         TransactionToken* token) override {
    mutex_lock m(mu_);
    auto dirname = StripRamFsPrefix(dirname_);

    auto it = fs_.find(dirname);
    if (it == fs_.end()) {
      return errors::NotFound("");
    }
    if (it->second != nullptr) {
      return errors::InvalidArgument("Not a directory");
    }
    fs_.erase(dirname);

    return absl::OkStatus();
  }

  absl::Status GetFileSize(const std::string& fname_, TransactionToken* token,
                           uint64* file_size) override {
    mutex_lock m(mu_);
    auto fname = StripRamFsPrefix(fname_);

    if (fs_.find(fname) != fs_.end()) {
      if (fs_[fname] == nullptr) {
        return errors::InvalidArgument("Not a file");
      }
      *file_size = fs_[fname]->size();
      return absl::OkStatus();
    }
    return errors::NotFound("");
  }

  absl::Status RenameFile(const std::string& src_, const std::string& target_,
                          TransactionToken* token) override {
    mutex_lock m(mu_);
    auto src = StripRamFsPrefix(src_);
    auto target = StripRamFsPrefix(target_);

    if (fs_.find(src) != fs_.end()) {
      fs_[target] = fs_[src];
      fs_.erase(fs_.find(src));
      return absl::OkStatus();
    }
    return errors::NotFound("");
  }

  RamFileSystem() {}
  ~RamFileSystem() override {}

 private:
  mutex mu_;
  std::map<std::string, std::shared_ptr<std::string>> fs_;

  std::vector<std::string> StrSplit(std::string s, std::string delim) {
    std::vector<std::string> ret;
    size_t curr_pos = 0;
    while ((curr_pos = s.find(delim)) != std::string::npos) {
      ret.push_back(s.substr(0, curr_pos));
      s.erase(0, curr_pos + delim.size());
    }
    ret.push_back(s);
    return ret;
  }

  bool StartsWith(std::string s, std::string prefix) {
    return absl::StartsWith(s, prefix);
  }

  string StripPrefix(std::string s, std::string prefix) {
    if (absl::StartsWith(s, prefix)) {
      return s.erase(0, prefix.size());
    }
    return s;
  }

  string StripRamFsPrefix(std::string name) {
    std::string s = StripPrefix(name, "ram://");
    if (*(s.rbegin()) == '/') {
      s.pop_back();
    }
    return s;
  }
};

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_RAM_FILE_SYSTEM_H_
