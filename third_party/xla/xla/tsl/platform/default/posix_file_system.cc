/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>

#if defined(__linux__)
#include <sys/sendfile.h>
#endif
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "xla/tsl/platform/default/posix_file_system.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system_helper.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/strcat.h"

namespace tsl {

using ::tsl::errors::IOError;

// 128KB of copy buffer
constexpr size_t kPosixCopyFileBufferSize = 128 * 1024;

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}
  ~PosixRandomAccessFile() override {
    if (close(fd_) < 0) {
      LOG(ERROR) << "close() failed: " << strerror(errno);
    }
  }

  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }

  absl::Status Read(uint64 offset, size_t n, absl::string_view* result,
                    char* scratch) const override {
    absl::Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      // Some platforms, notably macs, throw EINVAL if pread is asked to read
      // more than fits in a 32-bit integer.
      size_t requested_read_length;
      if (n > INT32_MAX) {
        requested_read_length = INT32_MAX;
      } else {
        requested_read_length = n;
      }
      ssize_t r =
          pread(fd_, dst, requested_read_length, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = absl::Status(absl::StatusCode::kOutOfRange,
                         "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = absl::string_view(scratch, dst - scratch);
    return s;
  }

#if defined(TF_CORD_SUPPORT)
  absl::Status Read(uint64 offset, size_t n, absl::Cord* cord) const override {
    if (n == 0) {
      return absl::OkStatus();
    }
    if (n < 0) {
      return errors::InvalidArgument(
          "Attempting to read ", n,
          " bytes. You cannot read a negative number of bytes.");
    }

    char* scratch = new char[n];
    if (scratch == nullptr) {
      return errors::ResourceExhausted("Unable to allocate ", n,
                                       " bytes for file reading.");
    }

    absl::string_view tmp;
    absl::Status s = Read(offset, n, &tmp, scratch);

    absl::Cord tmp_cord = absl::MakeCordFromExternal(
        absl::string_view(static_cast<char*>(scratch), tmp.size()),
        [scratch](absl::string_view) { delete[] scratch; });
    cord->Append(tmp_cord);
    return s;
  }
#endif
};

class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  PosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  absl::Status Append(absl::string_view data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return absl::OkStatus();
  }

#if defined(TF_CORD_SUPPORT)
  // \brief Append 'cord' to the file.
  absl::Status Append(const absl::Cord& cord) override {
    for (const auto& chunk : cord.Chunks()) {
      size_t r = fwrite(chunk.data(), 1, chunk.size(), file_);
      if (r != chunk.size()) {
        return IOError(filename_, errno);
      }
    }
    return absl::OkStatus();
  }
#endif

  absl::Status Close() override {
    if (file_ == nullptr) {
      return IOError(filename_, EBADF);
    }
    absl::Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  absl::Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return absl::OkStatus();
  }

  absl::Status Name(absl::string_view* result) const override {
    *result = filename_;
    return absl::OkStatus();
  }

  absl::Status Sync() override {
    absl::Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }

  absl::Status Tell(int64_t* position) override {
    absl::Status s;
    *position = ftell(file_);

    if (*position == -1) {
      s = IOError(filename_, errno);
    }

    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    munmap(const_cast<void*>(address_), length_);
  }
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

absl::Status PosixFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  absl::Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

absl::Status PosixFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  absl::Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

absl::Status PosixFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  absl::Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

absl::Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  absl::Status s = absl::OkStatus();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
    }
    if (close(fd) < 0) {
      s = IOError(fname, errno);
    }
  }
  return s;
}

absl::Status PosixFileSystem::FileExists(const string& fname,
                                         TransactionToken* token) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return absl::OkStatus();
  }
  return errors::NotFound(fname, " not found");
}

absl::Status PosixFileSystem::GetChildren(const string& dir,
                                          TransactionToken* token,
                                          std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    absl::string_view basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  if (closedir(d) < 0) {
    return IOError(dir, errno);
  }
  return absl::OkStatus();
}

absl::Status PosixFileSystem::GetMatchingPaths(const string& pattern,
                                               TransactionToken* token,
                                               std::vector<string>* results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

absl::Status PosixFileSystem::DeleteFile(const string& fname,
                                         TransactionToken* token) {
  absl::Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

absl::Status PosixFileSystem::CreateDir(const string& name,
                                        TransactionToken* token) {
  string translated = TranslateName(name);
  if (translated.empty()) {
    return errors::AlreadyExists(name);
  }
  if (mkdir(translated.c_str(), 0755) != 0) {
    return IOError(name, errno);
  }
  return absl::OkStatus();
}

absl::Status PosixFileSystem::DeleteDir(const string& name,
                                        TransactionToken* token) {
  absl::Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

absl::Status PosixFileSystem::GetFileSize(const string& fname,
                                          TransactionToken* token,
                                          uint64* size) {
  absl::Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

absl::Status PosixFileSystem::Stat(const string& fname, TransactionToken* token,
                                   FileStatistics* stats) {
  absl::Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

absl::Status PosixFileSystem::RenameFile(const string& src,
                                         const string& target,
                                         TransactionToken* token) {
  absl::Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

absl::Status PosixFileSystem::CopyFile(const string& src, const string& target,
                                       TransactionToken* token) {
  string translated_src = TranslateName(src);
  struct stat sbuf;
  if (stat(translated_src.c_str(), &sbuf) != 0) {
    return IOError(src, errno);
  }
  int src_fd = open(translated_src.c_str(), O_RDONLY);
  if (src_fd < 0) {
    return IOError(src, errno);
  }
  string translated_target = TranslateName(target);
  // O_WRONLY | O_CREAT | O_TRUNC:
  //   Open file for write and if file does not exist, create the file.
  //   If file exists, truncate its size to 0.
  // When creating file, use the same permissions as original
  mode_t mode = sbuf.st_mode & (S_IRWXU | S_IRWXG | S_IRWXO);
  int target_fd =
      open(translated_target.c_str(), O_WRONLY | O_CREAT | O_TRUNC, mode);
  if (target_fd < 0) {
    close(src_fd);
    return IOError(target, errno);
  }
  int rc = 0;
  off_t offset = 0;
  std::unique_ptr<char[]> buffer(new char[kPosixCopyFileBufferSize]);
  while (offset < sbuf.st_size) {
    // Use uint64 for safe compare SSIZE_MAX
    uint64 chunk = sbuf.st_size - offset;
    if (chunk > SSIZE_MAX) {
      chunk = SSIZE_MAX;
    }
#if defined(__linux__) && !defined(__ANDROID__)
    rc = sendfile(target_fd, src_fd, &offset, static_cast<size_t>(chunk));
#else
    if (chunk > kPosixCopyFileBufferSize) {
      chunk = kPosixCopyFileBufferSize;
    }
    rc = read(src_fd, buffer.get(), static_cast<size_t>(chunk));
    if (rc <= 0) {
      break;
    }
    rc = write(target_fd, buffer.get(), static_cast<size_t>(chunk));
    offset += chunk;
#endif
    if (rc <= 0) {
      break;
    }
  }

  absl::Status result = absl::OkStatus();
  if (rc < 0) {
    result = IOError(target, errno);
  }

  // Keep the error code
  rc = close(target_fd);
  if (rc < 0 && result == absl::OkStatus()) {
    result = IOError(target, errno);
  }
  rc = close(src_fd);
  if (rc < 0 && result == absl::OkStatus()) {
    result = IOError(target, errno);
  }

  return result;
}

}  // namespace tsl
