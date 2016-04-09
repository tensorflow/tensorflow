/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"

namespace tensorflow {

namespace {

error::Code ErrnoToCode(int err_number) {
  error::Code code;
  switch (err_number) {
    case 0:
      code = error::OK;
      break;
    case EINVAL:        // Invalid argument
    case ENAMETOOLONG:  // Filename too long
    case E2BIG:         // Argument list too long
    case EDESTADDRREQ:  // Destination address required
    case EDOM:          // Mathematics argument out of domain of function
    case EFAULT:        // Bad address
    case EILSEQ:        // Illegal byte sequence
    case ENOPROTOOPT:   // Protocol not available
    case ENOSTR:        // Not a STREAM
    case ENOTSOCK:      // Not a socket
    case ENOTTY:        // Inappropriate I/O control operation
    case EPROTOTYPE:    // Protocol wrong type for socket
    case ESPIPE:        // Invalid seek
      code = error::INVALID_ARGUMENT;
      break;
    case ETIMEDOUT:  // Connection timed out
    case ETIME:      // Timer expired
      code = error::DEADLINE_EXCEEDED;
      break;
    case ENODEV:  // No such device
    case ENOENT:  // No such file or directory
    case ENXIO:   // No such device or address
    case ESRCH:   // No such process
      code = error::NOT_FOUND;
      break;
    case EEXIST:         // File exists
    case EADDRNOTAVAIL:  // Address not available
    case EALREADY:       // Connection already in progress
      code = error::ALREADY_EXISTS;
      break;
    case EPERM:   // Operation not permitted
    case EACCES:  // Permission denied
    case EROFS:   // Read only file system
      code = error::PERMISSION_DENIED;
      break;
    case ENOTEMPTY:   // Directory not empty
    case EISDIR:      // Is a directory
    case ENOTDIR:     // Not a directory
    case EADDRINUSE:  // Address already in use
    case EBADF:       // Invalid file descriptor
    case EBUSY:       // Device or resource busy
    case ECHILD:      // No child processes
    case EISCONN:     // Socket is connected
    case ENOTBLK:     // Block device required
    case ENOTCONN:    // The socket is not connected
    case EPIPE:       // Broken pipe
    case ESHUTDOWN:   // Cannot send after transport endpoint shutdown
    case ETXTBSY:     // Text file busy
      code = error::FAILED_PRECONDITION;
      break;
    case ENOSPC:   // No space left on device
    case EDQUOT:   // Disk quota exceeded
    case EMFILE:   // Too many open files
    case EMLINK:   // Too many links
    case ENFILE:   // Too many open files in system
    case ENOBUFS:  // No buffer space available
    case ENODATA:  // No message is available on the STREAM read queue
    case ENOMEM:   // Not enough space
    case ENOSR:    // No STREAM resources
    case EUSERS:   // Too many users
      code = error::RESOURCE_EXHAUSTED;
      break;
    case EFBIG:      // File too large
    case EOVERFLOW:  // Value too large to be stored in data type
    case ERANGE:     // Result too large
      code = error::OUT_OF_RANGE;
      break;
    case ENOSYS:           // Function not implemented
    case ENOTSUP:          // Operation not supported
    case EAFNOSUPPORT:     // Address family not supported
    case EPFNOSUPPORT:     // Protocol family not supported
    case EPROTONOSUPPORT:  // Protocol not supported
    case ESOCKTNOSUPPORT:  // Socket type not supported
    case EXDEV:            // Improper link
      code = error::UNIMPLEMENTED;
      break;
    case EAGAIN:        // Resource temporarily unavailable
    case ECONNREFUSED:  // Connection refused
    case ECONNABORTED:  // Connection aborted
    case ECONNRESET:    // Connection reset
    case EINTR:         // Interrupted function call
    case EHOSTDOWN:     // Host is down
    case EHOSTUNREACH:  // Host is unreachable
    case ENETDOWN:      // Network is down
    case ENETRESET:     // Connection aborted by network
    case ENETUNREACH:   // Network unreachable
    case ENOLCK:        // No locks available
    case ENOLINK:       // Link has been severed
#if !defined(__APPLE__)
    case ENONET:  // Machine is not on the network
#endif
      code = error::UNAVAILABLE;
      break;
    case EDEADLK:  // Resource deadlock avoided
    case ESTALE:   // Stale file handle
      code = error::ABORTED;
      break;
    case ECANCELED:  // Operation cancelled
      code = error::CANCELLED;
      break;
    // NOTE: If you get any of the following (especially in a
    // reproducible way) and can propose a better mapping,
    // please email the owners about updating this mapping.
    case EBADMSG:      // Bad message
    case EIDRM:        // Identifier removed
    case EINPROGRESS:  // Operation in progress
    case EIO:          // I/O error
    case ELOOP:        // Too many levels of symbolic links
    case ENOEXEC:      // Exec format error
    case ENOMSG:       // No message of the desired type
    case EPROTO:       // Protocol error
    case EREMOTE:      // Object is remote
      code = error::UNKNOWN;
      break;
    default: {
      code = error::UNKNOWN;
      break;
    }
  }
  return code;
}

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}
  ~PosixRandomAccessFile() override { close(fd_); }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }
};

class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  PosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~PosixWritableFile() override {
    if (file_ != NULL) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(const StringPiece& data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = NULL;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() { munmap(const_cast<void*>(address_), length_); }
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

}  // namespace

Status PosixFileSystem::NewRandomAccessFile(const string& fname,
                                            RandomAccessFile** result) {
  string translated_fname = TranslateName(fname);
  *result = NULL;
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    *result = new PosixRandomAccessFile(translated_fname, fd);
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        WritableFile** result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == NULL) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new PosixWritableFile(translated_fname, f);
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(const string& fname,
                                          WritableFile** result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == NULL) {
    *result = NULL;
    s = IOError(fname, errno);
  } else {
    *result = new PosixWritableFile(translated_fname, f);
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, ReadOnlyMemoryRegion** result) {
  string translated_fname = TranslateName(fname);
  *result = nullptr;
  Status s = Status::OK();
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
      *result = new PosixReadOnlyMemoryRegion(address, st.st_size);
    }
    close(fd);
  }
  return s;
}

bool PosixFileSystem::FileExists(const string& fname) {
  return access(TranslateName(fname).c_str(), F_OK) == 0;
}

Status PosixFileSystem::GetChildren(const string& dir,
                                    std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == NULL) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != NULL) {
    StringPiece basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  closedir(d);
  return Status::OK();
}

Status PosixFileSystem::DeleteFile(const string& fname) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const string& name) {
  Status result;
  if (mkdir(TranslateName(name).c_str(), 0755) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::DeleteDir(const string& name) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const string& fname, uint64* size) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::RenameFile(const string& src, const string& target) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

Status IOError(const string& context, int err_number) {
  auto code = ErrnoToCode(err_number);
  if (code == error::UNKNOWN) {
    return Status(code, strings::StrCat(context, "; ", strerror(err_number)));
  } else {
    return Status(code, context);
  }
}

}  // namespace tensorflow
