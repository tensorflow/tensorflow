#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <thread>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/env.h"

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

static Status IOError(const string& context, int err_number) {
  auto code = ErrnoToCode(err_number);
  if (code == error::UNKNOWN) {
    return Status(ErrnoToCode(err_number),
                  context + "; " + strerror(err_number));
  } else {
    return Status(ErrnoToCode(err_number), context);
  }
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

class StdThread : public Thread {
 public:
  // name and thread_options are both ignored.
  StdThread(const ThreadOptions& thread_options, const string& name,
            std::function<void()> fn)
      : thread_(fn) {}
  ~StdThread() { thread_.join(); }

 private:
  std::thread thread_;
};

class PosixEnv : public Env {
 public:
  PosixEnv() {}

  ~PosixEnv() override { LOG(FATAL) << "Env::Default() must not be destroyed"; }

  Status NewRandomAccessFile(const string& fname,
                             RandomAccessFile** result) override {
    *result = NULL;
    Status s;
    int fd = open(fname.c_str(), O_RDONLY);
    if (fd < 0) {
      s = IOError(fname, errno);
    } else {
      *result = new PosixRandomAccessFile(fname, fd);
    }
    return s;
  }

  Status NewWritableFile(const string& fname, WritableFile** result) override {
    Status s;
    FILE* f = fopen(fname.c_str(), "w");
    if (f == NULL) {
      *result = NULL;
      s = IOError(fname, errno);
    } else {
      *result = new PosixWritableFile(fname, f);
    }
    return s;
  }

  Status NewAppendableFile(const string& fname,
                           WritableFile** result) override {
    Status s;
    FILE* f = fopen(fname.c_str(), "a");
    if (f == NULL) {
      *result = NULL;
      s = IOError(fname, errno);
    } else {
      *result = new PosixWritableFile(fname, f);
    }
    return s;
  }

  bool FileExists(const string& fname) override {
    return access(fname.c_str(), F_OK) == 0;
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    result->clear();
    DIR* d = opendir(dir.c_str());
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

  Status DeleteFile(const string& fname) override {
    Status result;
    if (unlink(fname.c_str()) != 0) {
      result = IOError(fname, errno);
    }
    return result;
  }

  Status CreateDir(const string& name) override {
    Status result;
    if (mkdir(name.c_str(), 0755) != 0) {
      result = IOError(name, errno);
    }
    return result;
  }

  Status DeleteDir(const string& name) override {
    Status result;
    if (rmdir(name.c_str()) != 0) {
      result = IOError(name, errno);
    }
    return result;
  }

  Status GetFileSize(const string& fname, uint64* size) override {
    Status s;
    struct stat sbuf;
    if (stat(fname.c_str(), &sbuf) != 0) {
      *size = 0;
      s = IOError(fname, errno);
    } else {
      *size = sbuf.st_size;
    }
    return s;
  }

  Status RenameFile(const string& src, const string& target) override {
    Status result;
    if (rename(src.c_str(), target.c_str()) != 0) {
      result = IOError(src, errno);
    }
    return result;
  }

  uint64 NowMicros() override {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return static_cast<uint64>(tv.tv_sec) * 1000000 + tv.tv_usec;
  }

  void SleepForMicroseconds(int micros) override { usleep(micros); }

  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return new StdThread(thread_options, name, fn);
  }
};

}  // namespace
#if defined(PLATFORM_POSIX) || defined(__ANDROID__)
Env* Env::Default() {
  static Env* default_env = new PosixEnv;
  return default_env;
}
#endif

}  // namespace tensorflow
