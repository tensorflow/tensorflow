#ifndef TENSORFLOW_PUBLIC_ENV_H_
#define TENSORFLOW_PUBLIC_ENV_H_

#include <stdint.h>
#include <string>
#include <vector>
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {

class RandomAccessFile;
class Thread;
class ThreadOptions;
class WritableFile;

/// \brief An interface used by the tensorflow implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations are safe for concurrent access from
/// multiple threads without any external synchronization.
class Env {
 public:
  Env() {}
  virtual ~Env();

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  virtual Status NewRandomAccessFile(const string& fname,
                                     RandomAccessFile** result) = 0;

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  virtual Status NewWritableFile(const string& fname,
                                 WritableFile** result) = 0;

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  virtual Status NewAppendableFile(const string& fname,
                                   WritableFile** result) = 0;

  /// Returns true iff the named file exists.
  virtual bool FileExists(const string& fname) = 0;

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  virtual Status GetChildren(const string& dir,
                             std::vector<string>* result) = 0;

  /// Deletes the named file.
  virtual Status DeleteFile(const string& fname) = 0;

  /// Creates the specified directory.
  virtual Status CreateDir(const string& dirname) = 0;

  /// Deletes the specified directory.
  virtual Status DeleteDir(const string& dirname) = 0;

  /// Stores the size of fname in *file_size.
  virtual Status GetFileSize(const string& fname, uint64* file_size) = 0;

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  virtual Status RenameFile(const string& src, const string& target) = 0;

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /// \brief Returns the number of micro-seconds since some fixed point in
  /// time. Only useful for computing deltas of time.
  virtual uint64 NowMicros() = 0;

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int micros) = 0;

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const string& name,
                              std::function<void()> fn) TF_MUST_USE_RESULT = 0;

 private:
  /// No copying allowed
  Env(const Env&);
  void operator=(const Env&);
};

/// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() {}
  virtual ~RandomAccessFile();

  /// \brief Reads up to "n" bytes from the file starting at "offset".
  ///
  /// "scratch[0..n-1]" may be written by this routine.  Sets "*result"
  /// to the data that was read (including if fewer than "n" bytes were
  /// successfully read).  May set "*result" to point at data in
  /// "scratch[0..n-1]", so "scratch[0..n-1]" must be live when
  /// "*result" is used.
  ///
  /// On OK returned status: "n" bytes have been stored in "*result".
  /// On non-OK returned status: [0..n] bytes have been stored in "*result".
  ///
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in "*result"
  /// because of EOF.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual Status Read(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const = 0;

 private:
  /// No copying allowed
  RandomAccessFile(const RandomAccessFile&);
  void operator=(const RandomAccessFile&);
};

/// \brief A file abstraction for sequential writing.
///
/// The implementation must provide buffering since callers may append
/// small fragments at a time to the file.
class WritableFile {
 public:
  WritableFile() {}
  virtual ~WritableFile();

  virtual Status Append(const StringPiece& data) = 0;
  virtual Status Close() = 0;
  virtual Status Flush() = 0;
  virtual Status Sync() = 0;

 private:
  /// No copying allowed
  WritableFile(const WritableFile&);
  void operator=(const WritableFile&);
};

/// \brief An implementation of Env that forwards all calls to another Env.
///
/// May be useful to clients who wish to override just part of the
/// functionality of another Env.
class EnvWrapper : public Env {
 public:
  /// Initializes an EnvWrapper that delegates all calls to *t
  explicit EnvWrapper(Env* t) : target_(t) {}
  virtual ~EnvWrapper();

  /// Returns the target to which this Env forwards all calls
  Env* target() const { return target_; }

  // The following text is boilerplate that forwards all methods to target()
  Status NewRandomAccessFile(const string& f, RandomAccessFile** r) override {
    return target_->NewRandomAccessFile(f, r);
  }
  Status NewWritableFile(const string& f, WritableFile** r) override {
    return target_->NewWritableFile(f, r);
  }
  Status NewAppendableFile(const string& f, WritableFile** r) override {
    return target_->NewAppendableFile(f, r);
  }
  bool FileExists(const string& f) override { return target_->FileExists(f); }
  Status GetChildren(const string& dir, std::vector<string>* r) override {
    return target_->GetChildren(dir, r);
  }
  Status DeleteFile(const string& f) override { return target_->DeleteFile(f); }
  Status CreateDir(const string& d) override { return target_->CreateDir(d); }
  Status DeleteDir(const string& d) override { return target_->DeleteDir(d); }
  Status GetFileSize(const string& f, uint64* s) override {
    return target_->GetFileSize(f, s);
  }
  Status RenameFile(const string& s, const string& t) override {
    return target_->RenameFile(s, t);
  }
  uint64 NowMicros() override { return target_->NowMicros(); }
  void SleepForMicroseconds(int micros) override {
    target_->SleepForMicroseconds(micros);
  }
  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return target_->StartThread(thread_options, name, fn);
  }

 private:
  Env* target_;
};

class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  /// No copying allowed
  Thread(const Thread&);
  void operator=(const Thread&);
};

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
};

/// A utility routine: reads contents of named file into *data
Status ReadFileToString(Env* env, const string& fname, string* data);

/// A utility routine: write contents of "data" to file named "fname"
/// (overwriting existing contents, if any).
Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into *proto.
Status ReadBinaryProto(Env* env, const string& fname,
                       ::tensorflow::protobuf::MessageLite* proto);

}  // namespace tensorflow

#endif  // TENSORFLOW_PUBLIC_ENV_H_
