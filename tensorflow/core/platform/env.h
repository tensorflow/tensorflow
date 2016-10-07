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

#ifndef TENSORFLOW_CORE_PLATFORM_ENV_H_
#define TENSORFLOW_CORE_PLATFORM_ENV_H_

#include <stdint.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Thread;
struct ThreadOptions;

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
  Env();
  virtual ~Env() = default;

  /// \brief Returns a default environment suitable for the current operating
  /// system.
  ///
  /// Sophisticated users may wish to provide their own Env
  /// implementation instead of relying on this default environment.
  ///
  /// The result of Default() belongs to this library and must never be deleted.
  static Env* Default();

  /// \brief Returns the FileSystem object to handle operations on the file
  /// specified by 'fname'. The FileSystem object is used as the implementation
  /// for the file system related (non-virtual) functions that follow.
  /// Returned FileSystem object is still owned by the Env object and will
  // (might) be destroyed when the environment is destroyed.
  virtual Status GetFileSystemForFile(const string& fname, FileSystem** result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes);

  // \brief Register a file system for a scheme.
  virtual Status RegisterFileSystem(const string& scheme,
                                    FileSystemRegistry::Factory factory);

  /// \brief Creates a brand new random access read-only file with the
  /// specified name.

  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewRandomAccessFile(const string& fname,
                             std::unique_ptr<RandomAccessFile>* result);

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result);

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used. The file object
  /// shouldn't live longer than the Env object.
  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result);

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used. The memory region
  /// object shouldn't live longer than the Env object.
  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result);

  /// Returns true iff the named file exists.
  bool FileExists(const string& fname);

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  Status GetChildren(const string& dir, std::vector<string>* result);

  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(const string& path, const string& pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual Status GetMatchingPaths(const string& pattern,
                                  std::vector<string>* results);

  /// Deletes the named file.
  Status DeleteFile(const string& fname);

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. undeleted_files and undeleted_dirs stores the number of
  /// files and directories that weren't deleted (unspecified if the return
  /// status is not OK).
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  /// Typical return codes
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  Status DeleteRecursively(const string& dirname, int64* undeleted_files,
                           int64* undeleted_dirs);

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  Status RecursivelyCreateDir(const string& dirname);

  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  Status CreateDir(const string& dirname);

  /// Deletes the specified directory.
  Status DeleteDir(const string& dirname);

  /// Obtains statistics for the given path.
  Status Stat(const string& fname, FileStatistics* stat);

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  Status IsDirectory(const string& fname);

  /// Stores the size of `fname` in `*file_size`.
  Status GetFileSize(const string& fname, uint64* file_size);

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  Status RenameFile(const string& src, const string& target);

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /// \brief Returns the number of micro-seconds since some fixed point in
  /// time. Only useful for computing deltas of time.
  virtual uint64 NowMicros() = 0;

  /// \brief Returns the number of seconds since some fixed point in
  /// time. Only useful for computing deltas of time.
  virtual uint64 NowSeconds() { return NowMicros() / 1000000L; }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64 micros) = 0;

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(const ThreadOptions& thread_options,
                              const string& name,
                              std::function<void()> fn) TF_MUST_USE_RESULT = 0;

  // \brief Schedules the given closure on a thread-pool.
  //
  // NOTE(mrry): This closure may block.
  virtual void SchedClosure(std::function<void()> closure) = 0;

  // \brief Schedules the given closure on a thread-pool after the given number
  // of microseconds.
  //
  // NOTE(mrry): This closure must not block.
  virtual void SchedClosureAfter(int64 micros,
                                 std::function<void()> closure) = 0;

  // \brief Load a dynamic library.
  //
  // Pass "library_filename" to a platform-specific mechanism for dynamically
  // loading a library.  The rules for determining the exact location of the
  // library are platform-specific and are not documented here.
  //
  // On success, returns a handle to the library in "*handle" and returns
  // OK from the function.
  // Otherwise returns nullptr in "*handle" and an error status from the
  // function.
  virtual Status LoadLibrary(const char* library_filename, void** handle) = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

 private:
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  TF_DISALLOW_COPY_AND_ASSIGN(Env);
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

  Status GetFileSystemForFile(const string& fname,
                              FileSystem** result) override {
    return target_->GetFileSystemForFile(fname, result);
  }

  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override {
    return target_->GetRegisteredFileSystemSchemes(schemes);
  }

  Status RegisterFileSystem(const string& scheme,
                            FileSystemRegistry::Factory factory) override {
    return target_->RegisterFileSystem(scheme, factory);
  }

  bool MatchPath(const string& path, const string& pattern) override {
    return target_->MatchPath(path, pattern);
  }

  uint64 NowMicros() override { return target_->NowMicros(); }
  void SleepForMicroseconds(int64 micros) override {
    target_->SleepForMicroseconds(micros);
  }
  Thread* StartThread(const ThreadOptions& thread_options, const string& name,
                      std::function<void()> fn) override {
    return target_->StartThread(thread_options, name, fn);
  }
  void SchedClosure(std::function<void()> closure) override {
    target_->SchedClosure(closure);
  }
  void SchedClosureAfter(int64 micros, std::function<void()> closure) override {
    target_->SchedClosureAfter(micros, closure);
  }
  Status LoadLibrary(const char* library_filename, void** handle) override {
    return target_->LoadLibrary(library_filename, handle);
  }
  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return target_->GetSymbolFromLibrary(handle, symbol_name, symbol);
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
  TF_DISALLOW_COPY_AND_ASSIGN(Thread);
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

/// A utility routine: reads contents of named file into `*data`
Status ReadFileToString(Env* env, const string& fname, string* data);

/// A utility routine: write contents of `data` to file named `fname`
/// (overwriting existing contents, if any).
Status WriteStringToFile(Env* env, const string& fname,
                         const StringPiece& data);

/// Write binary representation of "proto" to the named file.
Status WriteBinaryProto(Env* env, const string& fname,
                        const ::tensorflow::protobuf::MessageLite& proto);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into `*proto`.
Status ReadBinaryProto(Env* env, const string& fname,
                       ::tensorflow::protobuf::MessageLite* proto);

/// Read contents of named file and parse as text encoded proto data
/// and store into `*proto`.
Status ReadTextProto(Env* env, const string& fname,
                     ::tensorflow::protobuf::Message* proto);

namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, const string& scheme) {
    env->RegisterFileSystem(scheme,
                            []() -> FileSystem* { return new Factory; });
  }
};

}  // namespace register_file_system

}  // namespace tensorflow

// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory)   \
  static ::tensorflow::register_file_system::Register<factory> \
      register_ff##ctr TF_ATTRIBUTE_UNUSED =                   \
          ::tensorflow::register_file_system::Register<factory>(env, scheme)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(Env::Default(), scheme, factory);

#endif  // TENSORFLOW_CORE_PLATFORM_ENV_H_
