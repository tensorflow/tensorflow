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

#ifndef TENSORFLOW_TSL_PLATFORM_ENV_H_
#define TENSORFLOW_TSL_PLATFORM_ENV_H_

#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "absl/functional/any_invocable.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/file_system.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mutex.h"
#include "tsl/platform/numa.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/stringpiece.h"
#include "tsl/platform/types.h"

// Delete leaked Windows definitions.
#ifdef PLATFORM_WINDOWS
#undef CopyFile
#undef DeleteFile
#endif

namespace tsl {

class Thread;
struct ThreadOptions;

/// \brief An interface used by the tensorflow implementation to
/// access operating system functionality like the filesystem etc.
///
/// Callers may wish to provide a custom Env object to get fine grain
/// control.
///
/// All Env implementations of file-system modifying functionality are safe
/// for concurrent access from multiple threads without any external
/// synchronization, *however*, Envs and their underlying file systems are
/// global objects, and therefore, if any thread modifies options, the modified
/// options take effect process-wide. The SetOption functions themselves are
/// also *not* thread safe.
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
  virtual Status GetFileSystemForFile(const std::string& fname,
                                      FileSystem** result);

  /// \brief Returns the file system schemes registered for this Env.
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<std::string>* schemes);

  /// \brief Register a file system for a scheme.
  virtual Status RegisterFileSystem(const std::string& scheme,
                                    FileSystemRegistry::Factory factory);

  /// \brief Register a modular file system for a scheme.
  ///
  /// Same as `RegisterFileSystem` but for filesystems provided by plugins.
  ///
  /// TODO(b/139060984): After all filesystems are converted, make this be the
  /// canonical registration function.
  virtual Status RegisterFileSystem(const std::string& scheme,
                                    std::unique_ptr<FileSystem> filesystem);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::string& value);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<string>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<int64_t>& values);

  Status SetOption(const std::string& scheme, const std::string& key,
                   const std::vector<double>& values);

  /// \brief Flush filesystem caches for all registered filesystems.
  Status FlushFileSystemCaches();

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
  Status NewRandomAccessFile(const std::string& fname,
                             std::unique_ptr<RandomAccessFile>* result);

  Status NewRandomAccessFile(const std::string& fname, TransactionToken* token,
                             std::unique_ptr<RandomAccessFile>* result) {
    // We duplicate these methods due to Google internal coding style prevents
    // virtual functions with default arguments. See PR #41615.
    return OkStatus();
  }

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
  Status NewWritableFile(const std::string& fname,
                         std::unique_ptr<WritableFile>* result);

  Status NewWritableFile(const std::string& fname, TransactionToken* token,
                         std::unique_ptr<WritableFile>* result) {
    return OkStatus();
  }

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
  Status NewAppendableFile(const std::string& fname,
                           std::unique_ptr<WritableFile>* result);

  Status NewAppendableFile(const std::string& fname, TransactionToken* token,
                           std::unique_ptr<WritableFile>* result) {
    return OkStatus();
  }
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
      const std::string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result);

  Status NewReadOnlyMemoryRegionFromFile(
      const std::string& fname, TransactionToken* token,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) {
    return OkStatus();
  }

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  Status FileExists(const std::string& fname);

  Status FileExists(const std::string& fname, TransactionToken* token) {
    return OkStatus();
  }

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  bool FilesExist(const std::vector<string>& files,
                  std::vector<Status>* status);

  bool FilesExist(const std::vector<string>& files, TransactionToken* token,
                  std::vector<Status>* status) {
    return true;
  }

  /// \brief Stores in *result the names of the children of the specified
  /// directory. The names are relative to "dir".
  ///
  /// Original contents of *results are dropped.
  Status GetChildren(const std::string& dir, std::vector<string>* result);

  Status GetChildren(const std::string& dir, TransactionToken* token,
                     std::vector<string>* result) {
    return OkStatus();
  }

  /// \brief Returns true if the path matches the given pattern. The wildcards
  /// allowed in pattern are described in FileSystem::GetMatchingPaths.
  virtual bool MatchPath(const std::string& path,
                         const std::string& pattern) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// More details about `pattern` in FileSystem::GetMatchingPaths.
  virtual Status GetMatchingPaths(const std::string& pattern,
                                  std::vector<string>* results);

  Status GetMatchingPaths(const std::string& pattern, TransactionToken* token,
                          std::vector<string>* results) {
    return OkStatus();
  }

  /// Deletes the named file.
  Status DeleteFile(const std::string& fname);

  Status DeleteFile(const std::string& fname, TransactionToken* token) {
    return OkStatus();
  }

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. This is accomplished by traversing the directory tree
  /// rooted at dirname and deleting entries as they are encountered.
  ///
  /// If dirname itself is not readable or does not exist, *undeleted_dir_count
  /// is set to 1, *undeleted_file_count is set to 0 and an appropriate status
  /// (e.g. NOT_FOUND) is returned.
  ///
  /// If dirname and all its descendants were successfully deleted, TF_OK is
  /// returned and both error counters are set to zero.
  ///
  /// Otherwise, while traversing the tree, undeleted_file_count and
  /// undeleted_dir_count are updated if an entry of the corresponding type
  /// could not be deleted. The returned error status represents the reason that
  /// any one of these entries could not be deleted.
  ///
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  ///
  /// Typical return codes:
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  Status DeleteRecursively(const std::string& dirname, int64_t* undeleted_files,
                           int64_t* undeleted_dirs);

  Status DeleteRecursively(const std::string& dirname, TransactionToken* token,
                           int64_t* undeleted_files, int64_t* undeleted_dirs) {
    return OkStatus();
  }

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories. Typical return codes.
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  Status RecursivelyCreateDir(const std::string& dirname);

  Status RecursivelyCreateDir(const std::string& dirname,
                              TransactionToken* token) {
    return OkStatus();
  }
  /// \brief Creates the specified directory. Typical return codes
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  Status CreateDir(const std::string& dirname);

  Status CreateDir(const std::string& dirname, TransactionToken* token) {
    return OkStatus();
  }

  /// Deletes the specified directory.
  Status DeleteDir(const std::string& dirname);

  Status DeleteDir(const std::string& dirname, TransactionToken* token) {
    return OkStatus();
  }

  /// Obtains statistics for the given path.
  Status Stat(const std::string& fname, FileStatistics* stat);

  Status Stat(const std::string& fname, TransactionToken* token,
              FileStatistics* stat) {
    return OkStatus();
  }

  /// \brief Returns whether the given path is a directory or not.
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  Status IsDirectory(const std::string& fname);

  /// \brief Returns whether the given path is on a file system
  /// that has atomic move capabilities. This can be used
  /// to determine if there needs to be a temp location to safely write objects.
  /// The second boolean argument has_atomic_move contains this information.
  ///
  /// Returns one of the following status codes (not guaranteed exhaustive):
  ///  * OK - The path is on a recognized file system,
  ///         so has_atomic_move holds the above information.
  ///  * UNIMPLEMENTED - The file system of the path hasn't been implemented in
  ///  TF
  Status HasAtomicMove(const std::string& path, bool* has_atomic_move);

  /// Stores the size of `fname` in `*file_size`.
  Status GetFileSize(const std::string& fname, uint64* file_size);

  Status GetFileSize(const std::string& fname, TransactionToken* token,
                     uint64* file_size) {
    return OkStatus();
  }

  /// \brief Renames file src to target. If target already exists, it will be
  /// replaced.
  Status RenameFile(const std::string& src, const std::string& target);

  Status RenameFile(const std::string& src, const std::string& target,
                    TransactionToken* token) {
    return OkStatus();
  }

  /// \brief Copy the src to target.
  Status CopyFile(const std::string& src, const std::string& target);

  Status CopyFile(const std::string& src, const std::string& target,
                  TransactionToken* token) {
    return OkStatus();
  }

  /// \brief starts a new transaction on the filesystem that handles filename
  Status StartTransaction(const std::string& filename,
                          TransactionToken** token) {
    *token = nullptr;
    return OkStatus();
  }

  /// \brief Adds `path` to transaction in `token` if token belongs to
  /// filesystem that handles the path.
  Status AddToTransaction(const std::string& path, TransactionToken* token) {
    return OkStatus();
  }

  /// \brief Get token for `path` or start a new transaction and add `path` to
  /// it.
  Status GetTokenOrStartTransaction(const std::string& path,
                                    TransactionToken** token) {
    *token = nullptr;
    return OkStatus();
  }

  /// \brief Returns the transaction for `path` or nullptr in `token`
  Status GetTransactionForPath(const std::string& path,
                               TransactionToken** token) {
    *token = nullptr;
    return OkStatus();
  }

  /// \brief Finalizes the transaction
  Status EndTransaction(TransactionToken* token) { return OkStatus(); }

  /// \brief Returns the absolute path of the current executable. It resolves
  /// symlinks if there is any.
  std::string GetExecutablePath();

  /// Creates a local unique temporary file name. Returns true if success.
  bool LocalTempFilename(std::string* filename);

  /// Creates a local unique file name that starts with |prefix| and ends with
  /// |suffix|. Returns true if success.
  bool CreateUniqueFileName(std::string* prefix, const std::string& suffix);

  /// \brief Return the runfiles directory if running under bazel. Returns
  /// the directory the executable is located in if not running under bazel.
  virtual std::string GetRunfilesDir() = 0;

  // TODO(jeff,sanjay): Add back thread/thread-pool support if needed.
  // TODO(jeff,sanjay): if needed, tighten spec so relative to epoch, or
  // provide a routine to get the absolute time.

  /// \brief Returns the number of nano-seconds since the Unix epoch.
  virtual uint64 NowNanos() const { return EnvTime::NowNanos(); }

  /// \brief Returns the number of micro-seconds since the Unix epoch.
  virtual uint64 NowMicros() const { return EnvTime::NowMicros(); }

  /// \brief Returns the number of seconds since the Unix epoch.
  virtual uint64 NowSeconds() const { return EnvTime::NowSeconds(); }

  /// Sleeps/delays the thread for the prescribed number of micro-seconds.
  virtual void SleepForMicroseconds(int64_t micros) = 0;

  /// Returns the process ID of the calling process.
  int32 GetProcessId();

  /// \brief Returns a new thread that is running fn() and is identified
  /// (for debugging/performance-analysis) by "name".
  ///
  /// Caller takes ownership of the result and must delete it eventually
  /// (the deletion will block until fn() stops running).
  virtual Thread* StartThread(
      const ThreadOptions& thread_options, const std::string& name,
      absl::AnyInvocable<void()> fn) TF_MUST_USE_RESULT = 0;

  // Returns the thread id of calling thread.
  // Posix: Returns pthread id which is only guaranteed to be unique within a
  //        process.
  // Windows: Returns thread id which is unique.
  virtual int32 GetCurrentThreadId() = 0;

  // Copies current thread name to "name". Returns true if success.
  virtual bool GetCurrentThreadName(std::string* name) = 0;

  // \brief Schedules the given closure on a thread-pool.
  //
  // NOTE(mrry): This closure may block.
  virtual void SchedClosure(absl::AnyInvocable<void()> closure) = 0;

  // \brief Schedules the given closure on a thread-pool after the given number
  // of microseconds.
  //
  // NOTE(mrry): This closure must not block.
  virtual void SchedClosureAfter(int64_t micros,
                                 absl::AnyInvocable<void()> closure) = 0;

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
  virtual Status LoadDynamicLibrary(const char* library_filename,
                                    void** handle) = 0;

  // \brief Get a pointer to a symbol from a dynamic library.
  //
  // "handle" should be a pointer returned from a previous call to LoadLibrary.
  // On success, store a pointer to the located symbol in "*symbol" and return
  // OK from the function. Otherwise, returns nullptr in "*symbol" and an error
  // status from the function.
  virtual Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                                      void** symbol) = 0;

  // \brief build the name of dynamic library.
  //
  // "name" should be name of the library.
  // "version" should be the version of the library or NULL
  // returns the name that LoadLibrary() can use
  virtual std::string FormatLibraryFileName(const std::string& name,
                                            const std::string& version) = 0;

  // Returns a possible list of local temporary directories.
  virtual void GetLocalTempDirectories(std::vector<string>* list) = 0;

 private:
  std::unique_ptr<FileSystemRegistry> file_system_registry_;
  Env(const Env&) = delete;
  void operator=(const Env&) = delete;
};

/// \brief An implementation of Env that forwards all calls to another Env.
///
/// May be useful to clients who wish to override just part of the
/// functionality of another Env.
class EnvWrapper : public Env {
 public:
  /// Initializes an EnvWrapper that delegates all calls to *t
  explicit EnvWrapper(Env* t) : target_(t) {}
  ~EnvWrapper() override;

  /// Returns the target to which this Env forwards all calls
  Env* target() const { return target_; }

  Status GetFileSystemForFile(const std::string& fname,
                              FileSystem** result) override {
    return target_->GetFileSystemForFile(fname, result);
  }

  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override {
    return target_->GetRegisteredFileSystemSchemes(schemes);
  }

  Status RegisterFileSystem(const std::string& scheme,
                            FileSystemRegistry::Factory factory) override {
    return target_->RegisterFileSystem(scheme, factory);
  }

  bool MatchPath(const std::string& path, const std::string& pattern) override {
    return target_->MatchPath(path, pattern);
  }

  uint64 NowMicros() const override { return target_->NowMicros(); }
  void SleepForMicroseconds(int64_t micros) override {
    target_->SleepForMicroseconds(micros);
  }
  Thread* StartThread(const ThreadOptions& thread_options,
                      const std::string& name,
                      absl::AnyInvocable<void()> fn) override {
    return target_->StartThread(thread_options, name, std::move(fn));
  }
  int32 GetCurrentThreadId() override { return target_->GetCurrentThreadId(); }
  bool GetCurrentThreadName(std::string* name) override {
    return target_->GetCurrentThreadName(name);
  }
  void SchedClosure(absl::AnyInvocable<void()> closure) override {
    target_->SchedClosure(std::move(closure));
  }
  void SchedClosureAfter(int64_t micros,
                         absl::AnyInvocable<void()> closure) override {
    target_->SchedClosureAfter(micros, std::move(closure));
  }
  Status LoadDynamicLibrary(const char* library_filename,
                            void** handle) override {
    return target_->LoadDynamicLibrary(library_filename, handle);
  }
  Status GetSymbolFromLibrary(void* handle, const char* symbol_name,
                              void** symbol) override {
    return target_->GetSymbolFromLibrary(handle, symbol_name, symbol);
  }
  std::string FormatLibraryFileName(const std::string& name,
                                    const std::string& version) override {
    return target_->FormatLibraryFileName(name, version);
  }

  std::string GetRunfilesDir() override { return target_->GetRunfilesDir(); }

 private:
  void GetLocalTempDirectories(std::vector<string>* list) override {
    target_->GetLocalTempDirectories(list);
  }

  Env* target_;
};

/// Represents a thread used to run a TSL function.
class Thread {
 public:
  Thread() {}

  /// Blocks until the thread of control stops running.
  virtual ~Thread();

 private:
  Thread(const Thread&) = delete;
  void operator=(const Thread&) = delete;
};

/// \brief Cross-platform setenv.
///
/// Since setenv() is not available on windows, we provide an
/// alternative with platform specific implementations here.
int setenv(const char* name, const char* value, int overwrite);

/// Cross-platform unsetenv.
int unsetenv(const char* name);

/// \brief Options to configure a Thread.
///
/// Note that the options are all hints, and the
/// underlying implementation may choose to ignore it.
struct ThreadOptions {
  /// Thread stack size to use (in bytes).
  size_t stack_size = 0;  // 0: use system default value
  /// Guard area size to use near thread stacks to use (in bytes)
  size_t guard_size = 0;  // 0: use system default value
  int numa_node = port::kNUMANoAffinity;
};

/// A utility routine: copy contents of `src` in file system `src_fs`
/// to `target` in file system `target_fs`.
Status FileSystemCopyFile(FileSystem* src_fs, const std::string& src,
                          FileSystem* target_fs, const std::string& target);

/// A utility routine: reads contents of named file into `*data`
Status ReadFileToString(Env* env, const std::string& fname, std::string* data);

/// A utility routine: write contents of `data` to file named `fname`
/// (overwriting existing contents, if any).
Status WriteStringToFile(Env* env, const std::string& fname,
                         const StringPiece& data);

/// Write binary representation of "proto" to the named file.
Status WriteBinaryProto(Env* env, const std::string& fname,
                        const protobuf::MessageLite& proto);

/// Reads contents of named file and parse as binary encoded proto data
/// and store into `*proto`.
Status ReadBinaryProto(Env* env, const std::string& fname,
                       protobuf::MessageLite* proto);

/// Write the text representation of "proto" to the named file.
inline Status WriteTextProto(Env* /* env */, const std::string& /* fname */,
                             const protobuf::MessageLite& /* proto */) {
  return errors::Unimplemented("Can't write text protos with protolite.");
}
Status WriteTextProto(Env* env, const std::string& fname,
                      const protobuf::Message& proto);

/// Read contents of named file and parse as text encoded proto data
/// and store into `*proto`.
inline Status ReadTextProto(Env* /* env */, const std::string& /* fname */,
                            protobuf::MessageLite* /* proto */) {
  return errors::Unimplemented("Can't parse text protos with protolite.");
}
Status ReadTextProto(Env* env, const std::string& fname,
                     protobuf::Message* proto);

/// Read contents of named file and parse as either text or binary encoded proto
/// data and store into `*proto`.
Status ReadTextOrBinaryProto(Env* env, const std::string& fname,
                             protobuf::Message* proto);
Status ReadTextOrBinaryProto(Env* env, const std::string& fname,
                             protobuf::MessageLite* proto);

// START_SKIP_DOXYGEN

// The following approach to register filesystems is deprecated and will be
// replaced with modular filesystem plugins registration.
// TODO(b/139060984): After all filesystems are converted, remove this.
namespace register_file_system {

template <typename Factory>
struct Register {
  Register(Env* env, const std::string& scheme, bool try_modular_filesystems) {
    // TODO(yongtang): Remove legacy file system registration for hdfs/s3/gcs
    // after TF 2.6+.
    if (try_modular_filesystems) {
      const char* env_value = getenv("TF_USE_MODULAR_FILESYSTEM");
      string load_plugin = env_value ? absl::AsciiStrToLower(env_value) : "";
      if (load_plugin == "true" || load_plugin == "1") {
        // We don't register the static filesystem and wait for SIG IO one
        LOG(WARNING) << "Using modular file system for '" << scheme << "'."
                     << " Please switch to tensorflow-io"
                     << " (https://github.com/tensorflow/io) for file system"
                     << " support of '" << scheme << "'.";
        return;
      }
      // If the envvar is missing or not "true"/"1", then fall back to legacy
      // implementation to be backwards compatible.
    }
    // TODO(b/32704451): Don't just ignore the ::tensorflow::Status object!
    env->RegisterFileSystem(scheme, []() -> FileSystem* { return new Factory; })
        .IgnoreError();
  }
};

}  // namespace register_file_system

// END_SKIP_DOXYGEN

}  // namespace tsl

// Register a FileSystem implementation for a scheme. Files with names that have
// "scheme://" prefixes are routed to use this implementation.
#define REGISTER_FILE_SYSTEM_ENV(env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ_HELPER(__COUNTER__, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ_HELPER(ctr, env, scheme, factory, modular) \
  REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)
#define REGISTER_FILE_SYSTEM_UNIQ(ctr, env, scheme, factory, modular)    \
  static ::tsl::register_file_system::Register<factory> register_ff##ctr \
      TF_ATTRIBUTE_UNUSED =                                              \
          ::tsl::register_file_system::Register<factory>(env, scheme, modular)

#define REGISTER_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::tsl::Env::Default(), scheme, factory, false);

#define REGISTER_LEGACY_FILE_SYSTEM(scheme, factory) \
  REGISTER_FILE_SYSTEM_ENV(::tsl::Env::Default(), scheme, factory, true);

#endif  // TENSORFLOW_TSL_PLATFORM_ENV_H_
