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

#ifndef TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_

#include <stdint.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class RandomAccessFile;
class ReadOnlyMemoryRegion;
class WritableFile;

/// A generic interface for accessing a file system.
class FileSystem {
 public:
  FileSystem() {}

  virtual ~FileSystem();

  /// The following functions are the implementations used by the corresponding
  /// functions in the Env class.
  virtual Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) = 0;

  virtual Status NewWritableFile(const string& fname,
                                 std::unique_ptr<WritableFile>* result) = 0;

  virtual Status NewAppendableFile(const string& fname,
                                   std::unique_ptr<WritableFile>* result) = 0;

  virtual Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;

  virtual bool FileExists(const string& fname) = 0;

  /// \brief Returns the immediate children in the given directory.
  ///
  /// The returned paths are relative to 'dir'.
  virtual Status GetChildren(const string& dir,
                             std::vector<string>* result) = 0;

  // \brief Given a pattern, stores in *results the set of paths that matches
  // that pattern. *results is cleared.
  //
  // pattern must match all of a name, not just a substring.
  //
  // pattern: { term }
  // term:
  //   '*': matches any sequence of non-'/' characters
  //   '?': matches a single non-'/' character
  //   '[' [ '^' ] { match-list } ']':
  //        matches any single character (not) on the list
  //   c: matches character c (c != '*', '?', '\\', '[')
  //   '\\' c: matches character c
  // character-range:
  //   c: matches character c (c != '\\', '-', ']')
  //   '\\' c: matches character c
  //   lo '-' hi: matches character c for lo <= c <= hi
  //
  // Typical return codes
  //  * OK - no errors
  //  * UNIMPLEMENTED - Some underlying functions (like GetChildren) are not
  //                    implemented
  // The default implementation uses a combination of GetChildren, MatchPath
  // and IsDirectory.
  virtual Status GetMatchingPaths(const string& pattern,
                                  std::vector<string>* results);

  virtual Status Stat(const string& fname, FileStatistics* stat) = 0;

  virtual Status DeleteFile(const string& fname) = 0;

  virtual Status CreateDir(const string& dirname) = 0;

  // \brief Creates the specified directory and all the necessary
  // subdirectories. Typical return codes.
  //  * OK - successfully created the directory and sub directories, even if
  //         they were already created.
  //  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual Status RecursivelyCreateDir(const string& dirname);

  virtual Status DeleteDir(const string& dirname) = 0;

  // \brief Deletes the specified directory and all subdirectories and files
  // underneath it. undeleted_files and undeleted_dirs stores the number of
  // files and directories that weren't deleted (unspecified if the return
  // status is not OK).
  // REQUIRES: undeleted_files, undeleted_dirs to be not null.
  // Typical return codes
  //  * OK - dirname exists and we were able to delete everything underneath.
  //  * NOT_FOUND - dirname doesn't exist
  //  * PERMISSION_DENIED - dirname or some descendant is not writable
  //  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  //                    implemented
  virtual Status DeleteRecursively(const string& dirname,
                                   int64* undeleted_files,
                                   int64* undeleted_dirs);

  virtual Status GetFileSize(const string& fname, uint64* file_size) = 0;

  // Overwrites the target if it exists.
  virtual Status RenameFile(const string& src, const string& target) = 0;

  // Translate an URI to a filename usable by the FileSystem implementation. The
  // implementation in this class cleans up the path, removing duplicate /'s,
  // resolving .. and . (more details in tensorflow::lib::io::CleanPath).
  virtual string TranslateName(const string& name) const;

  // Returns whether the given path is a directory or not.
  // Typical return codes (not guaranteed exhaustive):
  //  * OK - The path exists and is a directory.
  //  * FAILED_PRECONDITION - The path exists and is not a directory.
  //  * NOT_FOUND - The path entry does not exist.
  //  * PERMISSION_DENIED - Insufficient permissions.
  //  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual Status IsDirectory(const string& fname);
};

#ifndef SWIG
// Degenerate file system that provides no implementations.
class NullFileSystem : public FileSystem {
 public:
  NullFileSystem() {}

  ~NullFileSystem() override = default;

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    return errors::Unimplemented("NewRandomAccessFile unimplemented");
  }

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  bool FileExists(const string& fname) override { return false; }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return errors::Unimplemented("GetChildren unimplemented");
  }

  Status DeleteFile(const string& fname) override {
    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname) override {
    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname) override {
    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return errors::Unimplemented("GetFileSize unimplemented");
  }

  Status RenameFile(const string& src, const string& target) override {
    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return errors::Unimplemented("Stat unimplemented");
  }
};
#endif

/// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() {}
  virtual ~RandomAccessFile();

  /// \brief Reads up to `n` bytes from the file starting at `offset`.
  ///
  /// `scratch[0..n-1]` may be written by this routine.  Sets `*result`
  /// to the data that was read (including if fewer than `n` bytes were
  /// successfully read).  May set `*result` to point at data in
  /// `scratch[0..n-1]`, so `scratch[0..n-1]` must be live when
  /// `*result` is used.
  ///
  /// On OK returned status: `n` bytes have been stored in `*result`.
  /// On non-OK returned status: `[0..n]` bytes have been stored in `*result`.
  ///
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual Status Read(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomAccessFile);
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
  TF_DISALLOW_COPY_AND_ASSIGN(WritableFile);
};

/// \brief A readonly memmapped file abstraction.
///
/// The implementation must guarantee that all memory is accessable when the
/// object exists, independently from the Env that created it.
class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() {}
  virtual ~ReadOnlyMemoryRegion() = default;
  virtual const void* data() = 0;
  virtual uint64 length() = 0;
};

/// \brief A registry for file system implementations.
///
/// Filenames are specified as an URI, which is of the form
/// [scheme://]<filename>.
/// File system implementations are registered using the REGISTER_FILE_SYSTEM
/// macro, providing the 'scheme' as the key.
class FileSystemRegistry {
 public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry();
  virtual Status Register(const string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const string& scheme) = 0;
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<string>* schemes) = 0;
};

// Populates the scheme, host, and path from a URI.
//
// Corner cases:
// - If the URI is invalid, scheme and host are set to empty strings and the
//   passed string is assumed to be a path
// - If the URI omits the path (e.g. file://host), then the path is left empty.
void ParseURI(StringPiece uri, StringPiece* scheme, StringPiece* host,
              StringPiece* path);

// Creates a URI from a scheme, host, and path. If the scheme is empty, we just
// return the path.
string CreateURI(StringPiece scheme, StringPiece host, StringPiece path);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
