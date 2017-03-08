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

#ifndef TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/core/platform/env.h"

namespace tensorflow {

// A file system that uses a graph saved in memmapped format by
// MemmappedEnvWriter as a file system.
//
// The format supports saved tensors and protos. Tensors are saved at aligned
// offsets.
//
// Format specification:
// - last 8 bytes of a package is encoded offset to the directory. The encoding
// is always little endian, independently from the platform, done by functions
// EncodeUint64LittleEndian/DecodeUint64LittleEndian
// - the directory starts from the encoded offset and is saved proto
// MemmappedFileSystemDirectory with names and offsets to the regions.
// - at the offsets in the directory the file regions are stored. Tensor regions
// are aligned such way that when the package mapped to RAM they have the right
// offset to be used by ImmutableConst operator.
//
// Region naming:
// Region naming is up to the application, all of them starts from
// kMemmappedPackagePrefix. The default graph usually has name
// kMemmappedPackageDefaultGraphDef; for more details see the conversion
// utility
// third_party/tensorflow/contrib/util/convert_graphdef_memmapped_format.cc
//
// A "frozen" GraphDef can be converted into this format using
// tensorflow/contrib/util/convert_graphdef_memmapped_format
class MemmappedFileSystem : public FileSystem {
 public:
  // Memmapped regions use this prefix to distinguish from
  // the filesystem.
  static constexpr char kMemmappedPackagePrefix[] = "memmapped_package://";
  // The default graphdef in the package.
  static constexpr char kMemmappedPackageDefaultGraphDef[] =
      "memmapped_package://.";

  MemmappedFileSystem();
  ~MemmappedFileSystem() override = default;
  Status FileExists(const string& fname) override;
  Status NewRandomAccessFile(
      const string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;
  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  // All these functions return Unimplemented error, the memmapped storage is
  // read only.
  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;
  Status GetChildren(const string& dir, std::vector<string>* r) override;
  Status DeleteFile(const string& f) override;
  Status CreateDir(const string& d) override;
  Status DeleteDir(const string& d) override;
  Status RenameFile(const string& s, const string& t) override;

  // These functions are implemented.
  Status GetFileSize(const string& f, uint64* s) override;
  // Currently just returns size.
  Status Stat(const string& fname, FileStatistics* stat) override;

  // Initializes filesystem from a file in memmapped format.
  Status InitializeFromFile(Env* env, const string& filename);

  // Checks if the filename has a correct prefix.
  static bool IsMemmappedPackageFilename(const string& filename);

  static bool IsWellFormedMemmappedPackageFilename(const string& filename);

 private:
  struct FileRegion {
    FileRegion(uint64 o, uint64 l) : offset(o), length(l) {}

    uint64 offset;  // Offset from the beginning of the file.
    uint64 length;  // Length of the region.
  };

  using DirectoryType = std::unordered_map<string, FileRegion>;

  const void* GetMemoryWithOffset(uint64 offset) const;

  std::unique_ptr<ReadOnlyMemoryRegion> mapped_memory_;
  DirectoryType directory_;

  TF_DISALLOW_COPY_AND_ASSIGN(MemmappedFileSystem);
};

class MemmappedEnv : public EnvWrapper {
 public:
  explicit MemmappedEnv(Env* env);
  ~MemmappedEnv() override = default;
  Status GetFileSystemForFile(const string& fname,
                              FileSystem** result) override;
  Status GetRegisteredFileSystemSchemes(std::vector<string>* schemes) override;
  Status InitializeFromFile(const string& filename);

 protected:
  std::unique_ptr<MemmappedFileSystem> memmapped_file_system_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_MEMMAPPED_FILE_SYSTEM_H_
