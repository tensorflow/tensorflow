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

#ifndef TENSORFLOW_CONTRIB_ANDROID_ASSET_MANAGER_FILESYSTEM_H_
#define TENSORFLOW_CONTRIB_ANDROID_ASSET_MANAGER_FILESYSTEM_H_

#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

#include "tensorflow/core/platform/file_system.h"

namespace tensorflow {

// FileSystem that uses Android's AAssetManager. Once initialized with a given
// AAssetManager, files in the given AAssetManager can be accessed through the
// prefix given when registered with the TensorFlow Env.
// Note that because APK assets are immutable, any operation that tries to
// modify the FileSystem will return tensorflow::error::code::UNIMPLEMENTED.
class AssetManagerFileSystem : public FileSystem {
 public:
  // Initialize an AssetManagerFileSystem. Note that this does not register the
  // file system with TensorFlow.
  // asset_manager - Non-null Android AAssetManager that backs this file
  //   system. The asset manager is not owned by this file system, and must
  //   outlive this class.
  // prefix - Common prefix to strip from all file URIs before passing them to
  //   the asset_manager. This is required because TensorFlow gives the entire
  //   file URI (file:///my_dir/my_file.txt) and AssetManager only knows paths
  //   relative to its base directory.
  AssetManagerFileSystem(AAssetManager* asset_manager, const string& prefix);
  ~AssetManagerFileSystem() override = default;

  Status FileExists(const string& fname) override;
  Status NewRandomAccessFile(
      const string& filename,
      std::unique_ptr<RandomAccessFile>* result) override;
  Status NewReadOnlyMemoryRegionFromFile(
      const string& filename,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override;

  Status GetFileSize(const string& f, uint64* s) override;
  // Currently just returns size.
  Status Stat(const string& fname, FileStatistics* stat) override;
  Status GetChildren(const string& dir, std::vector<string>* r) override;

  // All these functions return Unimplemented error. Asset storage is
  // read only.
  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override;
  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override;
  Status DeleteFile(const string& f) override;
  Status CreateDir(const string& d) override;
  Status DeleteDir(const string& d) override;
  Status RenameFile(const string& s, const string& t) override;

  Status GetMatchingPaths(const string& pattern,
                          std::vector<string>* results) override;

 private:
  string RemoveAssetPrefix(const string& name);

  // Return a string path that can be passed into AAssetManager functions.
  // For example, 'my_prefix://some/dir/' would return 'some/dir'.
  string NormalizeDirectoryPath(const string& fname);
  bool DirectoryExists(const std::string& fname);

  AAssetManager* asset_manager_;
  string prefix_;
};

}  // namespace tensorflow
#endif  // TENSORFLOW_CONTRIB_ANDROID_ASSET_MANAGER_FILESYSTEM_H_
