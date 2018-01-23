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

#include "tensorflow/contrib/android/asset_manager_filesystem.h"

#include <unistd.h>

#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {
namespace {

string RemoveSuffix(const string& name, const string& suffix) {
  string output(name);
  StringPiece piece(output);
  str_util::ConsumeSuffix(&piece, suffix);
  return piece.ToString();
}

// Closes the given AAsset when variable is destructed.
class ScopedAsset {
 public:
  ScopedAsset(AAsset* asset) : asset_(asset) {}
  ~ScopedAsset() {
    if (asset_ != nullptr) {
      AAsset_close(asset_);
    }
  }

  AAsset* get() const { return asset_; }

 private:
  AAsset* asset_;
};

// Closes the given AAssetDir when variable is destructed.
class ScopedAssetDir {
 public:
  ScopedAssetDir(AAssetDir* asset_dir) : asset_dir_(asset_dir) {}
  ~ScopedAssetDir() {
    if (asset_dir_ != nullptr) {
      AAssetDir_close(asset_dir_);
    }
  }

  AAssetDir* get() const { return asset_dir_; }

 private:
  AAssetDir* asset_dir_;
};

class ReadOnlyMemoryRegionFromAsset : public ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegionFromAsset(std::unique_ptr<char[]> data, uint64 length)
      : data_(std::move(data)), length_(length) {}
  ~ReadOnlyMemoryRegionFromAsset() override = default;

  const void* data() override { return reinterpret_cast<void*>(data_.get()); }
  uint64 length() override { return length_; }

 private:
  std::unique_ptr<char[]> data_;
  uint64 length_;
};

// Note that AAssets are not thread-safe and cannot be used across threads.
// However, AAssetManager is. Because RandomAccessFile must be thread-safe and
// used across threads, new AAssets must be created for every access.
// TODO(tylerrhodes): is there a more efficient way to do this?
class RandomAccessFileFromAsset : public RandomAccessFile {
 public:
  RandomAccessFileFromAsset(AAssetManager* asset_manager, const string& name)
      : asset_manager_(asset_manager), file_name_(name) {}
  ~RandomAccessFileFromAsset() override = default;

  Status Read(uint64 offset, size_t to_read, StringPiece* result,
              char* scratch) const override {
    auto asset = ScopedAsset(AAssetManager_open(
        asset_manager_, file_name_.c_str(), AASSET_MODE_RANDOM));
    if (asset.get() == nullptr) {
      return errors::NotFound("File ", file_name_, " not found.");
    }

    off64_t new_offset = AAsset_seek64(asset.get(), offset, SEEK_SET);
    off64_t length = AAsset_getLength64(asset.get());
    if (new_offset < 0) {
      *result = StringPiece(scratch, 0);
      return errors::OutOfRange("Read after file end.");
    }
    const off64_t region_left =
        std::min(length - new_offset, static_cast<off64_t>(to_read));
    int read = AAsset_read(asset.get(), scratch, region_left);
    if (read < 0) {
      return errors::Internal("Error reading from asset.");
    }
    *result = StringPiece(scratch, region_left);
    return (region_left == to_read)
               ? Status::OK()
               : errors::OutOfRange("Read less bytes than requested.");
  }

 private:
  AAssetManager* asset_manager_;
  string file_name_;
};

}  // namespace

AssetManagerFileSystem::AssetManagerFileSystem(AAssetManager* asset_manager,
                                               const string& prefix)
    : asset_manager_(asset_manager), prefix_(prefix) {}

Status AssetManagerFileSystem::FileExists(const string& fname) {
  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  return Status::OK();
}

Status AssetManagerFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  result->reset(new RandomAccessFileFromAsset(asset_manager_, path));
  return Status::OK();
}

Status AssetManagerFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_STREAMING));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }

  off64_t start, length;
  int fd = AAsset_openFileDescriptor64(asset.get(), &start, &length);
  std::unique_ptr<char[]> data;
  if (fd >= 0) {
    data.reset(new char[length]);
    ssize_t result = pread(fd, data.get(), length, start);
    if (result < 0) {
      return errors::Internal("Error reading from file ", fname,
                              " using 'read': ", result);
    }
    if (result != length) {
      return errors::Internal("Expected size does not match size read: ",
                              "Expected ", length, " vs. read ", result);
    }
    close(fd);
  } else {
    length = AAsset_getLength64(asset.get());
    data.reset(new char[length]);
    const void* asset_buffer = AAsset_getBuffer(asset.get());
    if (asset_buffer == nullptr) {
      return errors::Internal("Error reading ", fname, " from asset manager.");
    }
    memcpy(data.get(), asset_buffer, length);
  }
  result->reset(new ReadOnlyMemoryRegionFromAsset(std::move(data), length));
  return Status::OK();
}

Status AssetManagerFileSystem::GetChildren(const string& prefixed_dir,
                                           std::vector<string>* r) {
  std::string path = NormalizeDirectoryPath(prefixed_dir);
  auto dir =
      ScopedAssetDir(AAssetManager_openDir(asset_manager_, path.c_str()));
  if (dir.get() == nullptr) {
    return errors::NotFound("Directory ", prefixed_dir, " not found.");
  }
  const char* next_file = AAssetDir_getNextFileName(dir.get());
  while (next_file != nullptr) {
    r->push_back(next_file);
    next_file = AAssetDir_getNextFileName(dir.get());
  }
  return Status::OK();
}

Status AssetManagerFileSystem::GetFileSize(const string& fname, uint64* s) {
  // If fname corresponds to a directory, return early. It doesn't map to an
  // AAsset, and would otherwise return NotFound.
  if (DirectoryExists(fname)) {
    *s = 0;
    return Status::OK();
  }
  string path = RemoveAssetPrefix(fname);
  auto asset = ScopedAsset(
      AAssetManager_open(asset_manager_, path.c_str(), AASSET_MODE_RANDOM));
  if (asset.get() == nullptr) {
    return errors::NotFound("File ", fname, " not found.");
  }
  *s = AAsset_getLength64(asset.get());
  return Status::OK();
}

Status AssetManagerFileSystem::Stat(const string& fname, FileStatistics* stat) {
  uint64 size;
  stat->is_directory = DirectoryExists(fname);
  TF_RETURN_IF_ERROR(GetFileSize(fname, &size));
  stat->length = size;
  return Status::OK();
}

string AssetManagerFileSystem::NormalizeDirectoryPath(const string& fname) {
  return RemoveSuffix(RemoveAssetPrefix(fname), "/");
}

string AssetManagerFileSystem::RemoveAssetPrefix(const string& name) {
  string output(name);
  StringPiece piece(output);
  piece.Consume(prefix_);
  return piece.ToString();
}

bool AssetManagerFileSystem::DirectoryExists(const std::string& fname) {
  std::string path = NormalizeDirectoryPath(fname);
  auto dir =
      ScopedAssetDir(AAssetManager_openDir(asset_manager_, path.c_str()));
  // Note that openDir will return something even if the directory doesn't
  // exist. Therefore, we need to ensure one file exists in the folder.
  return AAssetDir_getNextFileName(dir.get()) != NULL;
}

Status AssetManagerFileSystem::NewWritableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::DeleteFile(const string& f) {
  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::CreateDir(const string& d) {
  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::DeleteDir(const string& d) {
  return errors::Unimplemented("Asset storage is read only.");
}
Status AssetManagerFileSystem::RenameFile(const string& s, const string& t) {
  return errors::Unimplemented("Asset storage is read only.");
}

}  // namespace tensorflow
