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

#include "tensorflow/core/platform/hadoop/hadoop_file_system.h"

#include <errno.h>

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/error.h"
#include "third_party/hadoop/hdfs.h"

namespace tensorflow {

template <typename R, typename... Args>
Status BindFunc(void* handle, const char* name,
                std::function<R(Args...)>* func) {
  void* symbol_ptr = nullptr;
  TF_RETURN_IF_ERROR(
      Env::Default()->GetSymbolFromLibrary(handle, name, &symbol_ptr));
  *func = reinterpret_cast<R (*)(Args...)>(symbol_ptr);
  return Status::OK();
}

class LibHDFS {
 public:
  static LibHDFS* Load() {
    static LibHDFS* lib = []() -> LibHDFS* {
      LibHDFS* lib = new LibHDFS;
      lib->LoadAndBind();
      return lib;
    }();

    return lib;
  }

  // The status, if any, from failure to load.
  Status status() { return status_; }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
  std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite;
  std::function<int(hdfsFS, hdfsFile)> hdfsFlush;
  std::function<int(hdfsFS, hdfsFile)> hdfsHSync;
  std::function<hdfsFile(hdfsFS, const char*, int, int, short, tSize)>
      hdfsOpenFile;
  std::function<int(hdfsFS, const char*)> hdfsExists;
  std::function<hdfsFileInfo*(hdfsFS, const char*, int*)> hdfsListDirectory;
  std::function<void(hdfsFileInfo*, int)> hdfsFreeFileInfo;
  std::function<int(hdfsFS, const char*, int recursive)> hdfsDelete;
  std::function<int(hdfsFS, const char*)> hdfsCreateDirectory;
  std::function<hdfsFileInfo*(hdfsFS, const char*)> hdfsGetPathInfo;
  std::function<int(hdfsFS, const char*, const char*)> hdfsRename;

 private:
  void LoadAndBind() {
    auto TryLoadAndBind = [this](const char* name, void** handle) -> Status {
      TF_RETURN_IF_ERROR(Env::Default()->LoadLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
  TF_RETURN_IF_ERROR(BindFunc(*handle, #function, &function));

      BIND_HDFS_FUNC(hdfsBuilderConnect);
      BIND_HDFS_FUNC(hdfsNewBuilder);
      BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
      BIND_HDFS_FUNC(hdfsCloseFile);
      BIND_HDFS_FUNC(hdfsPread);
      BIND_HDFS_FUNC(hdfsWrite);
      BIND_HDFS_FUNC(hdfsFlush);
      BIND_HDFS_FUNC(hdfsHSync);
      BIND_HDFS_FUNC(hdfsOpenFile);
      BIND_HDFS_FUNC(hdfsExists);
      BIND_HDFS_FUNC(hdfsListDirectory);
      BIND_HDFS_FUNC(hdfsFreeFileInfo);
      BIND_HDFS_FUNC(hdfsDelete);
      BIND_HDFS_FUNC(hdfsCreateDirectory);
      BIND_HDFS_FUNC(hdfsGetPathInfo);
      BIND_HDFS_FUNC(hdfsRename);
#undef BIND_HDFS_FUNC
      return Status::OK();
    };

    // libhdfs.so won't be in the standard locations. Use the path as specified
    // in the libhdfs documentation.
    char* hdfs_home = getenv("HADOOP_HDFS_HOME");
    if (hdfs_home == nullptr) {
      status_ = errors::FailedPrecondition(
          "Environment variable HADOOP_HDFS_HOME not set");
      return;
    }
    string path = io::JoinPath(hdfs_home, "lib", "native", "libhdfs.so");
    status_ = TryLoadAndBind(path.c_str(), &handle_);
    return;
  }

  Status status_;
  void* handle_ = nullptr;
};

HadoopFileSystem::HadoopFileSystem() : hdfs_(LibHDFS::Load()) {}

HadoopFileSystem::~HadoopFileSystem() {}

// We rely on HDFS connection caching here. The HDFS client calls
// org.apache.hadoop.fs.FileSystem.get(), which caches the connection
// internally.
Status HadoopFileSystem::Connect(StringPiece fname, hdfsFS* fs) {
  TF_RETURN_IF_ERROR(hdfs_->status());

  StringPiece scheme, namenode, path;
  ParseURI(fname, &scheme, &namenode, &path);
  const string nn = namenode.ToString();

  hdfsBuilder* builder = hdfs_->hdfsNewBuilder();
  if (scheme == "file") {
    hdfs_->hdfsBuilderSetNameNode(builder, nullptr);
  } else {
    hdfs_->hdfsBuilderSetNameNode(builder, nn.c_str());
  }
  *fs = hdfs_->hdfsBuilderConnect(builder);
  if (*fs == nullptr) {
    return errors::NotFound(strerror(errno));
  }
  return Status::OK();
}

string HadoopFileSystem::TranslateName(const string& name) const {
  StringPiece scheme, namenode, path;
  ParseURI(name, &scheme, &namenode, &path);
  return path.ToString();
}

class HDFSRandomAccessFile : public RandomAccessFile {
 public:
  HDFSRandomAccessFile(const string& fname, LibHDFS* hdfs, hdfsFS fs,
                       hdfsFile file)
      : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

  ~HDFSRandomAccessFile() override { hdfs_->hdfsCloseFile(fs_, file_); }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      tSize r = hdfs_->hdfsPread(fs_, file_, static_cast<tOffset>(offset), dst,
                                 static_cast<tSize>(n));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // hdfsPread may return EINTR too. Just retry.
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }

 private:
  string filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;
  hdfsFile file_;
};

Status HadoopFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_RDONLY, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(new HDFSRandomAccessFile(fname, hdfs_, fs, file));
  return Status::OK();
}

class HDFSWritableFile : public WritableFile {
 public:
  HDFSWritableFile(const string& fname, LibHDFS* hdfs, hdfsFS fs, hdfsFile file)
      : filename_(fname), hdfs_(hdfs), fs_(fs), file_(file) {}

  ~HDFSWritableFile() override {
    if (file_ != nullptr) {
      Close();
    }
  }

  Status Append(const StringPiece& data) override {
    if (hdfs_->hdfsWrite(fs_, file_, data.data(),
                         static_cast<tSize>(data.size())) == -1) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (hdfs_->hdfsCloseFile(fs_, file_) != 0) {
      result = IOError(filename_, errno);
    }
    hdfs_ = nullptr;
    fs_ = nullptr;
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (hdfs_->hdfsFlush(fs_, file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Sync() override {
    if (hdfs_->hdfsHSync(fs_, file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

 private:
  string filename_;
  LibHDFS* hdfs_;
  hdfsFS fs_;
  hdfsFile file_;
};

Status HadoopFileSystem::NewWritableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file =
      hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(), O_WRONLY, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  return Status::OK();
}

Status HadoopFileSystem::NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file = hdfs_->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                      O_WRONLY | O_APPEND, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(new HDFSWritableFile(fname, hdfs_, fs, file));
  return Status::OK();
}

Status HadoopFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  // hadoopReadZero() technically supports this call with the following
  // caveats:
  // - It only works up to 2 GB. We'd have to Stat() the file to ensure that
  //   it fits.
  // - If not on the local filesystem, the entire file will be read, making
  //   it inefficient for callers that assume typical mmap() behavior.
  return errors::Unimplemented("HDFS does not support ReadOnlyMemoryRegion");
}

bool HadoopFileSystem::FileExists(const string& fname) {
  hdfsFS fs = nullptr;
  Status status = Connect(fname, &fs);
  if (!status.ok()) {
    LOG(ERROR) << "Connect failed: " << status.error_message();
    return false;
  }

  return hdfs_->hdfsExists(fs, TranslateName(fname).c_str()) == 0;
}

Status HadoopFileSystem::GetChildren(const string& dir,
                                     std::vector<string>* result) {
  result->clear();
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(dir, &stat));

  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info == nullptr) {
    if (stat.is_directory) {
      // Assume it's an empty directory.
      return Status::OK();
    }
    return IOError(dir, errno);
  }
  for (int i = 0; i < entries; i++) {
    result->push_back(io::Basename(info[i].mName).ToString());
  }
  hdfs_->hdfsFreeFileInfo(info, entries);
  return Status::OK();
}

Status HadoopFileSystem::DeleteFile(const string& fname) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  if (hdfs_->hdfsDelete(fs, TranslateName(fname).c_str(),
                        /*recursive=*/0) != 0) {
    return IOError(fname, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::CreateDir(const string& dir) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  if (hdfs_->hdfsCreateDirectory(fs, TranslateName(dir).c_str()) != 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::DeleteDir(const string& dir) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  // Count the number of entries in the directory, and only delete if it's
  // non-empty. This is consistent with the interface, but note that there's
  // a race condition where a file may be added after this check, in which
  // case the directory will still be deleted.
  int entries = 0;
  hdfsFileInfo* info =
      hdfs_->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info != nullptr) {
    return IOError(dir, errno);
  }
  hdfs_->hdfsFreeFileInfo(info, entries);

  if (entries > 0) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }
  if (hdfs_->hdfsDelete(fs, TranslateName(dir).c_str(),
                        /*recursive=*/1) != 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::GetFileSize(const string& fname, uint64* size) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr) {
    return IOError(fname, errno);
  }
  *size = static_cast<uint64>(info->mSize);
  hdfs_->hdfsFreeFileInfo(info, 1);
  return Status::OK();
}

Status HadoopFileSystem::RenameFile(const string& src, const string& target) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(src, &fs));

  if (hdfs_->hdfsExists(fs, TranslateName(target).c_str()) == 0 &&
      hdfs_->hdfsDelete(fs, TranslateName(target).c_str(),
                        /*recursive=*/0) != 0) {
    return IOError(target, errno);
  }

  if (hdfs_->hdfsRename(fs, TranslateName(src).c_str(),
                        TranslateName(target).c_str()) != 0) {
    return IOError(src, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::Stat(const string& fname, FileStatistics* stats) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFileInfo* info = hdfs_->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr) {
    return IOError(fname, errno);
  }
  stats->length = static_cast<int64>(info->mSize);
  stats->mtime_nsec = static_cast<int64>(info->mLastMod) * 1e9;
  stats->is_directory = info->mKind == kObjectKindDirectory;
  hdfs_->hdfsFreeFileInfo(info, 1);
  return Status::OK();
}

REGISTER_FILE_SYSTEM("hdfs", HadoopFileSystem);

}  // namespace tensorflow
