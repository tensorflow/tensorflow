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

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/error.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/file_system_helper.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/strcat.h"
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
  LibHDFS() { LoadAndBind(); }

  // The status, if any, from failure to load.
  Status status() const { return status_; }

  std::function<hdfsFS(hdfsBuilder*)> hdfsBuilderConnect;
  std::function<hdfsBuilder*()> hdfsNewBuilder;
  std::function<void(hdfsBuilder*, const char*)> hdfsBuilderSetNameNode;
  std::function<int(const char*, char**)> hdfsConfGetStr;
  std::function<int(hdfsFS, hdfsFile)> hdfsCloseFile;
  std::function<tSize(hdfsFS, hdfsFile, tOffset, void*, tSize)> hdfsPread;
  std::function<tSize(hdfsFS, hdfsFile, const void*, tSize)> hdfsWrite;
  std::function<int(hdfsFS, hdfsFile)> hdfsHFlush;
  std::function<int(hdfsFS, hdfsFile)> hdfsHSync;
  std::function<tOffset(hdfsFS, hdfsFile)> hdfsTell;
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
      TF_RETURN_IF_ERROR(Env::Default()->LoadDynamicLibrary(name, handle));
#define BIND_HDFS_FUNC(function) \
  TF_RETURN_IF_ERROR(BindFunc(*handle, #function, &function));

      BIND_HDFS_FUNC(hdfsBuilderConnect);
      BIND_HDFS_FUNC(hdfsNewBuilder);
      BIND_HDFS_FUNC(hdfsBuilderSetNameNode);
      BIND_HDFS_FUNC(hdfsConfGetStr);
      BIND_HDFS_FUNC(hdfsCloseFile);
      BIND_HDFS_FUNC(hdfsPread);
      BIND_HDFS_FUNC(hdfsWrite);
      BIND_HDFS_FUNC(hdfsHFlush);
      BIND_HDFS_FUNC(hdfsTell);
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
#if defined(PLATFORM_WINDOWS)
    const char* kLibHdfsDso = "hdfs.dll";
#elif defined(MACOS) || defined(TARGET_OS_MAC)
    const char* kLibHdfsDso = "libhdfs.dylib";
#else
    const char* kLibHdfsDso = "libhdfs.so";
#endif
    char* hdfs_home = getenv("HADOOP_HDFS_HOME");
    if (hdfs_home != nullptr) {
      string path = io::JoinPath(hdfs_home, "lib", "native", kLibHdfsDso);
      status_ = TryLoadAndBind(path.c_str(), &handle_);
      if (status_.ok()) {
        return;
      } else {
        LOG(ERROR) << "HadoopFileSystem load error: "
                   << status_.error_message();
      }
    }

    // Try to load the library dynamically in case it has been installed
    // to a in non-standard location.
    status_ = TryLoadAndBind(kLibHdfsDso, &handle_);
  }

  Status status_;
  void* handle_ = nullptr;
};

HadoopFileSystem::HadoopFileSystem() {}

HadoopFileSystem::~HadoopFileSystem() {}

const LibHDFS* libhdfs() {
  static const LibHDFS* libhdfs = new LibHDFS();
  return libhdfs;
}

Status SplitArchiveNameAndPath(StringPiece& path, string& nn) {
  size_t index_end_archive_name = path.find(".har");
  if (index_end_archive_name == path.npos) {
    return errors::InvalidArgument(
        "Hadoop archive path does not contain a .har extension");
  }
  // Case of hadoop archive. Namenode is the path to the archive.
  std::ostringstream namenodestream;
  namenodestream << "har://" << nn
                 << path.substr(0, index_end_archive_name + 4);
  nn = namenodestream.str();
  path.remove_prefix(index_end_archive_name + 4);
  if (path.empty()) {
    // Root of the archive
    path = "/";
  }
  return Status::OK();
}

// We implement connection caching in Tensorflow, which can significantly
// improve performance. Fixes #43187
Status HadoopFileSystem::Connect(StringPiece fname, hdfsFS* fs) {
  TF_RETURN_IF_ERROR(libhdfs()->status());

  StringPiece scheme, namenode, path;
  io::ParseURI(fname, &scheme, &namenode, &path);
  string nn(namenode);

  string cacheKey(scheme.data(), scheme.size());
  if (scheme == "file") {
    nn = "";
  } else if (scheme == "viewfs") {
    char* defaultFS = nullptr;
    libhdfs()->hdfsConfGetStr("fs.defaultFS", &defaultFS);
    StringPiece defaultScheme, defaultCluster, defaultPath;
    io::ParseURI(defaultFS, &defaultScheme, &defaultCluster, &defaultPath);

    if (scheme != defaultScheme ||
        (namenode != "" && namenode != defaultCluster)) {
      return errors::Unimplemented(
          "viewfs is only supported as a fs.defaultFS.");
    }
    // The default NameNode configuration will be used (from the XML
    // configuration files). See:
    // https://github.com/tensorflow/tensorflow/blob/v1.0.0/third_party/hadoop/hdfs.h#L259
    nn = "default";
  } else if (scheme == "har") {
    TF_RETURN_IF_ERROR(SplitArchiveNameAndPath(path, nn));
  } else {
    if (nn.empty()) {
      nn = "default";
    }
  }
  cacheKey += nn;
  {
    mutex_lock lock(mu_);
    if (connectionCache_.find(cacheKey) == connectionCache_.end()) {
      hdfsBuilder* builder = libhdfs()->hdfsNewBuilder();
      libhdfs()->hdfsBuilderSetNameNode(builder,
                                        nn.empty() ? nullptr : nn.c_str());
      hdfsFS cacheFs = libhdfs()->hdfsBuilderConnect(builder);
      if (cacheFs == nullptr) {
        return errors::Aborted(strerror(errno));
      }
      connectionCache_[cacheKey] = cacheFs;
    }
    *fs = connectionCache_[cacheKey];
  }
  return Status::OK();
}

string HadoopFileSystem::TranslateName(const string& name) const {
  StringPiece scheme, namenode, path;
  io::ParseURI(name, &scheme, &namenode, &path);
  return string(path);
}

class HDFSRandomAccessFile : public RandomAccessFile {
 public:
  HDFSRandomAccessFile(const string& filename, const string& hdfs_filename,
                       hdfsFS fs, hdfsFile file)
      : filename_(filename),
        hdfs_filename_(hdfs_filename),
        fs_(fs),
        file_(file) {
    const char* disable_eof_retried = getenv("HDFS_DISABLE_READ_EOF_RETRIED");
    if (disable_eof_retried && disable_eof_retried[0] == '1') {
      disable_eof_retried_ = true;
    } else {
      disable_eof_retried_ = false;
    }
  }

  ~HDFSRandomAccessFile() override {
    if (file_ != nullptr) {
      mutex_lock lock(mu_);
      libhdfs()->hdfsCloseFile(fs_, file_);
    }
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    bool eof_retried = false;
    if (disable_eof_retried_) {
      // eof_retried = true, avoid calling hdfsOpenFile in Read, Fixes #42597
      eof_retried = true;
    }
    while (n > 0 && s.ok()) {
      // We lock inside the loop rather than outside so we don't block other
      // concurrent readers.
      mutex_lock lock(mu_);
      // Max read length is INT_MAX-2, for hdfsPread function take a parameter
      // of int32. -2 offset can avoid JVM OutOfMemoryError.
      size_t read_n =
          std::min(n, static_cast<size_t>(std::numeric_limits<int>::max() - 2));
      tSize r = libhdfs()->hdfsPread(fs_, file_, static_cast<tOffset>(offset),
                                     dst, static_cast<tSize>(read_n));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (!eof_retried && r == 0) {
        // Always reopen the file upon reaching EOF to see if there's more data.
        // If writers are streaming contents while others are concurrently
        // reading, HDFS requires that we reopen the file to see updated
        // contents.
        //
        // Fixes #5438
        if (file_ != nullptr && libhdfs()->hdfsCloseFile(fs_, file_) != 0) {
          return IOError(filename_, errno);
        }
        file_ = libhdfs()->hdfsOpenFile(fs_, hdfs_filename_.c_str(), O_RDONLY,
                                        0, 0, 0);
        if (file_ == nullptr) {
          return IOError(filename_, errno);
        }
        eof_retried = true;
      } else if (eof_retried && r == 0) {
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
  string hdfs_filename_;
  hdfsFS fs_;
  bool disable_eof_retried_;

  mutable mutex mu_;
  mutable hdfsFile file_ TF_GUARDED_BY(mu_);
};

Status HadoopFileSystem::NewRandomAccessFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<RandomAccessFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file = libhdfs()->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                          O_RDONLY, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(
      new HDFSRandomAccessFile(fname, TranslateName(fname), fs, file));
  return Status::OK();
}

class HDFSWritableFile : public WritableFile {
 public:
  HDFSWritableFile(const string& fname, hdfsFS fs, hdfsFile file)
      : filename_(fname), fs_(fs), file_(file) {}

  ~HDFSWritableFile() override {
    if (file_ != nullptr) {
      Close().IgnoreError();
    }
  }

  Status Append(StringPiece data) override {
    size_t cur_pos = 0, write_len = 0;
    bool retry = false;
    // max() - 2 can avoid OutOfMemoryError in JVM .
    static const size_t max_len_once =
        static_cast<size_t>(std::numeric_limits<tSize>::max() - 2);
    while (cur_pos < data.size()) {
      write_len = std::min(data.size() - cur_pos, max_len_once);
      tSize w = libhdfs()->hdfsWrite(fs_, file_, data.data() + cur_pos,
                                     static_cast<tSize>(write_len));
      if (w == -1) {
        if (!retry && (errno == EINTR || errno == EAGAIN)) {
          retry = true;
        } else {
          return IOError(filename_, errno);
        }
      } else {
        cur_pos += w;
      }
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (libhdfs()->hdfsCloseFile(fs_, file_) != 0) {
      result = IOError(filename_, errno);
    }
    fs_ = nullptr;
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (libhdfs()->hdfsHFlush(fs_, file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Name(StringPiece* result) const override {
    *result = filename_;
    return Status::OK();
  }

  Status Sync() override {
    if (libhdfs()->hdfsHSync(fs_, file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Tell(int64* position) override {
    *position = libhdfs()->hdfsTell(fs_, file_);
    if (*position == -1) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

 private:
  string filename_;
  hdfsFS fs_;
  hdfsFile file_;
};

Status HadoopFileSystem::NewWritableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file = libhdfs()->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                          O_WRONLY, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(new HDFSWritableFile(fname, fs, file));
  return Status::OK();
}

Status HadoopFileSystem::NewAppendableFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<WritableFile>* result) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFile file = libhdfs()->hdfsOpenFile(fs, TranslateName(fname).c_str(),
                                          O_WRONLY | O_APPEND, 0, 0, 0);
  if (file == nullptr) {
    return IOError(fname, errno);
  }
  result->reset(new HDFSWritableFile(fname, fs, file));
  return Status::OK();
}

Status HadoopFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, TransactionToken* token,
    std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  // hadoopReadZero() technically supports this call with the following
  // caveats:
  // - It only works up to 2 GB. We'd have to Stat() the file to ensure that
  //   it fits.
  // - If not on the local filesystem, the entire file will be read, making
  //   it inefficient for callers that assume typical mmap() behavior.
  return errors::Unimplemented("HDFS does not support ReadOnlyMemoryRegion");
}

Status HadoopFileSystem::FileExists(const string& fname,
                                    TransactionToken* token) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));
  if (libhdfs()->hdfsExists(fs, TranslateName(fname).c_str()) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found.");
}

Status HadoopFileSystem::GetChildren(const string& dir, TransactionToken* token,
                                     std::vector<string>* result) {
  result->clear();
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(dir, token, &stat));

  int entries = 0;
  hdfsFileInfo* info =
      libhdfs()->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info == nullptr) {
    if (stat.is_directory) {
      // Assume it's an empty directory.
      return Status::OK();
    }
    return IOError(dir, errno);
  }
  for (int i = 0; i < entries; i++) {
    result->push_back(string(io::Basename(info[i].mName)));
  }
  libhdfs()->hdfsFreeFileInfo(info, entries);
  return Status::OK();
}

Status HadoopFileSystem::GetMatchingPaths(const string& pattern,
                                          TransactionToken* token,
                                          std::vector<string>* results) {
  return internal::GetMatchingPaths(this, Env::Default(), pattern, results);
}

Status HadoopFileSystem::DeleteFile(const string& fname,
                                    TransactionToken* token) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  if (libhdfs()->hdfsDelete(fs, TranslateName(fname).c_str(),
                            /*recursive=*/0) != 0) {
    return IOError(fname, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::CreateDir(const string& dir, TransactionToken* token) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  if (libhdfs()->hdfsCreateDirectory(fs, TranslateName(dir).c_str()) != 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::DeleteDir(const string& dir, TransactionToken* token) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(dir, &fs));

  // Count the number of entries in the directory, and only delete if it's
  // non-empty. This is consistent with the interface, but note that there's
  // a race condition where a file may be added after this check, in which
  // case the directory will still be deleted.
  int entries = 0;
  hdfsFileInfo* info =
      libhdfs()->hdfsListDirectory(fs, TranslateName(dir).c_str(), &entries);
  if (info != nullptr) {
    libhdfs()->hdfsFreeFileInfo(info, entries);
  }
  // Due to HDFS bug HDFS-8407, we can't distinguish between an error and empty
  // folder, especially for Kerberos enable setup, EAGAIN is quite common when
  // the call is actually successful. Check again by Stat.
  if (info == nullptr && errno != 0) {
    FileStatistics stat;
    TF_RETURN_IF_ERROR(Stat(dir, token, &stat));
  }

  if (entries > 0) {
    return errors::FailedPrecondition("Cannot delete a non-empty directory.");
  }
  if (libhdfs()->hdfsDelete(fs, TranslateName(dir).c_str(),
                            /*recursive=*/1) != 0) {
    return IOError(dir, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::GetFileSize(const string& fname,
                                     TransactionToken* token, uint64* size) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFileInfo* info =
      libhdfs()->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr) {
    return IOError(fname, errno);
  }
  *size = static_cast<uint64>(info->mSize);
  libhdfs()->hdfsFreeFileInfo(info, 1);
  return Status::OK();
}

Status HadoopFileSystem::RenameFile(const string& src, const string& target,
                                    TransactionToken* token) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(src, &fs));

  if (libhdfs()->hdfsExists(fs, TranslateName(target).c_str()) == 0 &&
      libhdfs()->hdfsDelete(fs, TranslateName(target).c_str(),
                            /*recursive=*/0) != 0) {
    return IOError(target, errno);
  }

  if (libhdfs()->hdfsRename(fs, TranslateName(src).c_str(),
                            TranslateName(target).c_str()) != 0) {
    return IOError(src, errno);
  }
  return Status::OK();
}

Status HadoopFileSystem::Stat(const string& fname, TransactionToken* token,
                              FileStatistics* stats) {
  hdfsFS fs = nullptr;
  TF_RETURN_IF_ERROR(Connect(fname, &fs));

  hdfsFileInfo* info =
      libhdfs()->hdfsGetPathInfo(fs, TranslateName(fname).c_str());
  if (info == nullptr) {
    return IOError(fname, errno);
  }
  stats->length = static_cast<int64>(info->mSize);
  stats->mtime_nsec = static_cast<int64>(info->mLastMod) * 1e9;
  stats->is_directory = info->mKind == kObjectKindDirectory;
  libhdfs()->hdfsFreeFileInfo(info, 1);
  return Status::OK();
}

REGISTER_LEGACY_FILE_SYSTEM("hdfs", HadoopFileSystem);
REGISTER_LEGACY_FILE_SYSTEM("viewfs", HadoopFileSystem);
REGISTER_LEGACY_FILE_SYSTEM("har", HadoopFileSystem);

}  // namespace tensorflow
