/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/hadoop/hadoop_filesystem.h"

#include <stdlib.h>
#include <string.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>

#include "absl/synchronization/mutex.h"
#include "tensorflow/c/env.h"
#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_status.h"
#include "third_party/hadoop/hdfs.h"

// Implementation of a filesystem for HADOOP environments.
// This filesystem will support `hdfs://`, `viewfs://` and `har://` URI schemes.

static void* plugin_memory_allocate(size_t size) { return calloc(1, size); }
static void plugin_memory_free(void* ptr) { free(ptr); }

void ParseHadoopPath(const std::string& fname, std::string* scheme,
                     std::string* namenode, std::string* path) {
  size_t scheme_end = fname.find("://") + 2;
  // We don't want `://` in scheme.
  *scheme = fname.substr(0, scheme_end - 2);
  size_t nn_end = fname.find("/", scheme_end + 1);
  if (nn_end == std::string::npos) {
    *namenode = fname.substr(scheme_end + 1);
    *path = "";
    return;
  }
  *namenode = fname.substr(scheme_end + 1, nn_end - scheme_end - 1);
  // We keep `/` in path.
  *path = fname.substr(nn_end);
}

void SplitArchiveNameAndPath(std::string* path, std::string* nn,
                             TF_Status* status) {
  size_t index_end_archive_name = path->find(".har");
  if (index_end_archive_name == path->npos) {
    return TF_SetStatus(
        status, TF_INVALID_ARGUMENT,
        "Hadoop archive path does not contain a .har extension");
  }
  // Case of hadoop archive. Namenode is the path to the archive.
  std::ostringstream namenodestream;
  namenodestream << "har://" << *nn
                 << path->substr(0, index_end_archive_name + 4);
  *nn = namenodestream.str();
  path->erase(0, index_end_archive_name + 4);
  if (path->empty())
    // Root of the archive
    *path = "/";
  return TF_SetStatus(status, TF_OK, "");
}

template <typename R, typename... Args>
void BindFunc(void* handle, const char* name, std::function<R(Args...)>* func,
              TF_Status* status) {
  *func = reinterpret_cast<R (*)(Args...)>(
      TF_GetSymbolFromLibrary(handle, name, status));
}

class LibHDFS {
 public:
  explicit LibHDFS(TF_Status* status) { LoadAndBind(status); }

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
  void LoadAndBind(TF_Status* status) {
    auto TryLoadAndBind = [this](const char* name, void** handle,
                                 TF_Status* status) {
      *handle = TF_LoadSharedLibrary(name, status);
      if (TF_GetCode(status) != TF_OK) return;

#define BIND_HDFS_FUNC(function)                     \
  do {                                               \
    BindFunc(*handle, #function, &function, status); \
    if (TF_GetCode(status) != TF_OK) return;         \
  } while (0);

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
    };

    // libhdfs.so won't be in the standard locations. Use the path as specified
    // in the libhdfs documentation.
#if defined(_WIN32)
    constexpr char kLibHdfsDso[] = "hdfs.dll";
#elif defined(__GNUC__) && (defined(__APPLE_CPP__) || defined(__APPLE_CC__) || \
                            defined(__MACOS_CLASSIC__))
    constexpr char kLibHdfsDso[] = "libhdfs.dylib";
#else
    constexpr char kLibHdfsDso[] = "libhdfs.so";
#endif
    char* hdfs_home = getenv("HADOOP_HDFS_HOME");
    if (hdfs_home != nullptr) {
      auto JoinPath = [](std::string home, std::string lib) {
        if (home.back() != '/') home.push_back('/');
        return home + "lib/native/" + lib;
      };
      std::string path = JoinPath(hdfs_home, kLibHdfsDso);
      TryLoadAndBind(path.c_str(), &handle_, status);
      if (TF_GetCode(status) == TF_OK) {
        return;
      } else {
        std::cerr << "HadoopFileSystem load error: " << TF_Message(status);
      }
    }

    // Try to load the library dynamically in case it has been installed
    // to a in non-standard location.
    TryLoadAndBind(kLibHdfsDso, &handle_, status);
  }

  void* handle_;
};

// We rely on HDFS connection caching here. The HDFS client calls
// org.apache.hadoop.fs.FileSystem.get(), which caches the connection
// internally.
hdfsFS Connect(LibHDFS* libhdfs, const std::string& path, TF_Status* status) {
  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  hdfsBuilder* builder = libhdfs->hdfsNewBuilder();
  if (scheme == "file") {
    libhdfs->hdfsBuilderSetNameNode(builder, nullptr);
  } else if (scheme == "viewfs") {
    char* defaultFS = nullptr;
    libhdfs->hdfsConfGetStr("fs.defaultFS", &defaultFS);
    std::string defaultScheme, defaultCluster, defaultPath;
    ParseHadoopPath(defaultFS, &defaultScheme, &defaultCluster, &defaultPath);

    if (scheme != defaultScheme ||
        (namenode.empty() && namenode != defaultCluster)) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "viewfs is only supported as a fs.defaultFS.");
      return nullptr;
    }
    // The default NameNode configuration will be used (from the XML
    // configuration files). See:
    // https://github.com/tensorflow/tensorflow/blob/v1.0.0/third_party/hadoop/hdfs.h#L259
    libhdfs->hdfsBuilderSetNameNode(builder, "default");
  } else if (scheme == "har") {
    std::string path_har = path;
    SplitArchiveNameAndPath(&path_har, &namenode, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    libhdfs->hdfsBuilderSetNameNode(builder, namenode.c_str());
  } else {
    libhdfs->hdfsBuilderSetNameNode(
        builder, namenode.empty() ? "default" : namenode.c_str());
  }
  auto fs = libhdfs->hdfsBuilderConnect(builder);
  if (fs == nullptr)
    TF_SetStatusFromIOError(status, TF_NOT_FOUND, strerror(errno));
  else
    TF_SetStatus(status, TF_OK, "");
  return fs;
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {
typedef struct HDFSFile {
  std::string path;
  std::string hdfs_path;
  hdfsFS fs;
  LibHDFS* libhdfs;
  absl::Mutex mu;
  hdfsFile handle ABSL_GUARDED_BY(mu);
  HDFSFile(std::string path, std::string hdfs_path, hdfsFS fs, LibHDFS* libhdfs,
           hdfsFile handle)
      : path(std::move(path)),
        hdfs_path(std::move(hdfs_path)),
        fs(fs),
        libhdfs(libhdfs),
        mu(),
        handle(handle) {}
} HDFSFile;

void Cleanup(TF_RandomAccessFile* file) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  {
    absl::MutexLock l(&hdfs_file->mu);
    if (hdfs_file->handle != nullptr) {
      hdfs_file->libhdfs->hdfsCloseFile(hdfs_file->fs, hdfs_file->handle);
    }
  }
  delete hdfs_file;
}

int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
             char* buffer, TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  auto libhdfs = hdfs_file->libhdfs;
  auto fs = hdfs_file->fs;
  auto hdfs_path = hdfs_file->hdfs_path.c_str();
  auto path = hdfs_file->path.c_str();

  char* dst = buffer;
  bool eof_retried = false;
  int64_t read = 0;
  while (TF_GetCode(status) == TF_OK && n > 0) {
    // We lock inside the loop rather than outside so we don't block other
    // concurrent readers.
    absl::MutexLock l(&hdfs_file->mu);
    auto handle = hdfs_file->handle;
    // Max read length is INT_MAX-2, for hdfsPread function take a parameter
    // of int32. -2 offset can avoid JVM OutOfMemoryError.
    size_t read_n =
        (std::min)(n, static_cast<size_t>(std::numeric_limits<int>::max() - 2));
    int64_t r = libhdfs->hdfsPread(fs, handle, static_cast<tOffset>(offset),
                                   dst, static_cast<tSize>(read_n));
    if (r > 0) {
      dst += r;
      n -= r;
      offset += r;
      read += r;
    } else if (!eof_retried && r == 0) {
      // Always reopen the file upon reaching EOF to see if there's more data.
      // If writers are streaming contents while others are concurrently
      // reading, HDFS requires that we reopen the file to see updated
      // contents.
      //
      // Fixes #5438
      if (handle != nullptr && libhdfs->hdfsCloseFile(fs, handle) != 0) {
        TF_SetStatusFromIOError(status, errno, path);
        return -1;
      }
      hdfs_file->handle =
          libhdfs->hdfsOpenFile(fs, hdfs_path, O_RDONLY, 0, 0, 0);
      if (hdfs_file->handle == nullptr) {
        TF_SetStatusFromIOError(status, errno, path);
        return -1;
      }
      handle = hdfs_file->handle;
      eof_retried = true;
    } else if (eof_retried && r == 0) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read less bytes than requested");
    } else if (errno == EINTR || errno == EAGAIN) {
      // hdfsPread may return EINTR too. Just retry.
    } else {
      TF_SetStatusFromIOError(status, errno, path);
    }
  }
  return read;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {
typedef struct HDFSFile {
  std::string hdfs_path;
  hdfsFS fs;
  LibHDFS* libhdfs;
  hdfsFile handle;
  HDFSFile(std::string hdfs_path, hdfsFS fs, LibHDFS* libhdfs, hdfsFile handle)
      : hdfs_path(std::move(hdfs_path)),
        fs(fs),
        libhdfs(libhdfs),
        handle(handle) {}
} HDFSFile;

void Cleanup(TF_WritableFile* file) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  hdfs_file->libhdfs->hdfsCloseFile(hdfs_file->fs, hdfs_file->handle);
  hdfs_file->fs = nullptr;
  hdfs_file->handle = nullptr;
  delete hdfs_file;
}

void Append(const TF_WritableFile* file, const char* buffer, size_t n,
            TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  auto libhdfs = hdfs_file->libhdfs;
  auto fs = hdfs_file->fs;
  auto handle = hdfs_file->handle;

  size_t cur_pos = 0, write_len = 0;
  bool retry = false;
  // max() - 2 can avoid OutOfMemoryError in JVM .
  static const size_t max_len_once =
      static_cast<size_t>(std::numeric_limits<tSize>::max() - 2);
  while (cur_pos < n) {
    write_len = (std::min)(n - cur_pos, max_len_once);
    tSize w = libhdfs->hdfsWrite(fs, handle, buffer + cur_pos,
                                 static_cast<tSize>(write_len));
    if (w == -1) {
      if (!retry && (errno == EINTR || errno == EAGAIN)) {
        retry = true;
      } else {
        return TF_SetStatusFromIOError(status, errno,
                                       hdfs_file->hdfs_path.c_str());
      }
    } else {
      cur_pos += w;
    }
  }
  TF_SetStatus(status, TF_OK, "");
}

int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  int64_t position =
      hdfs_file->libhdfs->hdfsTell(hdfs_file->fs, hdfs_file->handle);
  if (position == -1)
    TF_SetStatusFromIOError(status, errno, hdfs_file->hdfs_path.c_str());
  else
    TF_SetStatus(status, TF_OK, "");
  return position;
}

void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  if (hdfs_file->libhdfs->hdfsHFlush(hdfs_file->fs, hdfs_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, hdfs_file->hdfs_path.c_str());
  else
    TF_SetStatus(status, TF_OK, "");
}

void Sync(const TF_WritableFile* file, TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  if (hdfs_file->libhdfs->hdfsHSync(hdfs_file->fs, hdfs_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, hdfs_file->hdfs_path.c_str());
  else
    TF_SetStatus(status, TF_OK, "");
}

void Close(const TF_WritableFile* file, TF_Status* status) {
  auto hdfs_file = static_cast<HDFSFile*>(file->plugin_file);
  TF_SetStatus(status, TF_OK, "");
  if (hdfs_file->libhdfs->hdfsCloseFile(hdfs_file->fs, hdfs_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, hdfs_file->hdfs_path.c_str());
  hdfs_file->fs = nullptr;
  hdfs_file->handle = nullptr;
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

// TODO(vnvo2409): Implement later

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_hadoop_filesystem {

void Init(TF_Filesystem* filesystem, TF_Status* status) {
  filesystem->plugin_filesystem = new LibHDFS(status);
  if (TF_GetCode(status) != TF_OK) return;
  TF_SetStatus(status, TF_OK, "");
}

void Cleanup(TF_Filesystem* filesystem) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  delete libhdfs;
}

void NewRandomAccessFile(const TF_Filesystem* filesystem, const char* path,
                         TF_RandomAccessFile* file, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  auto handle = libhdfs->hdfsOpenFile(fs, hdfs_path.c_str(), O_RDONLY, 0, 0, 0);
  if (handle == nullptr) return TF_SetStatusFromIOError(status, errno, path);

  file->plugin_file =
      new tf_random_access_file::HDFSFile(path, hdfs_path, fs, libhdfs, handle);
  TF_SetStatus(status, TF_OK, "");
}

void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                     TF_WritableFile* file, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  auto handle = libhdfs->hdfsOpenFile(fs, hdfs_path.c_str(), O_WRONLY, 0, 0, 0);
  if (handle == nullptr) return TF_SetStatusFromIOError(status, errno, path);

  file->plugin_file =
      new tf_writable_file::HDFSFile(hdfs_path, fs, libhdfs, handle);
  TF_SetStatus(status, TF_OK, "");
}

void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                       TF_WritableFile* file, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  auto handle = libhdfs->hdfsOpenFile(fs, hdfs_path.c_str(),
                                      O_WRONLY | O_APPEND, 0, 0, 0);
  if (handle == nullptr) return TF_SetStatusFromIOError(status, errno, path);

  file->plugin_file =
      new tf_writable_file::HDFSFile(hdfs_path, fs, libhdfs, handle);
  TF_SetStatus(status, TF_OK, "");
}

void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                     const char* path,
                                     TF_ReadOnlyMemoryRegion* region,
                                     TF_Status* status) {
  // hadoopReadZero() technically supports this call with the following
  // caveats:
  // - It only works up to 2 GB. We'd have to Stat() the file to ensure that
  //   it fits.
  // - If not on the local filesystem, the entire file will be read, making
  //   it inefficient for callers that assume typical mmap() behavior.
  TF_SetStatus(status, TF_UNIMPLEMENTED,
               "HDFS does not support ReadOnlyMemoryRegion");
}

void PathExists(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  if (libhdfs->hdfsExists(fs, hdfs_path.c_str()) == 0)
    TF_SetStatus(status, TF_OK, "");
  else
    TF_SetStatus(status, TF_NOT_FOUND,
                 (std::string(path) + " not found").c_str());
}

void Stat(const TF_Filesystem* filesystem, const char* path,
          TF_FileStatistics* stats, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  auto info = libhdfs->hdfsGetPathInfo(fs, hdfs_path.c_str());
  if (info == nullptr) return TF_SetStatusFromIOError(status, errno, path);

  stats->length = static_cast<int64_t>(info->mSize);
  stats->mtime_nsec = static_cast<int64_t>(info->mLastMod) * 1e9;
  stats->is_directory = info->mKind == kObjectKindDirectory;
  libhdfs->hdfsFreeFileInfo(info, 1);
  TF_SetStatus(status, TF_OK, "");
}

int64_t GetFileSize(const TF_Filesystem* filesystem, const char* path,
                    TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return -1;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  auto info = libhdfs->hdfsGetPathInfo(fs, hdfs_path.c_str());
  if (info == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return -1;
  }

  TF_SetStatus(status, TF_OK, "");
  auto size = static_cast<int64_t>(info->mSize);
  libhdfs->hdfsFreeFileInfo(info, 1);
  return size;
}

void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  if (libhdfs->hdfsDelete(fs, hdfs_path.c_str(), /*recursive=*/0) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

void CreateDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  if (libhdfs->hdfsCreateDirectory(fs, hdfs_path.c_str()) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

void DeleteDir(const TF_Filesystem* filesystem, const char* path,
               TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  // Count the number of entries in the directory, and only delete if it's
  // non-empty. This is consistent with the interface, but note that there's
  // a race condition where a file may be added after this check, in which
  // case the directory will still be deleted.
  int entries = 0;
  auto info = libhdfs->hdfsListDirectory(fs, hdfs_path.c_str(), &entries);
  if (info != nullptr) libhdfs->hdfsFreeFileInfo(info, entries);

  // Due to HDFS bug HDFS-8407, we can't distinguish between an error and empty
  // folder, especially for Kerberos enable setup, EAGAIN is quite common when
  // the call is actually successful. Check again by Stat.
  if (info == nullptr && errno != 0) {
    TF_FileStatistics stat;
    Stat(filesystem, path, &stat, status);
    if (TF_GetCode(status) != TF_OK) return;
  }

  if (entries > 0)
    return TF_SetStatus(status, TF_FAILED_PRECONDITION,
                        "Cannot delete a non-empty directory.");

  if (libhdfs->hdfsDelete(fs, hdfs_path.c_str(), /*recursive=*/1) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

void RenameFile(const TF_Filesystem* filesystem, const char* src,
                const char* dst, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, src, status);
  if (TF_GetCode(status) != TF_OK) return;

  std::string scheme, namenode, hdfs_path_src, hdfs_path_dst;
  ParseHadoopPath(src, &scheme, &namenode, &hdfs_path_src);
  ParseHadoopPath(dst, &scheme, &namenode, &hdfs_path_dst);

  if (libhdfs->hdfsExists(fs, hdfs_path_dst.c_str()) == 0 &&
      libhdfs->hdfsDelete(fs, hdfs_path_dst.c_str(), /*recursive=*/0) != 0)
    return TF_SetStatusFromIOError(status, errno, dst);

  if (libhdfs->hdfsRename(fs, hdfs_path_src.c_str(), hdfs_path_dst.c_str()) !=
      0)
    TF_SetStatusFromIOError(status, errno, src);
  else
    TF_SetStatus(status, TF_OK, "");
}

int GetChildren(const TF_Filesystem* filesystem, const char* path,
                char*** entries, TF_Status* status) {
  auto libhdfs = static_cast<LibHDFS*>(filesystem->plugin_filesystem);
  auto fs = Connect(libhdfs, path, status);
  if (TF_GetCode(status) != TF_OK) return -1;

  std::string scheme, namenode, hdfs_path;
  ParseHadoopPath(path, &scheme, &namenode, &hdfs_path);

  // hdfsListDirectory returns nullptr if the directory is empty. Do a separate
  // check to verify the directory exists first.
  TF_FileStatistics stat;
  Stat(filesystem, path, &stat, status);
  if (TF_GetCode(status) != TF_OK) return -1;

  int num_entries = 0;
  auto info = libhdfs->hdfsListDirectory(fs, hdfs_path.c_str(), &num_entries);
  if (info == nullptr) {
    if (stat.is_directory) {
      // Assume it's an empty directory.
      TF_SetStatus(status, TF_OK, "");
      return 0;
    }
    TF_SetStatusFromIOError(status, errno, path);
    return -1;
  }
  *entries = static_cast<char**>(
      plugin_memory_allocate(num_entries * sizeof((*entries)[0])));
  auto BaseName = [](const std::string& name) {
    return name.substr(name.find_last_of('/') + 1);
  };
  for (int i = 0; i < num_entries; i++) {
    (*entries)[i] = strdup(BaseName(info[i].mName).c_str());
  }
  libhdfs->hdfsFreeFileInfo(info, num_entries);
  TF_SetStatus(status, TF_OK, "");
  return num_entries;
}

// TODO(vnvo2409): Implement later

}  // namespace tf_hadoop_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = plugin_memory_allocate;
  info->plugin_memory_free = plugin_memory_free;
  info->num_schemes = 3;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      plugin_memory_allocate(info->num_schemes * sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "hdfs");
  ProvideFilesystemSupportFor(&info->ops[1], "viewfs");
  ProvideFilesystemSupportFor(&info->ops[2], "har");
}
