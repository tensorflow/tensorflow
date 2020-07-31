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
  *scheme = fname.substr(0, scheme_end + 1);
  size_t nn_end = fname.find("/", scheme_end + 1);
  if (nn_end == std::string::npos) return;
  *namenode = fname.substr(scheme_end + 1, nn_end - scheme_end - 1);
  *path = fname.substr(nn_end + 1);
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
  namenodestream << "har://" << nn
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
  LibHDFS(TF_Status* status) { LoadAndBind(status); }

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

#define BIND_HDFS_FUNC(function)                   \
  BindFunc(*handle, #function, &function, status); \
  if (TF_GetCode(status) != TF_OK) return

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

static const LibHDFS* libhdfs(TF_Status* status) {
  static const LibHDFS* libhdfs = new LibHDFS(status);
  return libhdfs;
}

// We rely on HDFS connection caching here. The HDFS client calls
// org.apache.hadoop.fs.FileSystem.get(), which caches the connection
// internally.
hdfsFS Connect(const std::string& path, TF_Status* status) {
  auto hdfs_file = libhdfs(status);
  if (TF_GetCode(status) != TF_OK) return nullptr;

  std::string scheme, namenode, nodepath;
  ParseHadoopPath(path, &scheme, &namenode, &nodepath);

  hdfsBuilder* builder = hdfs_file->hdfsNewBuilder();
  if (scheme == "file") {
    hdfs_file->hdfsBuilderSetNameNode(builder, nullptr);
  } else if (scheme == "viewfs") {
    char* defaultFS = nullptr;
    hdfs_file->hdfsConfGetStr("fs.defaultFS", &defaultFS);
    std::string defaultScheme, defaultCluster, defaultPath;
    ParseHadoopPath(defaultFS, &defaultScheme, &defaultCluster, &defaultPath);

    if (scheme != defaultScheme ||
        (namenode != "" && namenode != defaultCluster)) {
      TF_SetStatus(status, TF_UNIMPLEMENTED,
                   "viewfs is only supported as a fs.defaultFS.");
      return nullptr;
    }
    // The default NameNode configuration will be used (from the XML
    // configuration files). See:
    // https://github.com/tensorflow/tensorflow/blob/v1.0.0/third_party/hadoop/hdfs.h#L259
    hdfs_file->hdfsBuilderSetNameNode(builder, "default");
  } else if (scheme == "har") {
    std::string path_har = path;
    SplitArchiveNameAndPath(&path_har, &namenode, status);
    if (TF_GetCode(status) != TF_OK) return nullptr;
    hdfs_file->hdfsBuilderSetNameNode(builder, namenode.c_str());
  } else {
    hdfs_file->hdfsBuilderSetNameNode(
        builder, namenode.empty() ? "default" : namenode.c_str());
  }
  auto fs = hdfs_file->hdfsBuilderConnect(builder);
  if (fs == nullptr)
    TF_SetStatusFromIOError(status, TF_NOT_FOUND, strerror(errno));
  else
    TF_SetStatus(status, TF_OK, "");
  return fs;
}

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

// TODO(vnvo2409): Implement later

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

// TODO(vnvo2409): Implement later

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
  TF_SetStatus(status, TF_OK, "");
}

void Cleanup(TF_Filesystem* filesystem) {}

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
