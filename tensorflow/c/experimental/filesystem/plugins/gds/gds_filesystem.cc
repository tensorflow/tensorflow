/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/experimental/filesystem/plugins/gds/gds_filesystem.h"

#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mutex>  // NOLINT

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_file_statistics.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem plugin for the "gds" URI scheme
// (e.g. "gds:///data/shard-00001.tfrecord") backed by NVIDIA GPUDirect
// Storage.

namespace {

// One process-wide cuFile driver session, opened lazily and intentionally kept
// alive for the process lifetime.
class CuFileDriver {
 public:
  static CuFileDriver* Get() {
    static CuFileDriver* driver = new CuFileDriver();
    return driver;
  }

  bool available() const { return available_; }

 private:
  CuFileDriver() {
    CUfileError_t status = cuFileDriverOpen();
    available_ = (status.err == CU_FILE_SUCCESS);
  }

  bool available_;
};

}  // namespace

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct GdsFile {
  const char* filename;
  int host_fd;
  int gds_fd;
  CUfileHandle_t cf_handle;
  bool cf_registered;
} GdsFile;

static void Cleanup(TF_RandomAccessFile* file) {
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  if (gds_file->cf_registered) {
    cuFileHandleDeregister(gds_file->cf_handle);
  }
  if (gds_file->gds_fd >= 0 && gds_file->gds_fd != gds_file->host_fd) {
    close(gds_file->gds_fd);
  }
  if (gds_file->host_fd >= 0) {
    close(gds_file->host_fd);
  }
  free(const_cast<char*>(gds_file->filename));
  delete gds_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  char* dst = buffer;
  int64_t total_read = 0;

  while (n > 0) {
    size_t requested = (n > INT32_MAX) ? INT32_MAX : n;
    int64_t r = static_cast<int64_t>(
        pread(gds_file->host_fd, dst, requested, static_cast<off_t>(offset)));
    if (r > 0) {
      dst += r;
      offset += static_cast<uint64_t>(r);
      n -= r;
      total_read += r;
    } else if (r == 0) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
      return total_read;
    } else if (errno == EINTR || errno == EAGAIN) {
      continue;
    } else {
      TF_SetStatusFromIOError(status, errno, gds_file->filename);
      return total_read;
    }
  }

  TF_SetStatus(status, TF_OK, "");
  return total_read;
}

static int64_t ReadToDevice(const TF_RandomAccessFile* file, uint64_t offset,
                            size_t n, void* device_ptr, TF_Status* status) {
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  if (!gds_file->cf_registered) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "GDS not available for this file; fall back to Read()");
    return -1;
  }

  ssize_t r = cuFileRead(gds_file->cf_handle, device_ptr, n, offset, 0);
  if (r < 0) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "GDS direct read is not available for this file");
    return -1;
  }
  if (static_cast<size_t>(r) < n) {
    TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
    return static_cast<int64_t>(r);
  }

  TF_SetStatus(status, TF_OK, "");
  return static_cast<int64_t>(r);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_Filesystem`
// ----------------------------------------------------------------------------
namespace tf_gds_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  CuFileDriver::Get();
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  int host_fd = open(path, O_RDONLY);
  if (host_fd < 0) {
    TF_SetStatus(status, TF_NOT_FOUND, strerror(errno));
    return;
  }

  struct stat st;
  if (fstat(host_fd, &st) != 0) {
    TF_SetStatus(status, TF_UNKNOWN, strerror(errno));
    close(host_fd);
    return;
  }
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    close(host_fd);
    return;
  }

  int gds_fd = -1;
  if (CuFileDriver::Get()->available()) {
    gds_fd = open(path, O_RDONLY | O_DIRECT);
    if (gds_fd >= 0) {
      auto* gds_file = new tf_random_access_file::GdsFile();
      gds_file->filename = strdup(path);
      gds_file->host_fd = host_fd;
      gds_file->gds_fd = gds_fd;
      gds_file->cf_registered = false;

      CUfileDescr_t cf_descr{};
      cf_descr.handle.fd = gds_fd;
      cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
      CUfileError_t reg_status =
          cuFileHandleRegister(&gds_file->cf_handle, &cf_descr);
      if (reg_status.err == CU_FILE_SUCCESS) {
        gds_file->cf_registered = true;
      } else {
        close(gds_fd);
        gds_file->gds_fd = -1;
      }

      file->plugin_file = gds_file;
      TF_SetStatus(status, TF_OK, "");
      return;
    }
  }

  auto* gds_file = new tf_random_access_file::GdsFile();
  gds_file->filename = strdup(path);
  gds_file->host_fd = host_fd;
  gds_file->gds_fd = -1;
  gds_file->cf_registered = false;
  file->plugin_file = gds_file;
  TF_SetStatus(status, TF_OK, "");
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  if (access(path, F_OK) != 0) {
    TF_SetStatus(status, TF_NOT_FOUND, strerror(errno));
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  struct stat sbuf;
  if (stat(path, &sbuf) != 0) {
    TF_SetStatus(status, TF_NOT_FOUND, strerror(errno));
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * (1000 * 1000 * 1000);
    stats->is_directory = S_ISDIR(sbuf.st_mode);
    TF_SetStatus(status, TF_OK, "");
  }
}

static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  return strdup(uri);
}

}  // namespace tf_gds_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                        const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      calloc(1, TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;
  ops->random_access_file_ops->read_to_device =
      tf_random_access_file::ReadToDevice;

  ops->filesystem_ops =
      static_cast<TF_FilesystemOps*>(calloc(1, TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_gds_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_gds_filesystem::Cleanup;
  ops->filesystem_ops->new_random_access_file =
      tf_gds_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->path_exists = tf_gds_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_gds_filesystem::Stat;
  ops->filesystem_ops->translate_name = tf_gds_filesystem::TranslateName;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = malloc;
  info->plugin_memory_free = free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      calloc(info->num_schemes, sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gds");
}