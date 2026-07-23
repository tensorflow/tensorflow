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

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/tf_file_statistics.h"
#include "tensorflow/c/tf_status.h"

#if defined(__linux__)

#include <cufile.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <mutex>  // NOLINT

// Implementation of a filesystem plugin for the "gds" URI scheme
// (e.g. "gds:///data/shard-00001.tfrecord") backed by NVIDIA GPUDirect
// Storage. See gds_filesystem.h for scope and fallback behavior.
//
// NOTE ON SCOPE (first PR): this plugin proves out the cuFile driver/handle
// lifecycle and provides a `read_to_device` capability behind a proposed
// extension to TF_RandomAccessFileOps (see filesystem_interface.patch in the
// same change). It intentionally does NOT yet wire `read_to_device` into any
// tf.data reader or checkpoint path -- that is separate, larger follow-up
// work once the interface extension itself is reviewed. Standard `read()`
// (required by the existing ABI) always works via pread() so this plugin is
// safe to register and use as a plain file backend even where GDS itself
// is unavailable.

namespace {

// One process-wide cuFile driver session, opened lazily and closed once.
// cuFileDriverOpen/Close are documented as needing to bracket all other
// cuFile calls; TensorFlow's modular filesystem never unloads plugin DSOs
// (see filesystem_interface.h), so a static, never-closed session mirrors
// how in-process cuFile users normally do this.
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
    // Deliberately do not treat failure as fatal: every call site falls
    // back to plain pread() when `available_` is false, so a machine
    // without nvidia-fs.ko, without a supported GPU, or without a
    // GDS-enabled filesystem still works, just without the DMA fast path.
  }

  bool available_;
};

}  // namespace

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct GdsFile {
  const char* filename;
  int fd;
  CUfileHandle_t cf_handle;
  bool cf_registered;
} GdsFile;

static void Cleanup(TF_RandomAccessFile* file) {
  if (file == nullptr || file->plugin_file == nullptr) {
    return;
  }
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  if (gds_file->cf_registered) {
    cuFileHandleDeregister(gds_file->cf_handle);
  }
  close(gds_file->fd);
  free(const_cast<char*>(gds_file->filename));
  delete gds_file;
}

// Required op: fills a host-memory `buffer`. Used by every existing TF
// code path (tf.data readers, checkpoint loaders, etc.) that has not been
// updated to ask for a device destination. Implemented as plain pread() --
// correct and portable even when cuFile/GDS is unavailable.
static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset,
                     size_t n, char* buffer, TF_Status* status) {
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  char* dst = buffer;
  int64_t total_read = 0;

  while (n > 0) {
    size_t requested = (n > INT32_MAX) ? INT32_MAX : n;
    int64_t r = static_cast<int64_t>(
        pread(gds_file->fd, dst, requested, static_cast<off_t>(offset)));
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

// PROPOSED new op (see filesystem_interface.patch): reads directly into a
// caller-registered GPU buffer via cuFileRead, bypassing the host bounce
// buffer entirely. `device_ptr` must already be registered with
// cuFileBufferRegister by the caller -- that registration is deliberately
// kept out of this plugin because its lifetime is tied to the tensor
// allocation, not to the file handle.
//
// Returns false (via status TF_UNIMPLEMENTED) if GDS is not available so
// callers know to fall back to Read() + a separate host-to-device copy.
static int64_t ReadToDevice(const TF_RandomAccessFile* file, uint64_t offset,
                             size_t n, void* device_ptr, TF_Status* status) {
  auto gds_file = static_cast<GdsFile*>(file->plugin_file);
  if (!gds_file->cf_registered) {
    TF_SetStatus(status, TF_UNIMPLEMENTED,
                 "GDS not available for this file; fall back to Read()");
    return -1;
  }

  ssize_t r = cuFileRead(gds_file->cf_handle, device_ptr,
                         /*size=*/n, /*file_offset=*/offset,
                         /*devPtr_offset=*/0);
  if (r < 0) {
    // cuFileRead returns -1 for a POSIX-style error (errno is set), or a
    // negative CUfileOpError enum value for a cuFile/driver-level failure.
    // Collapsing both into a single generic status would mask real I/O or
    // runtime errors from callers, so propagate the specific one.
    if (r == -1) {
      TF_SetStatusFromIOError(status, errno, gds_file->filename);
    } else {
      TF_SetStatus(status, TF_INTERNAL, CUFILE_ERRSTR(r));
    }
    return -1;
  }
  TF_SetStatus(status, TF_OK, "");
  return static_cast<int64_t>(r);
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_Filesystem`
// ----------------------------------------------------------------------------
namespace tf_gds_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  // Touch the driver singleton so failure to open surfaces early via logs,
  // but never fail Init() itself -- see CuFileDriver's fallback comment.
  CuFileDriver::Get();
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

// GDS is a local-backed filesystem: every op below eventually calls a POSIX
// function (open, stat, access, ...) with the path it is given. Those calls
// need a bare filesystem path, not a "gds://..." URI, so strip the scheme
// here rather than at every call site.
static char* TranslateName(const TF_Filesystem* filesystem, const char* uri) {
  if (strncmp(uri, "gds://", 6) == 0) {
    const char* path = uri + 6;
    const char* slash = strchr(path, '/');
    if (slash != nullptr) {
      return strdup(slash);
    }
    // "gds://" with no path at all -- fall back to root rather than
    // handing POSIX calls an empty string.
    return strdup("/");
  }
  return strdup(uri);
}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                 const char* path, TF_RandomAccessFile* file,
                                 TF_Status* status) {
  // O_DIRECT is required for the DMA path; see the GDS O_DIRECT
  // Requirements Guide. If this open fails (e.g. filesystem doesn't
  // support O_DIRECT), fall back to a buffered open so the file is still
  // usable through the plain Read() path above.
  int fd = open(path, O_RDONLY | O_DIRECT);
  if (fd < 0) {
    fd = open(path, O_RDONLY);
  }
  if (fd < 0) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    TF_SetStatusFromIOError(status, errno, path);
    close(fd);
    return;
  }
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    close(fd);
    return;
  }

  auto* gds_file = new tf_random_access_file::GdsFile();
  gds_file->filename = strdup(path);
  gds_file->fd = fd;
  gds_file->cf_registered = false;

  if (CuFileDriver::Get()->available()) {
    CUfileDescr_t cf_descr{};
    cf_descr.handle.fd = fd;
    cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    CUfileError_t reg_status =
        cuFileHandleRegister(&gds_file->cf_handle, &cf_descr);
    gds_file->cf_registered = (reg_status.err == CU_FILE_SUCCESS);
    // If registration fails (e.g. filesystem not GDS-enabled for this
    // mount), we keep the plain fd open and just never set cf_registered;
    // ReadToDevice() reports TF_UNIMPLEMENTED and Read() keeps working.
  }

  file->plugin_file = gds_file;
  TF_SetStatus(status, TF_OK, "");
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                        TF_Status* status) {
  if (access(path, F_OK) != 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    TF_SetStatus(status, TF_OK, "");
  }
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                  TF_FileStatistics* stats, TF_Status* status) {
  struct stat sbuf;
  if (stat(path, &sbuf) != 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    stats->length = sbuf.st_size;
    // Use st_mtim (not st_mtime) to preserve sub-second precision;
    // st_mtime is just the whole-seconds field and multiplying it by 1e9
    // would silently drop the nanosecond component.
    stats->mtime_nsec = static_cast<int64_t>(sbuf.st_mtim.tv_sec) * 1000000000LL +
                         sbuf.st_mtim.tv_nsec;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
    TF_SetStatus(status, TF_OK, "");
  }
}

// NOTE: this plugin is read-path only for its first version. Write support
// (NewWritableFile/NewAppendableFile) and directory ops (CreateDir,
// GetChildren, ...) are left unimplemented on purpose -- training/inference
// input pipelines are the read-heavy, GDS-relevant case; add write support
// only if a checkpoint-writing use case actually needs it.

}  // namespace tf_gds_filesystem

static void ProvideFilesystemSupportFor(TF_FilesystemPluginOps* ops,
                                         const char* uri) {
  TF_SetFilesystemVersionMetadata(ops);
  ops->scheme = strdup(uri);

  ops->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
      calloc(1, TF_RANDOM_ACCESS_FILE_OPS_SIZE));
  ops->random_access_file_ops->cleanup = tf_random_access_file::Cleanup;
  ops->random_access_file_ops->read = tf_random_access_file::Read;
  // `read_to_device` is the proposed new field appended at the end of
  // TF_RandomAccessFileOps -- see filesystem_interface.patch.
  ops->random_access_file_ops->read_to_device =
      tf_random_access_file::ReadToDevice;

  ops->filesystem_ops =
      static_cast<TF_FilesystemOps*>(calloc(1, TF_FILESYSTEM_OPS_SIZE));
  ops->filesystem_ops->init = tf_gds_filesystem::Init;
  ops->filesystem_ops->cleanup = tf_gds_filesystem::Cleanup;
  ops->filesystem_ops->translate_name = tf_gds_filesystem::TranslateName;
  ops->filesystem_ops->new_random_access_file =
      tf_gds_filesystem::NewRandomAccessFile;
  ops->filesystem_ops->path_exists = tf_gds_filesystem::PathExists;
  ops->filesystem_ops->stat = tf_gds_filesystem::Stat;
}

void TF_InitPlugin(TF_FilesystemPluginInfo* info) {
  info->plugin_memory_allocate = malloc;
  info->plugin_memory_free = free;
  info->num_schemes = 1;
  info->ops = static_cast<TF_FilesystemPluginOps*>(
      calloc(info->num_schemes, sizeof(info->ops[0])));
  ProvideFilesystemSupportFor(&info->ops[0], "gds");
}

#else

void TF_InitPlugin(TF_FilesystemPluginInfo* info) { (void)info; }

#endif  // defined(__linux__)