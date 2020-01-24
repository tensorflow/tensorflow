/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
#include "tensorflow/c/experimental/filesystem/plugins/posix/posix_filesystem_helper.h"
#include "tensorflow/c/tf_status.h"

// Implementation of a filesystem for POSIX environments.
// This filesystem will support `file://` and empty (local) URI schemes.

// SECTION 1. Implementation for `TF_RandomAccessFile`
// ----------------------------------------------------------------------------
namespace tf_random_access_file {

typedef struct PosixFile {
  const char* filename;
  int fd;
} PosixFile;

static void Cleanup(TF_RandomAccessFile* file) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  close(posix_file->fd);
  free(const_cast<char*>(posix_file->filename));
  delete posix_file;
}

static int64_t Read(const TF_RandomAccessFile* file, uint64_t offset, size_t n,
                    char* buffer, TF_Status* status) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  char* dst = buffer;
  int64_t read = 0;

  while (n > 0) {
    // Some platforms, notably macs, throw `EINVAL` if `pread` is asked to read
    // more than fits in a 32-bit integer.
    size_t requested_read_length;
    if (n > INT32_MAX)
      requested_read_length = INT32_MAX;
    else
      requested_read_length = n;

    // `pread` returns a `ssize_t` on POSIX, but due to interface being
    // cross-platform, return type of `Read` is `int64_t`.
    int64_t r = int64_t{pread(posix_file->fd, dst, requested_read_length,
                              static_cast<off_t>(offset))};
    if (r > 0) {
      dst += r;
      offset += static_cast<uint64_t>(r);
      n -= r;  // safe as 0 < r <= n so n will never underflow
      read += r;
    } else if (r == 0) {
      TF_SetStatus(status, TF_OUT_OF_RANGE, "Read fewer bytes than requested");
      break;
    } else if (errno == EINTR || errno == EAGAIN) {
      // Retry
    } else {
      TF_SetStatusFromIOError(status, errno, posix_file->filename);
      break;
    }
  }

  return read;
}

}  // namespace tf_random_access_file

// SECTION 2. Implementation for `TF_WritableFile`
// ----------------------------------------------------------------------------
namespace tf_writable_file {

typedef struct PosixFile {
  const char* filename;
  FILE* handle;
} PosixFile;

static void Cleanup(TF_WritableFile* file) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);
  free(const_cast<char*>(posix_file->filename));
  delete posix_file;
}

static void Append(const TF_WritableFile* file, const char* buffer, size_t n,
                   TF_Status* status) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  size_t r = fwrite(buffer, 1, n, posix_file->handle);
  if (r != n)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");
}

static int64_t Tell(const TF_WritableFile* file, TF_Status* status) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  // POSIX's `ftell` returns `long`, do a manual cast.
  int64_t position = int64_t{ftell(posix_file->handle)};
  if (position < 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");

  return position;
}

static void Flush(const TF_WritableFile* file, TF_Status* status) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  TF_SetStatus(status, TF_OK, "");
  if (fflush(posix_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
}

static void Sync(const TF_WritableFile* file, TF_Status* status) {
  // For historical reasons, this does the same as `Flush` at the moment.
  // TODO(b/144055243): This should use `fsync`/`sync`.
  Flush(file, status);
}

static void Close(const TF_WritableFile* file, TF_Status* status) {
  auto posix_file = static_cast<PosixFile*>(file->plugin_file);

  if (fclose(posix_file->handle) != 0)
    TF_SetStatusFromIOError(status, errno, posix_file->filename);
  else
    TF_SetStatus(status, TF_OK, "");
}

}  // namespace tf_writable_file

// SECTION 3. Implementation for `TF_ReadOnlyMemoryRegion`
// ----------------------------------------------------------------------------
namespace tf_read_only_memory_region {

typedef struct PosixMemoryRegion {
  const void* const address;
  const uint64_t length;
} PosixMemoryRegion;

static void Cleanup(TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  munmap(const_cast<void*>(r->address), r->length);
  delete r;
}

static const void* Data(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  return r->address;
}

static uint64_t Length(const TF_ReadOnlyMemoryRegion* region) {
  auto r = static_cast<PosixMemoryRegion*>(region->plugin_memory_region);
  return r->length;
}

}  // namespace tf_read_only_memory_region

// SECTION 4. Implementation for `TF_Filesystem`, the actual filesystem
// ----------------------------------------------------------------------------
namespace tf_posix_filesystem {

static void Init(TF_Filesystem* filesystem, TF_Status* status) {
  TF_SetStatus(status, TF_OK, "");
}

static void Cleanup(TF_Filesystem* filesystem) {}

static void NewRandomAccessFile(const TF_Filesystem* filesystem,
                                const char* path, TF_RandomAccessFile* file,
                                TF_Status* status) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  struct stat st;
  fstat(fd, &st);
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
    close(fd);
    return;
  }

  file->plugin_file = new tf_random_access_file::PosixFile({strdup(path), fd});
  TF_SetStatus(status, TF_OK, "");
}

static void NewWritableFile(const TF_Filesystem* filesystem, const char* path,
                            TF_WritableFile* file, TF_Status* status) {
  FILE* f = fopen(path, "w");
  if (f == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  file->plugin_file = new tf_writable_file::PosixFile({strdup(path), f});
  TF_SetStatus(status, TF_OK, "");
}

static void NewAppendableFile(const TF_Filesystem* filesystem, const char* path,
                              TF_WritableFile* file, TF_Status* status) {
  FILE* f = fopen(path, "a");
  if (f == nullptr) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  file->plugin_file = new tf_writable_file::PosixFile({strdup(path), f});
  TF_SetStatus(status, TF_OK, "");
}

static void NewReadOnlyMemoryRegionFromFile(const TF_Filesystem* filesystem,
                                            const char* path,
                                            TF_ReadOnlyMemoryRegion* region,
                                            TF_Status* status) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) {
    TF_SetStatusFromIOError(status, errno, path);
    return;
  }

  struct stat st;
  fstat(fd, &st);
  if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "path is a directory");
  } else {
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      TF_SetStatusFromIOError(status, errno, path);
    } else {
      region->plugin_memory_region =
          new tf_read_only_memory_region::PosixMemoryRegion{
              address, static_cast<uint64_t>(st.st_size)};
      TF_SetStatus(status, TF_OK, "");
    }
  }

  close(fd);
}

static void CreateDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  if (strlen(path) == 0)
    TF_SetStatus(status, TF_ALREADY_EXISTS, "already exists");
  else if (mkdir(path, /*mode=*/0755) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void DeleteFile(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  if (unlink(path) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void DeleteDir(const TF_Filesystem* filesystem, const char* path,
                      TF_Status* status) {
  if (rmdir(path) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void RenameFile(const TF_Filesystem* filesystem, const char* src,
                       const char* dst, TF_Status* status) {
  // If target is a directory return TF_FAILED_PRECONDITION.
  // Target might be missing, so don't error in that case.
  struct stat st;
  if (stat(dst, &st) != 0) {
    if (errno != ENOENT) {
      TF_SetStatusFromIOError(status, errno, dst);
      return;
    }
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "target path is a directory");
    return;
  }

  // We cannot rename directories yet, so prevent this.
  if (stat(src, &st) != 0) {
    TF_SetStatusFromIOError(status, errno, src);
    return;
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "source path is a directory");
    return;
  }

  // Do the actual rename. Here both arguments are filenames.
  if (rename(src, dst) != 0)
    TF_SetStatusFromIOError(status, errno, dst);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void CopyFile(const TF_Filesystem* filesystem, const char* src,
                     const char* dst, TF_Status* status) {
  // If target is a directory return TF_FAILED_PRECONDITION.
  // Target might be missing, so don't error in that case.
  struct stat st;
  if (stat(dst, &st) != 0) {
    if (errno != ENOENT) {
      TF_SetStatusFromIOError(status, errno, dst);
      return;
    }
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "target path is a directory");
    return;
  }

  // We cannot copy directories yet, so prevent this.
  if (stat(src, &st) != 0) {
    TF_SetStatusFromIOError(status, errno, src);
    return;
  } else if (S_ISDIR(st.st_mode)) {
    TF_SetStatus(status, TF_FAILED_PRECONDITION, "source path is a directory");
    return;
  }

  // Both `src` and `dst` point to files here. Delegate to helper.
  if (TransferFileContents(src, dst, st.st_mode, st.st_size) < 0)
    TF_SetStatusFromIOError(status, errno, dst);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void PathExists(const TF_Filesystem* filesystem, const char* path,
                       TF_Status* status) {
  if (access(path, F_OK) != 0)
    TF_SetStatusFromIOError(status, errno, path);
  else
    TF_SetStatus(status, TF_OK, "");
}

static void Stat(const TF_Filesystem* filesystem, const char* path,
                 TF_FileStatistics* stats, TF_Status* status) {
  struct stat sbuf;
  if (stat(path, &sbuf) != 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * (1000 * 1000 * 1000);
    stats->is_directory = S_ISDIR(sbuf.st_mode);
    TF_SetStatus(status, TF_OK, "");
  }
}

static int GetChildren(const TF_Filesystem* filesystem, const char* path,
                       char*** entries, TF_Status* status) {
  struct dirent** dir_entries = nullptr;
  /* we don't promise entries would be sorted */
  int num_entries =
      scandir(path, &dir_entries, RemoveSpecialDirectoryEntries, nullptr);
  if (num_entries < 0) {
    TF_SetStatusFromIOError(status, errno, path);
  } else {
    *entries = static_cast<char**>(calloc(num_entries, sizeof((*entries)[0])));
    for (int i = 0; i < num_entries; i++) {
      (*entries)[i] = strdup(dir_entries[i]->d_name);
      free(dir_entries[i]);
    }
    free(dir_entries);
  }

  return num_entries;
}

}  // namespace tf_posix_filesystem

int TF_InitPlugin(void* (*allocator)(size_t), TF_FilesystemPluginInfo** info) {
  const int num_schemes = 2;
  *info = static_cast<TF_FilesystemPluginInfo*>(
      allocator(num_schemes * sizeof((*info)[0])));

  for (int i = 0; i < num_schemes; i++) {
    TF_FilesystemPluginInfo* current_info = &((*info)[i]);
    TF_SetFilesystemVersionMetadata(current_info);

    current_info->random_access_file_ops = static_cast<TF_RandomAccessFileOps*>(
        allocator(TF_RANDOM_ACCESS_FILE_OPS_SIZE));
    current_info->random_access_file_ops->cleanup =
        tf_random_access_file::Cleanup;
    current_info->random_access_file_ops->read = tf_random_access_file::Read;

    current_info->writable_file_ops =
        static_cast<TF_WritableFileOps*>(allocator(TF_WRITABLE_FILE_OPS_SIZE));
    current_info->writable_file_ops->cleanup = tf_writable_file::Cleanup;
    current_info->writable_file_ops->append = tf_writable_file::Append;
    current_info->writable_file_ops->tell = tf_writable_file::Tell;
    current_info->writable_file_ops->flush = tf_writable_file::Flush;
    current_info->writable_file_ops->sync = tf_writable_file::Sync;
    current_info->writable_file_ops->close = tf_writable_file::Close;

    current_info->read_only_memory_region_ops =
        static_cast<TF_ReadOnlyMemoryRegionOps*>(
            allocator(TF_READ_ONLY_MEMORY_REGION_OPS_SIZE));
    current_info->read_only_memory_region_ops->cleanup =
        tf_read_only_memory_region::Cleanup;
    current_info->read_only_memory_region_ops->data =
        tf_read_only_memory_region::Data;
    current_info->read_only_memory_region_ops->length =
        tf_read_only_memory_region::Length;

    current_info->filesystem_ops =
        static_cast<TF_FilesystemOps*>(allocator(TF_FILESYSTEM_OPS_SIZE));
    current_info->filesystem_ops->init = tf_posix_filesystem::Init;
    current_info->filesystem_ops->cleanup = tf_posix_filesystem::Cleanup;
    current_info->filesystem_ops->new_random_access_file =
        tf_posix_filesystem::NewRandomAccessFile;
    current_info->filesystem_ops->new_writable_file =
        tf_posix_filesystem::NewWritableFile;
    current_info->filesystem_ops->new_appendable_file =
        tf_posix_filesystem::NewAppendableFile;
    current_info->filesystem_ops->new_read_only_memory_region_from_file =
        tf_posix_filesystem::NewReadOnlyMemoryRegionFromFile;
    current_info->filesystem_ops->create_dir = tf_posix_filesystem::CreateDir;
    current_info->filesystem_ops->delete_file = tf_posix_filesystem::DeleteFile;
    current_info->filesystem_ops->delete_dir = tf_posix_filesystem::DeleteDir;
    current_info->filesystem_ops->rename_file = tf_posix_filesystem::RenameFile;
    current_info->filesystem_ops->copy_file = tf_posix_filesystem::CopyFile;
    current_info->filesystem_ops->path_exists = tf_posix_filesystem::PathExists;
    current_info->filesystem_ops->stat = tf_posix_filesystem::Stat;
    current_info->filesystem_ops->get_children =
        tf_posix_filesystem::GetChildren;
  }

  (*info)[0].scheme = strdup("");
  (*info)[1].scheme = strdup("file");

  return num_schemes;
}
