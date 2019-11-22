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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <vector>

#include "tensorflow/c/experimental/filesystem/filesystem_interface.h"
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

static int RemoveSpecialDirectoryEntries(const struct dirent* d) {
  return strcmp(d->d_name, ".") != 0 && strcmp(d->d_name, "..") != 0;
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

void TF_InitPlugin(TF_Status* status) {
  TF_RandomAccessFileOps random_access_file_ops = {
      tf_random_access_file::Cleanup,
      tf_random_access_file::Read,
  };
  TF_WritableFileOps writable_file_ops = {
      tf_writable_file::Cleanup, tf_writable_file::Append,
      tf_writable_file::Tell,    tf_writable_file::Flush,
      tf_writable_file::Sync,    tf_writable_file::Close,
  };
  TF_ReadOnlyMemoryRegionOps read_only_memory_region_ops = {
      tf_read_only_memory_region::Cleanup,
      tf_read_only_memory_region::Data,
      tf_read_only_memory_region::Length,
  };
  TF_FilesystemOps filesystem_ops = {
      tf_posix_filesystem::Init,
      tf_posix_filesystem::Cleanup,
      tf_posix_filesystem::NewRandomAccessFile,
      tf_posix_filesystem::NewWritableFile,
      tf_posix_filesystem::NewAppendableFile,
      tf_posix_filesystem::NewReadOnlyMemoryRegionFromFile,
      tf_posix_filesystem::CreateDir,
      /*recursively_create_dir=*/nullptr,
      tf_posix_filesystem::DeleteFile,
      tf_posix_filesystem::DeleteDir,
      /*delete_recursively=*/nullptr,
      /*rename_file=*/nullptr,
      /*copy_file=*/nullptr,
      tf_posix_filesystem::PathExists,
      /*paths_exist=*/nullptr,
      tf_posix_filesystem::Stat,
      /*is_directory=*/nullptr,
      /*get_file_size=*/nullptr,
      /*translate_name=*/nullptr,
      tf_posix_filesystem::GetChildren,
      nullptr,
  };

  for (const char* scheme : {"", "file"})
    TF_REGISTER_FILESYSTEM_PLUGIN(scheme, &filesystem_ops,
                                  &random_access_file_ops, &writable_file_ops,
                                  &read_only_memory_region_ops, status);
}
