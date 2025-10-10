/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/xnnpack/mmap_handle.h"

#if defined(_MSC_VER)
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/windows_util.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

#define XNNPACK_VAR_ARG_HEAD(FIRST, ...) FIRST

#define XNNPACK_RETURN_CHECK(TEST, ...)                              \
  if (!(TEST)) {                                                     \
    if (sizeof(XNNPACK_VAR_ARG_HEAD("" __VA_ARGS__)) > sizeof("")) { \
      TFLITE_LOG_PROD(tflite::TFLITE_LOG_ERROR,                      \
                      "XNNPack weight cache: " __VA_ARGS__);         \
    }                                                                \
    return false;                                                    \
  }

namespace tflite::xnnpack {

void swap(MMapHandle& a, MMapHandle& b) {
  using std::swap;
  swap(a.size_, b.size_);
  swap(a.offset_, b.offset_);
  swap(a.offset_page_adjustment_, b.offset_page_adjustment_);
  swap(a.data_, b.data_);
}

MMapHandle::~MMapHandle() { UnMap(); }

MMapHandle::MMapHandle(MMapHandle&& other) { swap(*this, other); }

MMapHandle& MMapHandle::operator=(MMapHandle&& other) {
  swap(*this, other);
  return *this;
}

bool MMapHandle::Map(const char* path, const size_t offset) {
  return this->Map(FileDescriptor::Open(path, O_RDONLY), offset, path);
}

bool MMapHandle::Map(const FileDescriptorView& fd, const size_t offset,
                     const char* const path) {
  this->UnMap();
  const char* const safe_path = path != nullptr ? path : "[unspecified]";

  XNNPACK_RETURN_CHECK(fd.IsValid(),
                       "cannot mmap invalid file descriptor %d ('%s').",
                       fd.Value(), path);

#if defined(_MSC_VER)
  struct _stat64 file_stats;
  XNNPACK_RETURN_CHECK(_fstat64(fd.Value(), &file_stats) == 0,
                       "could not access file stats to get size ('%s'): %s.",
                       safe_path, strerror(errno));
#else
  struct stat file_stats;
  XNNPACK_RETURN_CHECK(fstat(fd.Value(), &file_stats) == 0,
                       "could not access file stats to get size ('%s'): %s.",
                       safe_path, strerror(errno));
#endif

  // This will reset data_ and size_ on return until it is deactivated.
  ScopeGuard unmap_on_error([this] { UnMap(); });
  size_ = file_stats.st_size - offset;
  offset_ = offset;
#if defined(XNNPACK_CACHE_NO_MMAP_FOR_TEST)
  // This allocation is freed in UnMap and in the destructor.
  data_ = new uint8_t[size_];
  fd.SetPos(offset);
  XNNPACK_RETURN_CHECK(fd.Read(data_, size_), "could not read file ('%s'): %s.",
                       safe_path, strerror(errno));
#elif defined(_MSC_VER)
  HANDLE osf_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd.Value()));
  XNNPACK_RETURN_CHECK(osf_handle != INVALID_HANDLE_VALUE,
                       "could not convert file descriptor to file handle: %s.",
                       strerror(errno));

  // Create the handle name, which is either NULL or the path with backslashes
  // replaced by `_`.
  std::string name;
  const char* handle_name = nullptr;
  if (path && path[0] != '\0') {
    name = path;
    std::replace(name.begin(), name.end(), '\\', '_');
    handle_name = name.c_str();
  }
  file_mapping_ =
      CreateFileMappingA(osf_handle, /*lpFileMappingAttributes=*/nullptr,
                         /*flProtect=*/PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/handle_name);
  XNNPACK_RETURN_CHECK(file_mapping_ != NULL,
                       "could not create a file mapping: %s.",
                       GetLastErrorString().c_str());

  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);

  offset_page_adjustment_ = offset_ % sys_info.dwAllocationGranularity;

  const size_t adjusted_offset = offset - offset_page_adjustment_;
  const DWORD file_offset_high =
      sizeof(DWORD) < sizeof(adjusted_offset)
          ? (adjusted_offset >> CHAR_BIT * sizeof(DWORD))
          : 0;
  const DWORD file_offset_low = static_cast<DWORD>(adjusted_offset);

  data_ = static_cast<uint8_t*>(MapViewOfFile(file_mapping_, FILE_MAP_READ,
                                              file_offset_high, file_offset_low,
                                              /*dwNumberOfBytesToMap=*/0));

  XNNPACK_RETURN_CHECK(data_ != nullptr, "could not map file (%s): %s.",
                       safe_path, GetLastErrorString().c_str());
#else
  offset_page_adjustment_ = offset_ % getpagesize();
  data_ = static_cast<uint8_t*>(
      mmap(/*addr=*/nullptr, size_ + offset_page_adjustment_, PROT_READ,
           MAP_SHARED, fd.Value(), offset_ - offset_page_adjustment_));
  XNNPACK_RETURN_CHECK(data_ != MAP_FAILED, "could not mmap file (%s): %s.",
                       safe_path, strerror(errno));
#endif
  unmap_on_error.Deactivate();
  return true;
}

bool MMapHandle::Resize(size_t new_size) {
#if (defined(__linux__) || defined(__ANDROID__)) && \
    !defined(XNNPACK_CACHE_NO_MMAP_FOR_TEST)
  void* const remapped_data =
      mremap(data_, size_ + offset_page_adjustment_,
             new_size + offset_page_adjustment_, /*flags=*/0);
  if (remapped_data == MAP_FAILED) {
    XNNPACK_RETURN_CHECK(errno == ENOMEM, "remap failed: %s", strerror(errno));
    return false;
  }
  size_ = new_size;
  return true;
#else
  // The current implementation uses new/delete which doesn't provide a way to
  // modify an allocation size. Changing to malloc/realloc/free doesn't ensure
  // that a memory allocation will not be moved when reallocating
  return false;
#endif
}

void MMapHandle::UnMap() {
  if (data_) {
#if defined(XNNPACK_CACHE_NO_MMAP_FOR_TEST)
    delete[] data_;
#elif defined(_MSC_VER)
    UnmapViewOfFile(data_);
    CloseHandle(file_mapping_);
#else
    munmap(data_, size_ + offset_page_adjustment_);
#endif
  }
  data_ = nullptr;
  offset_ = 0;
  offset_page_adjustment_ = 0;
  size_ = 0;
}

}  // namespace tflite::xnnpack
