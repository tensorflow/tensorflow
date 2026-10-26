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

#if defined(_WIN32)
#include <io.h>
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <sys/prctl.h>
// Added to mainline Linux UAPI <linux/prctl.h> in 5.17; older sysroots omit it.
#ifndef PR_SET_VMA
#define PR_SET_VMA 0x53564d41
#endif
#ifndef PR_SET_VMA_ANON_NAME
#define PR_SET_VMA_ANON_NAME 0
#endif
#endif

#if defined(__linux__) && !defined(MADV_PAGEOUT)
#define MADV_PAGEOUT 21
#endif

#include <fcntl.h>
#include <sys/stat.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>  // IWYU pragma: keep

#include "tensorflow/lite/delegates/xnnpack/file_util.h"
#include "tensorflow/lite/delegates/xnnpack/macros.h"
#include "tensorflow/lite/delegates/xnnpack/windows_util.h"

namespace tflite::xnnpack {

namespace {

size_t RoundUpToPageSize(size_t size) {
#if defined(_WIN32) || defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  return size == 0 ? 1 : size;
#else
  const size_t page_size = getpagesize();
  const size_t effective_size = size == 0 ? 1 : size;
  const size_t remainder = effective_size % page_size;
  return remainder == 0 ? effective_size
                        : effective_size + page_size - remainder;
#endif
}

void SetAnonVmaName(void* addr, size_t size, const char* vma_name) {
#if (defined(__linux__) || defined(__ANDROID__)) && \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (vma_name != nullptr && addr != nullptr && size > 0) {
    prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, reinterpret_cast<uintptr_t>(addr),
          size, vma_name);
  }
#else
  (void)addr;
  (void)size;
  (void)vma_name;
#endif
}

}  // namespace

// NOLINTBEGIN(whitespace/line_length)
#ifdef XNNPACK_CACHE_NO_MMAP_FOR_TEST
#pragma message( \
    "XNNPACK_CACHE_NO_MMAP_FOR_TEST has been deprecated and doesn't do anything anymore. If you were using it, consider whether filing a bug to get your usecase fixed wouldn't be a better solution. If you really need that, the new define is XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG.")
#endif

#ifdef XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG
#pragma message( \
    "XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG is defined. This can lead to highly degraded performance.")
#endif
// NOLINTEND(whitespace/line_length)

#ifdef _WIN32
// Helper to split a value in high/low parts to pass to Windows APIs.
struct HighLow {
  DWORD high;
  DWORD low;
  static HighLow From(uint64_t val) {
    static_assert(sizeof(val) <= 2 * sizeof(DWORD),
                  "Value type doesn't fit in two DWORDs.");
    return {static_cast<DWORD>(val >> CHAR_BIT * sizeof(DWORD)),
            static_cast<DWORD>(val)};
  }
};
#endif

void swap(MMapHandle& a, MMapHandle& b) {
  using std::swap;
  swap(a.size_, b.size_);
  swap(a.offset_, b.offset_);
  swap(a.offset_page_adjustment_, b.offset_page_adjustment_);
  swap(a.data_, b.data_);
  swap(a.is_locked_, b.is_locked_);
#if defined(_WIN32)
  swap(a.file_mapping_, b.file_mapping_);
#endif
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

#if defined(_WIN32)
  struct _stat64 file_stats;
  XNNPACK_RETURN_CHECK(_fstat64(fd.Value(), &file_stats) == 0,
                       "could not access file stats to get size ('%s'): %s.",
                       safe_path, strerror(errno));
#else
  struct stat file_stats;
  XNNPACK_RETURN_CHECK(
      fstat(fd.Value(), &file_stats) == 0,
      "could not access file descriptor %d stats to get size ('%s'): %s.",
      fd.Value(), safe_path, strerror(errno));
#endif

  // This will reset data_ and size_ on return until it is deactivated.
  ScopeGuard unmap_on_error([this] { UnMap(); });
  size_ = file_stats.st_size - offset;
  offset_ = offset;
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  // This allocation is freed in UnMap and in the destructor.
  data_ = new uint8_t[size_];
  fd.SetPos(offset);
  XNNPACK_RETURN_CHECK(fd.Read(data_, size_), "could not read file ('%s'): %s.",
                       safe_path, strerror(errno));
#elif defined(_WIN32)
  HANDLE osf_handle = reinterpret_cast<HANDLE>(_get_osfhandle(fd.Value()));
  XNNPACK_RETURN_CHECK(osf_handle != INVALID_HANDLE_VALUE,
                       "could not convert file descriptor to file handle: %s.",
                       strerror(errno));

  file_mapping_ =
      CreateFileMappingA(osf_handle, /*lpFileMappingAttributes=*/nullptr,
                         /*flProtect=*/PAGE_READONLY, /*dwMaximumSizeHigh=*/0,
                         /*dwMaximumSizeLow=*/0, /*lpName=*/nullptr);
  XNNPACK_RETURN_CHECK(file_mapping_ != NULL,
                       "could not create a file mapping: %s",
                       GetLastErrorString().c_str());

  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);

  offset_page_adjustment_ = offset_ % sys_info.dwAllocationGranularity;

  const size_t adjusted_offset = offset_ - offset_page_adjustment_;
  const size_t adjusted_size = size_ + offset_page_adjustment_;
  HighLow file_offset = HighLow::From(adjusted_offset);

  data_ = static_cast<uint8_t*>(MapViewOfFile(
      file_mapping_, FILE_MAP_READ, file_offset.high, file_offset.low,
      /*dwNumberOfBytesToMap=*/adjusted_size));

  XNNPACK_RETURN_CHECK(data_ != nullptr, "could not map file (%s): %s",
                       safe_path, GetLastErrorString().c_str());
#else
  offset_page_adjustment_ = offset_ % getpagesize();
  data_ = static_cast<uint8_t*>(
      mmap(/*addr=*/nullptr, size_ + offset_page_adjustment_, PROT_READ,
           MAP_SHARED, fd.Value(), offset_ - offset_page_adjustment_));
  XNNPACK_RETURN_CHECK(data_ != MAP_FAILED,
                       "could not mmap file descriptor %d (%s): %s.",
                       fd.Value(), safe_path, strerror(errno));
#endif
  unmap_on_error.Deactivate();
  return true;
}

bool MMapHandle::MapAnonymous(size_t size, const char* vma_name) {
  this->UnMap();
  const size_t mapped_size = RoundUpToPageSize(size);
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG) || defined(_WIN32)
  uint8_t* const allocated = new (std::nothrow) uint8_t[mapped_size];
  XNNPACK_RETURN_CHECK(allocated != nullptr,
                       "could not allocate anonymous buffer of size %zu.",
                       mapped_size);
  data_ = allocated;
#else
  void* const mapped =
      mmap(/*addr=*/nullptr, mapped_size, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS, /*fd=*/-1, /*offset=*/0);
  XNNPACK_RETURN_CHECK(mapped != MAP_FAILED,
                       "could not mmap anonymous buffer of size %zu: %s.",
                       mapped_size, strerror(errno));
  data_ = static_cast<uint8_t*>(mapped);
#endif
  SetAnonVmaName(data_, mapped_size, vma_name);
  size_ = mapped_size;
  return true;
}

bool MMapHandle::Resize(size_t new_size) {
#if (defined(__linux__) || defined(__ANDROID__)) && \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  void* const remapped_data =
      mremap(data_, size_ + offset_page_adjustment_,
             new_size + offset_page_adjustment_, /*flags=*/0);
  if (remapped_data == MAP_FAILED) {
    XNNPACK_RETURN_CHECK(errno == ENOMEM, "remap failed: %s", strerror(errno));
    return false;
  }
  size_ = new_size;
  is_locked_ = false;
  return true;
#else
  // The current implementation uses new/delete which doesn't provide a way to
  // modify an allocation size. Changing to malloc/realloc/free doesn't ensure
  // that a memory allocation will not be moved when reallocating
  return false;
#endif
}

bool MMapHandle::ResizeAnonymous(size_t new_size, size_t used_size,
                                 const char* vma_name) {
  XNNPACK_RETURN_CHECK(IsMapped(),
                       "cannot resize an unmapped anonymous handle.");
  const size_t mapped_new_size = RoundUpToPageSize(new_size);
  if (mapped_new_size == size_) {
    return true;
  }
#if (defined(__linux__) || defined(__ANDROID__)) && \
    !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  (void)used_size;
  void* const remapped_data =
      mremap(data_, size_, mapped_new_size, MREMAP_MAYMOVE);
  XNNPACK_RETURN_CHECK(remapped_data != MAP_FAILED,
                       "anonymous mremap failed: %s.", strerror(errno));
  data_ = static_cast<uint8_t*>(remapped_data);
  size_ = mapped_new_size;
  is_locked_ = false;
  SetAnonVmaName(data_, size_, vma_name);
  return true;
#else
  MMapHandle new_handle;
  XNNPACK_RETURN_CHECK(new_handle.MapAnonymous(mapped_new_size, vma_name));
  const size_t copy_size = std::min({used_size, size_, mapped_new_size});
  if (copy_size > 0) {
    std::memcpy(new_handle.data(), data_, copy_size);
  }
  swap(*this, new_handle);
  return true;
#endif
}

bool MMapHandle::FinalizeAnonymous(size_t used_size, const char* vma_name) {
  XNNPACK_RETURN_CHECK(IsMapped(),
                       "cannot finalize an unmapped anonymous handle.");
#if !defined(_WIN32) && !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  const size_t mapped_used_size = RoundUpToPageSize(used_size);
  if (mapped_used_size < size_) {
    XNNPACK_RETURN_CHECK(
        munmap(data_ + mapped_used_size, size_ - mapped_used_size) == 0,
        "could not unmap unused tail of anonymous buffer: %s.",
        strerror(errno));
    size_ = mapped_used_size;
  }
  XNNPACK_RETURN_CHECK(mprotect(data_, size_, PROT_READ) == 0,
                       "could not mprotect anonymous buffer to PROT_READ: %s.",
                       strerror(errno));
#else
  size_ = std::min(size_, RoundUpToPageSize(used_size));
#endif
  SetAnonVmaName(data_, size_, vma_name);
  return true;
}

void MMapHandle::UnMap() {
  if (data_) {
#if defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
    delete[] data_;
#elif defined(_WIN32)
    if (file_mapping_ != 0) {
      UnmapViewOfFile(data_);
      CloseHandle(file_mapping_);
      file_mapping_ = 0;
    } else {
      delete[] data_;
    }
#else
    munmap(data_, size_ + offset_page_adjustment_);
#endif
  }
  data_ = nullptr;
  offset_ = 0;
  offset_page_adjustment_ = 0;
  size_ = 0;
  is_locked_ = false;
}

bool MMapHandle::LockMemory() {
  if (is_locked_) {
    return true;
  }
#if defined(_WIN32) && !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (VirtualLock(data_, size_) == 0) {
    return false;
  }
#elif !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (mlock(data_, size_ + offset_page_adjustment_) != 0) {
    return false;
  }
#endif
  is_locked_ = true;
  return true;
}

bool MMapHandle::UnlockMemory() {
  if (!is_locked_) {
    return true;
  }
#if defined(_WIN32) && !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (VirtualUnlock(data_, size_) == 0) {
    return false;
  }
#elif !defined(XNNPACK_CACHE_NO_FILE_MAPPING_FOR_DEBUG)
  if (munlock(data_, size_ + offset_page_adjustment_) != 0) {
    return false;
  }
#endif
  is_locked_ = false;
  return true;
}

bool MarkMemoryNotNeeded(void* data, size_t size) {
#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  DWORD page_size = sysInfo.dwPageSize;
  // We don't mark memory chunks that are too small.
  // TODO: b/510899567 - This threshold should be re-evaluated on Windows.
  constexpr const size_t size_threshold = 1024 * 1024;
  if (size <= size_threshold) {
    return true;
  }
#else
  size_t page_size = getpagesize();
#endif

  // We align the data buffer to the next page boundary and the size to be a
  // multiple of the page size.
  //
  // - Windows will unlock all pages that contain at least a byte of the given
  // range, which we want to avoid.
  // - Linux requires the address to be on a page boundary.
  void* aligned_data = std::align(page_size, page_size, data, size);
  size -= size % page_size;
  if (aligned_data) {
#if defined(_WIN32)
    return VirtualUnlock(aligned_data, size);
#elif defined(__ANDROID__) || defined(__linux__)
    return madvise(aligned_data, size, MADV_PAGEOUT) == 0;
#endif
  }
  return true;
}

}  // namespace tflite::xnnpack
