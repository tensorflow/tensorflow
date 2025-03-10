// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/runtime/dmabuf_buffer.h"

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

#if LITERT_HAS_DMABUF_SUPPORT
#include <dlfcn.h>
#include <sys/mman.h>
#endif  // LITERT_HAS_DMABUF_SUPPORT

namespace litert {
namespace internal {

#if LITERT_HAS_DMABUF_SUPPORT
namespace {

class DmaBufLibrary {
 public:
  using Ptr = std::unique_ptr<DmaBufLibrary>;

  ~DmaBufLibrary() {
    if (allocator_) {
      free_allocator_(allocator_);
    }
  }

  static Expected<Ptr> Create() {
    DlHandle dlhandle(::dlopen("libdmabufheap.so", RTLD_LAZY | RTLD_LOCAL),
                      ::dlclose);
    if (!dlhandle) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "libdmabufheap.so not found");
    }

    auto create_allocator = reinterpret_cast<CreateAllocator>(
        ::dlsym(dlhandle.get(), "CreateDmabufHeapBufferAllocator"));
    if (!create_allocator) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "CreateDmabufHeapBufferAllocator not found");
    }

    auto free_allocator = reinterpret_cast<FreeAllocator>(
        ::dlsym(dlhandle.get(), "FreeDmabufHeapBufferAllocator"));
    if (!free_allocator) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "FreeDmabufHeapBufferAllocator not found");
    }

    auto alloc_buffer = reinterpret_cast<AllocBuffer>(
        ::dlsym(dlhandle.get(), "DmabufHeapAlloc"));
    if (!alloc_buffer) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "DmabufHeapAlloc not found");
    }

    void* allocator = create_allocator();
    if (!allocator) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "CreateDmabufHeapBufferAllocator failed");
    }

    return Ptr(new DmaBufLibrary(std::move(dlhandle), allocator, free_allocator,
                                 alloc_buffer));
  }

  Expected<DmaBufBuffer> Alloc(size_t size) {
    int fd = alloc_buffer_(allocator_, kDmaBufHeap, size, /*flags=*/0,
                           /*legacy_align=*/0);
    if (fd < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to allocate DMA-BUF buffer");
    }
    void* addr =
        ::mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to mem-map DMA-BUF buffer");
    }
    records_[addr] = Record{.fd = fd, .addr = addr, .size = size};
    return DmaBufBuffer{.fd = fd, .addr = addr};
  }

  void Free(void* addr) {
    auto iter = records_.find(addr);
    if (iter == records_.end()) {
      return;
    }
    auto& record = iter->second;
    ::munmap(record.addr, record.size);
    ::close(record.fd);
    records_.erase(iter);
  }

 private:
  static constexpr const char* kDmaBufHeap = "system";

  struct Record {
    int fd;
    void* addr;
    size_t size;
  };

  using DlHandle = std::unique_ptr<void, int (*)(void*)>;
  using CreateAllocator = void* (*)();
  using FreeAllocator = void (*)(void*);
  using AllocBuffer = int (*)(void*, const char*, size_t, unsigned int, size_t);

  DmaBufLibrary(DlHandle&& dlhandle, void* allocator,
                FreeAllocator free_allocator, AllocBuffer alloc_buffer)
      : dlhandle_(std::move(dlhandle)) {
    allocator_ = allocator;
    free_allocator_ = free_allocator;
    alloc_buffer_ = alloc_buffer;
  }

  DlHandle dlhandle_;
  void* allocator_;
  FreeAllocator free_allocator_;
  AllocBuffer alloc_buffer_;
  absl::node_hash_map<void*, Record> records_;
};

DmaBufLibrary* TheDmaBufLibrary;
ABSL_CONST_INIT absl::Mutex TheMutex(absl::kConstInit);

Expected<void> InitLibraryIfNeededUnlocked() {
  if (!TheDmaBufLibrary) {
    if (auto library = DmaBufLibrary::Create(); library) {
      TheDmaBufLibrary = library->release();
    } else {
      return Unexpected(library.Error());
    }
  }
  return {};
}

}  // namespace
#endif  // LITERT_HAS_DMABUF_SUPPORT

bool DmaBufBuffer::IsSupported() {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  auto status = InitLibraryIfNeededUnlocked();
  return static_cast<bool>(status);
#else   // LITERT_HAS_DMABUF_SUPPORT
  return false;
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

Expected<DmaBufBuffer> DmaBufBuffer::Alloc(size_t size) {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  if (auto status = InitLibraryIfNeededUnlocked(); !status) {
    return Unexpected(status.Error());
  }
  return TheDmaBufLibrary->Alloc(size);
#else   // LITERT_HAS_DMABUF_SUPPORT
  return Unexpected(kLiteRtStatusErrorUnsupported,
                    "DmaBufBuffer::Alloc not implemented for this platform");
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

void DmaBufBuffer::Free(void* addr) {
#if LITERT_HAS_DMABUF_SUPPORT
  absl::MutexLock lock(&TheMutex);
  if (TheDmaBufLibrary) {
    TheDmaBufLibrary->Free(addr);
  }
#endif  // LITERT_HAS_DMABUF_SUPPORT
}

}  // namespace internal
}  // namespace litert
