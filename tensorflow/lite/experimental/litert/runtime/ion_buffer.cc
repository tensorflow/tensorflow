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

#include "tensorflow/lite/experimental/litert/runtime/ion_buffer.h"

#include <dlfcn.h>
#include <sys/mman.h>

#include <cstddef>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace internal {

namespace {

class IonLibrary {
 public:
  using Ptr = std::unique_ptr<IonLibrary>;

  ~IonLibrary() {
    if (client_fd_ > 0) {
      ion_close_(client_fd_);
    }
  }

  static Expected<Ptr> Create() {
    DlHandle dlhandle(::dlopen("libion.so", RTLD_NOW | RTLD_LOCAL), ::dlclose);
    if (!dlhandle) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "libion.so not found");
    }

    auto ion_open =
        reinterpret_cast<IonOpen>(::dlsym(dlhandle.get(), "ion_open"));
    if (!ion_open) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure, "ion_open not found");
    }

    auto ion_close =
        reinterpret_cast<IonClose>(::dlsym(dlhandle.get(), "ion_close"));
    if (!ion_close) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "ion_close not found");
    }

    auto ion_alloc_fd =
        reinterpret_cast<IonAllocFd>(::dlsym(dlhandle.get(), "ion_alloc_fd"));
    if (!ion_alloc_fd) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "ion_alloc_fd not found");
    }

    int client_fd = ion_open();
    if (client_fd < 0) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "Failed to open ion device");
    }

    return Ptr(new IonLibrary(std::move(dlhandle), client_fd, ion_close,
                              ion_alloc_fd));
  }

  Expected<IonBuffer> Alloc(size_t size, size_t alignment) {
    int heap_id_mask = 1 << kIonHeapId;
    int fd;
    if (auto status = ion_alloc_fd_(client_fd_, size, alignment, heap_id_mask,
                                    kIonFlags, &fd);
        status != 0) {
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
    return IonBuffer{.fd = fd, .addr = addr};
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
  static constexpr const int kIonHeapId = 25;
  static constexpr const int kIonFlags = 1;

  struct Record {
    int fd;
    void* addr;
    size_t size;
  };

  using DlHandle = std::unique_ptr<void, int (*)(void*)>;
  using IonOpen = int (*)();
  using IonClose = int (*)(int);
  using IonAllocFd = int (*)(int, size_t, size_t, unsigned int, unsigned int,
                             int*);

  IonLibrary(DlHandle&& dlhandle, int client_fd, IonClose ion_close,
             IonAllocFd ion_alloc_fd)
      : dlhandle_(std::move(dlhandle)),
        client_fd_(client_fd),
        ion_close_(ion_close),
        ion_alloc_fd_(ion_alloc_fd) {}

  DlHandle dlhandle_;
  int client_fd_;
  IonClose ion_close_;
  IonAllocFd ion_alloc_fd_;
  absl::node_hash_map<void*, Record> records_;
};

IonLibrary* TheIonLibrary;
ABSL_CONST_INIT absl::Mutex TheMutex(absl::kConstInit);

Expected<void> InitLibraryIfNeededUnlocked() {
  if (!TheIonLibrary) {
    if (auto library = IonLibrary::Create(); library) {
      TheIonLibrary = library->release();
    } else {
      return Unexpected(library.Error());
    }
  }
  return {};
}

}  // namespace

bool IonBuffer::IsSupported() {
  absl::MutexLock lock(&TheMutex);
  auto status = InitLibraryIfNeededUnlocked();
  return static_cast<bool>(status);
}

Expected<IonBuffer> IonBuffer::Alloc(size_t size, size_t alignment) {
  absl::MutexLock lock(&TheMutex);
  if (auto status = InitLibraryIfNeededUnlocked(); !status) {
    return status.Error();
  }
  return TheIonLibrary->Alloc(size, alignment);
}

void IonBuffer::Free(void* addr) {
  absl::MutexLock lock(&TheMutex);
  if (TheIonLibrary) {
    TheIonLibrary->Free(addr);
  }
}

}  // namespace internal
}  // namespace litert
