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

#include "tensorflow/lite/experimental/litert/runtime/fastrpc_buffer.h"

#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"

namespace litert {
namespace internal {

namespace {

class FastRpcMemLibrary {
 public:
  using Ptr = std::unique_ptr<FastRpcMemLibrary>;

  static Expected<Ptr> Create() {
    DlHandle dlhandle(::dlopen("libcdsprpc.so", RTLD_NOW | RTLD_LOCAL),
                      ::dlclose);
    if (!dlhandle) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "libcdsprpc.so not found");
    }

    auto rpcmem_alloc =
        reinterpret_cast<RpcMemAlloc>(::dlsym(dlhandle.get(), "rpcmem_alloc"));
    if (!rpcmem_alloc) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "rpcmem_alloc not found");
    }

    auto rpcmem_free =
        reinterpret_cast<RpcMemFree>(::dlsym(dlhandle.get(), "rpcmem_free"));
    if (!rpcmem_free) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "rpcmem_free not found");
    }

    auto rpcmem_to_fd =
        reinterpret_cast<RpcMemToFd>(::dlsym(dlhandle.get(), "rpcmem_to_fd"));
    if (!rpcmem_to_fd) {
      return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                        "rpcmem_to_fd not found");
    }

    return Ptr(new FastRpcMemLibrary(std::move(dlhandle), rpcmem_alloc,
                                     rpcmem_free, rpcmem_to_fd));
  }

  void* Alloc(size_t size) const {
    return rpcmem_alloc_(kRpcmemHeapIdSystem, kRpcmemDefaultFlags, size);
  }

  void Free(void* buffer) const { return rpcmem_free_(buffer); }

  int ToFd(void* buffer) const { return rpcmem_to_fd_(buffer); }

 private:
  static constexpr int kRpcmemHeapIdSystem = 25;
  static constexpr uint32_t kRpcmemDefaultFlags = 1;

  using DlHandle = std::unique_ptr<void, int (*)(void*)>;
  using RpcMemAlloc = void* (*)(int, uint32_t, int);
  using RpcMemFree = void (*)(void*);
  using RpcMemToFd = int (*)(void*);

  FastRpcMemLibrary(DlHandle&& dlhandle, RpcMemAlloc rpcmem_alloc,
                    RpcMemFree rpcmem_free, RpcMemToFd rpcmem_to_fd)
      : dlhandle_(std::move(dlhandle)) {
    rpcmem_alloc_ = rpcmem_alloc;
    rpcmem_free_ = rpcmem_free;
    rpcmem_to_fd_ = rpcmem_to_fd;
  }

  DlHandle dlhandle_;
  RpcMemAlloc rpcmem_alloc_;
  RpcMemFree rpcmem_free_;
  RpcMemToFd rpcmem_to_fd_;
};

FastRpcMemLibrary* TheFastRpcMemLibrary;
ABSL_CONST_INIT absl::Mutex TheMutex(absl::kConstInit);

Expected<void> InitLibraryIfNeededUnlocked() {
  if (!TheFastRpcMemLibrary) {
    if (auto library = FastRpcMemLibrary::Create(); library) {
      TheFastRpcMemLibrary = library->release();
    } else {
      return Unexpected(library.Error());
    }
  }
  return {};
}

}  // namespace

bool FastRpcBuffer::IsSupported() {
  absl::MutexLock lock(&TheMutex);
  auto status = InitLibraryIfNeededUnlocked();
  return static_cast<bool>(status);
}

Expected<FastRpcBuffer> FastRpcBuffer::Alloc(size_t size) {
  absl::MutexLock lock(&TheMutex);
  if (auto status = InitLibraryIfNeededUnlocked(); !status) {
    return status.Error();
  }
  void* addr = TheFastRpcMemLibrary->Alloc(size);
  int fd = TheFastRpcMemLibrary->ToFd(addr);
  return FastRpcBuffer{.fd = fd, .addr = addr};
}

void FastRpcBuffer::Free(void* addr) {
  absl::MutexLock lock(&TheMutex);
  if (TheFastRpcMemLibrary) {
    TheFastRpcMemLibrary->Free(addr);
  }
}

}  // namespace internal
}  // namespace litert
