/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_
#define XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

// Inert stand-in for the subset of the MORI shmem host API used by the MORI
// collectives/communicator backbone. These placeholders let the backbone
// compile and link without depending on the MORI library. All operations are
// no-ops. Replace by including the real "mori/shmem/shmem_api.hpp" once the
// MORI bindings are wired up.

#define MORI_SHMEM_UNIQUE_ID_BYTES 128

namespace mori {
namespace shmem {

using mori_shmem_uniqueid_t = std::array<uint8_t, MORI_SHMEM_UNIQUE_ID_BYTES>;

struct mori_shmem_init_attr_t {
  int32_t rank;
  int32_t nranks;
  mori_shmem_uniqueid_t uid;
  void* mpi_comm;  // Optional MPI_Comm pointer.
};

// Initialization flags.
[[maybe_unused]] constexpr unsigned int MORI_SHMEM_INIT_WITH_MPI_COMM = 0;
[[maybe_unused]] constexpr unsigned int MORI_SHMEM_INIT_WITH_UNIQUEID = 1;

inline int ShmemGetUniqueId(mori_shmem_uniqueid_t* /*uid*/) { return 0; }

inline int ShmemSetAttrUniqueIdArgs(int /*rank*/, int /*nranks*/,
                                    mori_shmem_uniqueid_t* /*uid*/,
                                    mori_shmem_init_attr_t* /*attr*/) {
  return 0;
}

inline int ShmemInitAttr(unsigned int /*flags*/,
                         mori_shmem_init_attr_t* /*attr*/) {
  return 0;
}

inline int ShmemFinalize() { return 0; }

inline int ShmemMyPe() { return 0; }

inline int ShmemNPes() { return 0; }

inline void* ShmemMalloc(size_t /*size*/) {
  return reinterpret_cast<void*>(0xBABEFEEDDEADBULL);
}

inline void ShmemFree(void* /*ptr*/) {}

}  // namespace shmem
}  // namespace mori

namespace mori {
namespace collective {

// Element type + reduction op enums mirror the real facade's non-templated API
// (mori/collective/collectives_facade.hpp), so the communicator's enum dispatch
// compiles against either the stub or the real facade.
enum class DataType {
  F8E5M2,
  F8E4M3FN,
  F16,
  BF16,
  S8,
  U8,
  S32,
  U32,
  S64,
  U64,
  F32,
  F64
};
enum class ReduceOpKind { SUM, PRODUCT, MIN, MAX };

// Inert stand-in for the real MORI CollectivesFacade. Header-only, all Run* are
// no-ops returning hipSuccess. Lets the collectives/communicator wiring compile
// and link without @roc_mori.
class CollectivesFacade {
  CollectivesFacade() = default;

 public:
  using AddressVector = std::vector<std::pair<const void*, void*>>;

  CollectivesFacade(const CollectivesFacade&) = delete;
  CollectivesFacade& operator=(const CollectivesFacade&) = delete;

  static std::unique_ptr<CollectivesFacade> Create(int /*myPe*/, int /*nPes*/,
                                                   size_t /*maxStagingBytes*/) {
    return std::unique_ptr<CollectivesFacade>(new CollectivesFacade());
  }
  ~CollectivesFacade() = default;

  hipError_t RunReduceScatter(const void*, void*, size_t, DataType,
                              ReduceOpKind, hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunAllReduce(const void*, void*, size_t, DataType, ReduceOpKind,
                          hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunAllGather(const void*, void*, size_t, hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunAllToAll(const AddressVector&, size_t, hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunBarrier(hipStream_t) { return hipSuccess; }
  hipError_t RunSend(const void*, size_t, int, hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunRecv(void*, size_t, int, hipStream_t) { return hipSuccess; }
  hipError_t RunCollectivePermute(const void*, void*, size_t, int,
                                  const std::vector<int>&, hipStream_t) {
    return hipSuccess;
  }
  hipError_t RunQuiet(hipStream_t) { return hipSuccess; }
  hipError_t RunFence() { return hipSuccess; }
};

}  // namespace collective
}  // namespace mori

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_
