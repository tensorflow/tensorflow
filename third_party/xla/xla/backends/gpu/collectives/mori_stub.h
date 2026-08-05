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

#include <array>
#include <cstddef>
#include <cstdint>

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

inline void* ShmemMalloc(size_t /*size*/) { return nullptr; }

inline void ShmemFree(void* /*ptr*/) {}

}  // namespace shmem
}  // namespace mori

#endif  // XLA_BACKENDS_GPU_COLLECTIVES_MORI_STUB_H_
