/* Copyright 2025 The OpenXLA Authors.

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

#include <cstdint>

#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/select_k_exec_raft_impl.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace xla::gpu {
namespace raft_internal {

// ============================================================================
// choose_select_k_algorithm
//
// Purpose:
//   Heuristic-based selection of the optimal "select k" algorithm depending on
//   problem shape (rows, cols, k). The decision is based on benchmark data.
//
// How the heuristic is generated:
//
//   1. Build the benchmark module:
//        cd raft
//        ./build.sh libraft bench-prims
//
//   2. Collect performance data by running microbenchmarks:
//
//        From the RAFT project root:
//          ./cpp/build/bench/prims/MATRIX_BENCH \
//            --benchmark_filter=Select \
//            --benchmark_out_format=json \
//            --benchmark_out=select_k_times.json
//
//        Output:
//          - Benchmark results are written to `select_k_times.json`
//
//   3. Generate the heuristic using the provided notebook:
//
//        ./cpp/scripts/heuristics/select_k/generate_heuristic.ipynb
//
//        The notebook consumes `select_k_times.json`, analyzes performance
//        trade-offs, and produces the decision tree implemented here.
//
// Notes:
//   - To generate performance data for BFloat16,
//     modify cpp/bench/prims/matrix/select_k.cu  and register nv_bfloat16 type
//     using SELECTION_REGISTER macros.
// ============================================================================

template <>
SelectAlgo choose_select_k_algorithm<nv_bfloat16>(uint32_t rows, uint32_t cols,
                                                  uint32_t k) {
  if (k > 40) {
    if (cols > 24576) {
      if (rows > 768) {
        if (k > 192) {
          return SelectAlgo::kRadix8bits;
        } else {
          if (cols > 196608) {
            if (k > 80) {
              return SelectAlgo::kRadix11bits;
            } else {
              if (rows > 1536) {
                return SelectAlgo::kWarpDistributed;
              } else {
                return SelectAlgo::kWarpDistributedShm;
              }
            }
          } else {
            return SelectAlgo::kWarpDistributedShm;
          }
        }
      } else {
        if (k > 192) {
          if (cols > 393216) {
            if (cols > 6291456) {
              return SelectAlgo::kRadix11bitsExtraPass;
            } else {
              if (k > 768) {
                if (cols > 3145728) {
                  return SelectAlgo::kRadix11bits;
                } else {
                  return SelectAlgo::kRadix11bitsExtraPass;
                }
              } else {
                return SelectAlgo::kRadix11bits;
              }
            }
          } else {
            return SelectAlgo::kRadix11bitsExtraPass;
          }
        } else {
          if (cols > 6291456) {
            if (k > 80) {
              return SelectAlgo::kRadix11bitsExtraPass;
            } else {
              if (cols > 25165824) {
                if (rows > 6) {
                  return SelectAlgo::kWarpDistributed;
                } else {
                  return SelectAlgo::kRadix11bitsExtraPass;
                }
              } else {
                return SelectAlgo::kRadix11bitsExtraPass;
              }
            }
          } else {
            if (cols > 98304) {
              return SelectAlgo::kRadix11bits;
            } else {
              if (rows > 48) {
                return SelectAlgo::kRadix11bitsExtraPass;
              } else {
                return SelectAlgo::kWarpFiltered;
              }
            }
          }
        }
      }
    } else {
      if (k > 256) {
        return SelectAlgo::kRadix8bits;
      } else {
        if (rows > 768) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          if (rows > 48) {
            if (cols > 6144) {
              return SelectAlgo::kRadix8bits;
            } else {
              return SelectAlgo::kWarpFiltered;
            }
          } else {
            if (cols > 6144) {
              return SelectAlgo::kWarpFiltered;
            } else {
              if (k > 80) {
                return SelectAlgo::kWarpImmediate;
              } else {
                return SelectAlgo::kWarpFiltered;
              }
            }
          }
        }
      }
    }
  } else {
    if (k > 1) {
      if (cols > 24576) {
        if (cols > 98304) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          if (rows > 48) {
            return SelectAlgo::kWarpDistributedShm;
          } else {
            if (k > 24) {
              return SelectAlgo::kWarpDistributedShm;
            } else {
              return SelectAlgo::kWarpImmediate;
            }
          }
        }
      } else {
        if (rows > 384) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          return SelectAlgo::kWarpImmediate;
        }
      }
    } else {
      return SelectAlgo::kWarpImmediate;
    }
  }
}

}  // namespace raft_internal

template absl::Status select_k_exec<nv_bfloat16>(
    int, se::DeviceAddressAllocator*, se::Stream*, se::DeviceAddressBase,
    se::DeviceAddressBase, se::DeviceAddressBase, std::uint32_t, std::uint32_t,
    std::uint32_t);

template <>
absl::Status select_k_exec<::xla::bfloat16>(
    int device_ordinal, se::DeviceAddressAllocator* allocator,
    se::Stream* stream, se::DeviceAddressBase data_in,
    se::DeviceAddressBase data_out, se::DeviceAddressBase indices_out,
    std::uint32_t batch, std::uint32_t n, std::uint32_t k) {
  static_assert(sizeof(::xla::bfloat16) == sizeof(nv_bfloat16),
                "xla::bfloat16 and nv_bfloat16 must have the same size");

  return select_k_exec<nv_bfloat16>(device_ordinal, allocator, stream, data_in,
                                    data_out, indices_out, batch, n, k);
}

}  // namespace xla::gpu
