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
SelectAlgo choose_select_k_algorithm<float>(uint32_t rows, uint32_t cols,
                                            uint32_t k) {
  if (k > 163) {
    if (rows > 803) {
      if (k > 256) {
        return SelectAlgo::kRadix8bits;
      } else {
        if (cols > 3120) {
          return SelectAlgo::kWarpDistributedShm;
        } else {
          return SelectAlgo::kWarpFiltered;
        }
      }
    } else {
      if (cols > 19332) {
        if (cols > 106382976) {
          return SelectAlgo::kRadix11bitsExtraPass;
        } else {
          if (k > 256) {
            return SelectAlgo::kRadix11bits;
          } else {
            if (rows > 384) {
              return SelectAlgo::kWarpDistributedShm;
            } else {
              if (cols > 110946) {
                return SelectAlgo::kRadix11bits;
              } else {
                return SelectAlgo::kWarpFiltered;
              }
            }
          }
        }
      } else {
        if (k > 256) {
          return SelectAlgo::kRadix8bits;
        } else {
          return SelectAlgo::kWarpFiltered;
        }
      }
    }
  } else {
    if (k > 1) {
      if (cols > 34520) {
        if (k > 73) {
          if (rows > 132) {
            return SelectAlgo::kWarpDistributedShm;
          } else {
            if (cols > 259191) {
              if (cols > 12021888) {
                if (rows > 2) {
                  return SelectAlgo::kWarpDistributedShm;
                } else {
                  return SelectAlgo::kRadix11bits;
                }
              } else {
                if (rows > 36) {
                  return SelectAlgo::kWarpDistributedShm;
                } else {
                  return SelectAlgo::kRadix11bits;
                }
              }
            } else {
              if (rows > 48) {
                return SelectAlgo::kRadix11bits;
              } else {
                return SelectAlgo::kWarpFiltered;
              }
            }
          }
        } else {
          if (k > 4) {
            return SelectAlgo::kWarpDistributedShm;
          } else {
            if (rows > 18) {
              return SelectAlgo::kWarpDistributedShm;
            } else {
              if (cols > 1652995) {
                return SelectAlgo::kWarpDistributedShm;
              } else {
                return SelectAlgo::kWarpImmediate;
              }
            }
          }
        }
      } else {
        if (rows > 363) {
          if (cols > 1115) {
            return SelectAlgo::kWarpDistributedShm;
          } else {
            if (rows > 1536) {
              return SelectAlgo::kWarpDistributedShm;
            } else {
              return SelectAlgo::kWarpImmediate;
            }
          }
        } else {
          if (k > 85) {
            if (cols > 1034) {
              return SelectAlgo::kWarpFiltered;
            } else {
              return SelectAlgo::kWarpImmediate;
            }
          } else {
            if (rows > 81) {
              if (k > 4) {
                if (cols > 12288) {
                  return SelectAlgo::kWarpDistributedShm;
                } else {
                  return SelectAlgo::kWarpImmediate;
                }
              } else {
                return SelectAlgo::kWarpImmediate;
              }
            } else {
              return SelectAlgo::kWarpImmediate;
            }
          }
        }
      }
    } else {
      return SelectAlgo::kWarpImmediate;
    }
  }
}

}  // namespace raft_internal

template absl::Status select_k_exec<float>(int, se::DeviceAddressAllocator*,
                                           se::Stream*, se::DeviceAddressBase,
                                           se::DeviceAddressBase,
                                           se::DeviceAddressBase, std::uint32_t,
                                           std::uint32_t, std::uint32_t);

}  // namespace xla::gpu
