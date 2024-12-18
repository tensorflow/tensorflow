/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_RUNTIME_TMA_METADATA_H_
#define XLA_SERVICE_GPU_RUNTIME_TMA_METADATA_H_

#include <stdint.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Types.h"

namespace xla {
namespace gpu {

class TmaInfo {
 public:
  // Constructs TmaInfo to be used at runtime.
  static absl::StatusOr<TmaInfo> Create(llvm::ArrayRef<int64_t> tensor_shape,
                                        absl::Span<const int64_t> global_dims,
                                        mlir::Type element_type);

  // Calls cuTensorMapEncodeTiled with the appropriate parameters.
  CUresult CreateTensorMap(CUtensorMap* tensor_map, void* global_address);

  // Returns a string with all TmaInfo fields within.
  std::string ToString();

 private:
  TmaInfo(llvm::ArrayRef<int64_t> tensor_shape, absl::Span<const int64_t> dims,
          int element_size);
  // The naming of the variables correspond to the naming in the
  // cuTensorMapEncodeTiled API here:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  // Variables below must be set by the constructor.
  uint32_t tensor_dims_[2];
  uint64_t global_dims_[2];
  cuuint64_t global_strides_[2];
  CUtensorMapDataType data_type_;
  CUtensorMapSwizzle swizzle_;

  // These have valid defaults.
  uint32_t rank_ = 2;
  cuuint32_t element_strides_[2] = {1, 1};
  CUtensorMapInterleave interleave_ = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapL2promotion l2_promotion_ = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  CUtensorMapFloatOOBfill float_oob_fill_ = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
};

struct TmaMetadata {
  // Maps the index of the kernel argument to the corresponding TmaInfo to be
  // used at runtime.
  absl::flat_hash_map<int64_t, TmaInfo> arg_index_to_tma_info = {};
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_TMA_METADATA_H_
