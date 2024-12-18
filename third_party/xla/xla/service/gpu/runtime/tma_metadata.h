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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace xla {
namespace gpu {

// Tensor Memory Accelerator (TMA) is a CUDA-specific feature that allows for
// a more sophisticated asynchronous copy. See documentation here:
// https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html.
// This feature is available on Hopper and later architectures.
//
// This feature depends on information getting passed from compile time to the
// runtime.
//
// TmaDescriptor holds all information necessary about a tensor from compile
// time to create a CUDA TensorMap object at runtime.
//
// This class intentionally does not depend on CUDA headers or types to avoid
// pulling in CUDA dependencies as some of the code using it is shared with
// non-CUDA platforms.
class TmaDescriptor {
 public:
  // The following enums are a mirror to the types described in NVIDIA's
  // documentation here:
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html

  // Type of interleaved layout the tensor addresses.
  typedef enum {
    TMA_INTERLEAVE_NONE = 0,
    TMA_INTERLEAVE_16B,
    TMA_INTERLEAVE_32B
  } tma_interleave;

  // Bank swizzling pattern inside shared memory.
  typedef enum {
    TMA_SWIZZLE_NONE = 0,
    TMA_SWIZZLE_32B,
    TMA_SWIZZLE_64B,
    TMA_SWIZZLE_128B
  } tma_swizzle;

  // L2 promotion size.
  typedef enum {
    TMA_L2_PROMOTION_NONE = 0,
    TMA_L2_PROMOTION_64B,
    TMA_L2_PROMOTION_128B,
    TMA_L2_PROMOTION_256B
  } tma_l2_promotion;

  // Indicate whether zero or special NaN constant must be used to fill
  // out-of-bound elements.
  typedef enum {
    TMA_FLOAT_OOB_FILL_NONE = 0,
    TMA_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
  } tma_float_oob_fill;

  // Constructs TmaDescriptor to be used at runtime.
  static absl::StatusOr<TmaDescriptor> Create(
      llvm::ArrayRef<int64_t> global_dims,
      llvm::ArrayRef<int64_t> global_strides, llvm::ArrayRef<int32_t> box_dims,
      llvm::ArrayRef<int32_t> element_strides, int element_byte_width,
      tma_interleave interleave = TMA_INTERLEAVE_NONE,
      tma_swizzle swizzle = TMA_SWIZZLE_NONE,
      tma_l2_promotion l2_promotion = TMA_L2_PROMOTION_NONE,
      tma_float_oob_fill float_oob_fill = TMA_FLOAT_OOB_FILL_NONE);

  // Returns a string with all TmaDescriptor fields within.
  std::string ToString() const;

  // Getters.
  int element_size() const { return element_size_; }
  uint32_t rank() const { return rank_; }
  llvm::ArrayRef<uint64_t> global_dims() const { return global_dims_; }
  llvm::ArrayRef<uint64_t> global_strides() const { return global_strides_; }
  llvm::ArrayRef<uint32_t> box_dims() const { return box_dims_; }
  llvm::ArrayRef<uint32_t> element_strides() const { return element_strides_; }
  tma_interleave interleave() const { return interleave_; }
  tma_swizzle swizzle() const { return swizzle_; }
  tma_l2_promotion l2_promotion() const { return l2_promotion_; }
  tma_float_oob_fill float_oob_fill() const { return float_oob_fill_; }

 private:
  TmaDescriptor(llvm::ArrayRef<int64_t> global_dims,
                llvm::ArrayRef<int64_t> global_strides,
                llvm::ArrayRef<int32_t> box_dims,
                llvm::ArrayRef<int32_t> element_strides, int element_size,
                tma_interleave interleave, tma_swizzle swizzle,
                tma_l2_promotion l2_promotion,
                tma_float_oob_fill float_oob_fill);

  // Element size in bytes of the tensor. Can be 1, 2, 4, 8.
  int element_size_;
  // Rank of the tensor. Can be 1-5.
  uint32_t rank_;
  // Array containing tensor size (number of elements) along each of the rank_
  // dimensions.
  llvm::SmallVector<uint64_t> global_dims_;
  // Array containing stride size (in bytes) along each of the rank_ dimensions.
  llvm::SmallVector<uint64_t> global_strides_;
  // Array containing traversal box size (number of elements) along each of the
  // rank_ dimensions. Specifies how many elements to be traversed along each
  // tensor dimension. Can be max 256.
  llvm::SmallVector<uint32_t> box_dims_;
  // Array containing traversal stride in each of the rank_ dimensions.
  llvm::SmallVector<uint32_t> element_strides_;
  // Type of interleaved layout the tensor addresses.
  tma_interleave interleave_;
  // Bank swizzling pattern inside shared memory.
  tma_swizzle swizzle_;
  // L2 promotion size.
  tma_l2_promotion l2_promotion_;
  // Indicate whether zero or special NaN constant must be used to fill
  // out-of-bound elements.
  tma_float_oob_fill float_oob_fill_;
};

struct TmaMetadata {
  // Maps the index of the kernel argument to the corresponding TmaDescriptor to
  // be used at runtime.
  absl::flat_hash_map<int64_t, TmaDescriptor> arg_index_to_tma_info = {};
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_RUNTIME_TMA_METADATA_H_
