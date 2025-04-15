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

#ifndef XLA_STREAM_EXECUTOR_GPU_TMA_METADATA_H_
#define XLA_STREAM_EXECUTOR_GPU_TMA_METADATA_H_

#include <stdint.h>

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace stream_executor {
namespace gpu {

// Tensor Memory Accelerator (TMA) is a CUDA-specific feature that allows for
// a more sophisticated asynchronous copy. See documentation here:
// https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html.
// This feature is available on Hopper and later architectures.
// The current restrictions are as documented for Hopper specifically. These may
// change in future architectures.
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
  enum class TmaInterleave {
    kNone = 0,
    k16B,
    k32B,
  };

  // Bank swizzling pattern inside shared memory.
  enum class TmaSwizzle {
    kNone = 0,
    k32B,
    k64B,
    k128B,
  };

  // L2 promotion size.
  enum class TmaL2Promotion {
    kNone = 0,
    k64B,
    k128B,
    k256B,
  };

  // Indicates whether zero or special NaN constant must be used to fill
  // out-of-bound elements.
  enum class TmaFloatOobFill {
    kNone = 0,
    kNanRequestZeroFma,
  };

  // Constructs TmaDescriptor to be used at runtime.
  static absl::StatusOr<TmaDescriptor> Create(
      llvm::ArrayRef<uint64_t> global_dims,
      llvm::ArrayRef<uint64_t> global_strides,
      llvm::ArrayRef<uint32_t> box_dims,
      llvm::ArrayRef<uint32_t> element_strides, int element_byte_width,
      TmaInterleave interleave = TmaInterleave::kNone,
      TmaSwizzle swizzle = TmaSwizzle::kNone,
      TmaL2Promotion l2_promotion = TmaL2Promotion::kNone,
      TmaFloatOobFill float_oob_fill = TmaFloatOobFill::kNone);

  // Returns a string with all TmaDescriptor fields within.
  std::string ToString() const;

  // Element size in bytes of the tensor. Can be 1, 2, 4, 8.
  int element_size() const { return element_size_; }

  // Number of dimensions of the tensor. Can be 1-5.
  uint32_t num_dimensions() const { return num_dimensions_; }

  // Array containing tensor size (number of elements) along each of the rank
  // dimensions.
  llvm::ArrayRef<uint64_t> global_dims() const { return global_dims_; }

  // Array containing stride size (in bytes) along each of the rank dimensions.
  llvm::ArrayRef<uint64_t> global_strides() const { return global_strides_; }

  // Array containing traversal box size (number of elements) along each of the
  // rank dimensions. Specifies how many elements to be traversed along each
  // tensor dimension. Can be max 256.
  llvm::ArrayRef<uint32_t> box_dims() const { return box_dims_; }

  // Array containing traversal stride in each of the rank dimensions.
  llvm::ArrayRef<uint32_t> element_strides() const { return element_strides_; }

  // Type of interleaved layout the tensor addresses.
  TmaInterleave interleave() const { return interleave_; }

  // Bank swizzling pattern inside shared memory.
  TmaSwizzle swizzle() const { return swizzle_; }

  // L2 promotion size.
  TmaL2Promotion l2_promotion() const { return l2_promotion_; }

  // Indicate whether zero or special NaN constant must be used to fill
  // out-of-bound elements.
  TmaFloatOobFill float_oob_fill() const { return float_oob_fill_; }

 private:
  TmaDescriptor(llvm::ArrayRef<uint64_t> global_dims,
                llvm::ArrayRef<uint64_t> global_strides,
                llvm::ArrayRef<uint32_t> box_dims,
                llvm::ArrayRef<uint32_t> element_strides, int element_size,
                TmaInterleave interleave, TmaSwizzle swizzle,
                TmaL2Promotion l2_promotion, TmaFloatOobFill float_oob_fill);

  // Element size in bytes of the tensor. Can be 1, 2, 4, 8.
  int element_size_;

  // Number of dimensions of the tensor. Can be 1-5.
  uint32_t num_dimensions_;

  // Array containing tensor size (number of elements) along each of the rank
  // dimensions.
  llvm::SmallVector<uint64_t> global_dims_;

  // Array containing stride size (in bytes) along each of the rank - 1
  // dimensions.
  llvm::SmallVector<uint64_t> global_strides_;

  // Array containing traversal box size (number of elements) along each of the
  // rank dimensions. Specifies how many elements to be traversed along each
  // tensor dimension. Can be max 256.
  llvm::SmallVector<uint32_t> box_dims_;

  // Array containing traversal stride in each of the rank dimensions.
  llvm::SmallVector<uint32_t> element_strides_;

  // Type of interleaved layout the tensor addresses.
  TmaInterleave interleave_;

  // Bank swizzling pattern inside shared memory.
  TmaSwizzle swizzle_;

  // L2 promotion size.
  TmaL2Promotion l2_promotion_;

  // Indicate whether zero or special NaN constant must be used to fill
  // out-of-bound elements.
  TmaFloatOobFill float_oob_fill_;
};

struct TmaMetadata {
  // Maps the index of the kernel argument to the corresponding TmaDescriptor to
  // be used at runtime.
  absl::flat_hash_map<int64_t, TmaDescriptor> arg_index_to_tma_info = {};
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_TMA_METADATA_H_
