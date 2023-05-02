/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_SHARDING_PARAM_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_SHARDING_PARAM_H_

#include <cstdint>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace xla {
namespace ifrt {

// Represents the sharding of an array in IFRT IR.
//
// The assembly format is
//   $dim_shards to $permutation on $axis_sizes
//
// `dim_shards` has rank matching the tensor. Its sizes tell how to distribute
// the corresponding dimensions of the tensor to the mesh axes. The `dim_shards`
// then will be mapped to the `permutation` of axes in `minor_to_major`,
// uniquely determining the slice of tensor on each logical device. For example:
//
// 2x1x3 to [1,0] on 3x2
//   means to shard a rank-3 tensor into 2 slices in dim-0 and 3 slices in
//   dim-2. The 6 slices will be distributed to 6 logical devices in the order
//   of 0,3,1,4,2,5.
//
// 2x1 to [0,1] on 2x3
//   means to shard a rank-2 tensor into 2 slices in dim-0. The 2 slices will
//   be distributed to 2 groups replicated on the 3 devices in each group. The
//   groups of logical devices are (0,1,2), (3,4,5).
//
// 4 to [1,0] on 2x2
//   means to shard a rank-1 tensor into 4 slices. The 4 slices will be
//   distributed to 4 logical devices in the order of 0,2,1,3.
//
// 1x1 to [0,1] on 2
//   is invalid, because `permutation` and `axis_sizes` has different sizes.
//
// 2x2 to [0] on 2
//   is invalid, because the 4 slices can't be distributed to 2 devices.
//
// 1x2 to [0,1] on 3x2
//   is invalid, because the 2 slices on dim-1 can't be distributed to 3 devices
//   in axis-0.
//
// See `support` directory for conversions with other sharding annotations.
//
// TODO(b/271129892): Should we support maximal sharding here?
class ShardingParam {
 public:
  // Represents a permutation of mesh dimensions from minor to major.
  //
  // Sizes of `permutation` and `sizes` must be equal.
  struct MinorToMajor {
    // A permutation of range [0...n].
    llvm::SmallVector<int64_t, 4> permutation;
    // The size of mesh dimensions before the permutation.
    llvm::SmallVector<int64_t, 4> axis_sizes;

    mlir::LogicalResult verify(
        llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const;

    bool operator==(const MinorToMajor& other) const {
      return permutation == other.permutation && axis_sizes == other.axis_sizes;
    }

    // Produces a flat list of device ids according to the permutation.
    void ToDeviceList(llvm::SmallVectorImpl<int64_t>& out_devices) const;
  };

  ShardingParam(llvm::ArrayRef<int64_t> dim_shards, MinorToMajor minor_to_major)
      : dim_shards_(dim_shards), minor_to_major_(minor_to_major) {}

  static mlir::FailureOr<ShardingParam> Parse(mlir::AsmParser& ods_parser);
  mlir::LogicalResult verify(
      llvm::function_ref<mlir::InFlightDiagnostic()> emit_error) const;

  llvm::ArrayRef<int64_t> dim_shards() const { return dim_shards_; }
  const MinorToMajor& minor_to_major() const { return minor_to_major_; }

  bool operator==(const ShardingParam& other) const {
    return dim_shards_ == other.dim_shards_ &&
           minor_to_major_ == other.minor_to_major_;
  }

  bool operator!=(const ShardingParam& other) const {
    return !(*this == other);
  }

  llvm::hash_code hash_value() const {
    return llvm::hash_combine(
        dim_shards(), llvm::ArrayRef<int64_t>(minor_to_major_.permutation),
        llvm::ArrayRef<int64_t>(minor_to_major_.axis_sizes));
  }

 private:
  llvm::SmallVector<int64_t, 4> dim_shards_;
  MinorToMajor minor_to_major_;
};

llvm::hash_code hash_value(ShardingParam sharding);

mlir::AsmPrinter& operator<<(mlir::AsmPrinter& os, ShardingParam sharding);

llvm::raw_ostream& operator<<(llvm::raw_ostream& os, ShardingParam sharding);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_IR_SHARDING_PARAM_H_
