/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

//===- cubin_creator.h ------------------------------------------*- C++ -*-===//
//
// This file declares the function to compile a TF kernel function to a cubin.
//
//===----------------------------------------------------------------------===//
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CUBIN_CREATOR_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CUBIN_CREATOR_H_

#include <utility>
#include <vector>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace tensorflow {
namespace kernel_gen {
xla::StatusOr<std::vector<uint8_t>> GenerateCubinForTfCode(
    llvm::StringRef tf_code,
    std::pair<int32_t, int32_t> compute_capability = {7, 5},
    llvm::ArrayRef<uint32_t> tile_sizes = {16, 64},
    llvm::ArrayRef<uint32_t> same_shape = {},
    llvm::ArrayRef<uint32_t> unroll_factors = {});
}  // namespace kernel_gen
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_CUBIN_CREATOR_H_
