/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_TRIVIAL_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_TRIVIAL_SPMD_EXPANDER_H_

#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"

namespace tensorflow {
namespace dtensor {

class NoOpSPMDExpander : public SPMDExpanderBase {
 private:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override {
    return InferSPMDExpandedLocalShape(op);
  }

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override {
    return llvm::DenseMap<int, Layout>();
  }

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override {
    return llvm::DenseMap<int, Layout>();
  }
};

class TerminatorSPMDExpander : public SPMDExpanderBase {
 private:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override {
    return llvm::DenseMap<int, Layout>();
  }

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override {
    return llvm::DenseMap<int, Layout>();
  }
};

// Expansion for metadata operations (like BroadcastGradientArgs) which always
// take replicated inputs and emit a fully replicated output.
class MetadataSPMDExpander : public SPMDExpanderBase {
 public:
  // BroadcastGradientArgs accepts 2 shape tensors and returns 2 values, which
  // indicate reduction dimensions that should be used a part of gradient
  // computation. These dimensions should be computed against the global shape,
  // as there are various specializations that occur when a dimension is of
  // size 1.
  //
  // Since shapes are passed in as value (as opposed to being pulled directly
  // from the input shape), the operation will be performed on the global shape
  // by default.
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_TRIVIAL_SPMD_EXPANDER_H_
