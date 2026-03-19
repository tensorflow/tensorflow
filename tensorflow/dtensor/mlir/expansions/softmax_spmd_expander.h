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

#ifndef TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_SOFTMAX_SPMD_EXPANDER_H_
#define TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_SOFTMAX_SPMD_EXPANDER_H_

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"

namespace tensorflow {
namespace dtensor {

// Expander for Softmax and LogSoftmax ops.
class SoftmaxOpSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;
};

// Expander for SoftmaxCrossEntropyWithLogits ops.
class SoftmaxLossOpSPMDExpander : public SPMDExpanderBase {
 public:
  StatusOr<mlir::Operation*> ExpandOp(mlir::Operation* op) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutForward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& input_layouts) override;

  StatusOr<llvm::DenseMap<int, Layout>> ComputeLayoutBackward(
      mlir::Operation* op,
      const llvm::DenseMap<int, Layout>& output_layouts) override;

 private:
  // Computes the relayouts of the inputs of the softmax loss op. Returns the
  // internal layout of the softmax loss in new_features_layout and
  // new_labels_layout.
  StatusOr<Layout> MaybeRelayoutInputs(mlir::Operation* op, bool is_sparse,
                                       const Layout& features_layout,
                                       const Layout& labels_layout,
                                       const Layout& loss_layout,
                                       const Layout& backprop_layout,
                                       Layout& new_features_layout,
                                       Layout& new_labels_layout);

  // Computes relayouts of the outputs, returning an IdentityN op that ties
  // together the loss and backprop returns.
  StatusOr<mlir::Operation*> MaybeRelayoutOutputs(
      mlir::Operation* op, const mlir::Value& loss, const mlir::Value& backprop,
      const Layout& output_layout, const Layout& loss_layout,
      const Layout& backprop_layout);
};

}  // namespace dtensor
}  // namespace tensorflow

#endif  // TENSORFLOW_DTENSOR_MLIR_EXPANSIONS_SOFTMAX_SPMD_EXPANDER_H_
