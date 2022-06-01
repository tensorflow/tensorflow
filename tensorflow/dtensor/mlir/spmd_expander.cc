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

#include "tensorflow/dtensor/mlir/spmd_expander.h"

#include <climits>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/optional.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

// static
SPMDExpanderRegistry* SPMDExpanderRegistry::Global() {
  static SPMDExpanderRegistry* registry = new SPMDExpanderRegistry();
  return registry;
}

SPMDExpanderBase* SPMDExpanderRegistry::GetPropagateFnForOp(
    mlir::Operation* op) {
  auto key = OpName(op);
  auto fn = op_to_propagate_fn_map_.find(key);
  if (fn == op_to_propagate_fn_map_.end()) return nullptr;
  return fn->second.get();
}

InitOnStartupMarker SPMDExpanderRegistry::RegisterPropagateFn(
    std::string opName, std::unique_ptr<SPMDExpanderBase> prop) {
  CHECK(op_to_propagate_fn_map_  // Crash ok
            .insert_or_assign(opName, std::move(prop))
            .second);
  return {};
}

Status SPMDExpanderBase::ExpandOpAndSetLayout(mlir::Operation* op,
                                              mlir::Operation** output) {
  TF_ASSIGN_OR_RETURN(std::vector<absl::optional<Layout>> computed_layout,
                      ExtractLayoutFromOp(op));

  if (computed_layout.empty() && op->getNumResults() != 0) {
    return errors::InvalidArgument(
        absl::StrCat("No attachced layout found for op : ", OpName(op),
                     " This might be due to an error in layout propagation.")
            .c_str());
  }

  // `op` may be removed/replaced from the graph during SPMD expansion, so
  // extract the global output shape before expansion.
  llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> global_output_shapes;
  global_output_shapes.reserve(op->getNumResults());
  for (auto output_value : op->getResults()) {
    auto maybe_ranked =
        output_value.getType().dyn_cast<mlir::RankedTensorType>();
    // Do not extract global shape if the shape isn't statically known.
    //
    // This is a bit subtle and relies on the check of static shape of output
    // value below when extracting local_shape. We probably should consider a
    // placeholder for unknown shapes to avoid surprises in the future.
    //
    // Given the nature of RestoreV2 op and its output ranks, we only special
    // case for RestoreV2 for now.
    if (llvm::isa<mlir::TF::RestoreV2Op, mlir::TF::DTensorRestoreV2Op>(op) &&
        (!maybe_ranked || !maybe_ranked.hasStaticShape()))
      continue;
    TF_ASSIGN_OR_RETURN(auto global_shape,
                        ExtractGlobalOutputShape(output_value));
    global_output_shapes.emplace_back(llvm::SmallVector<int64_t, 4>{
        global_shape.begin(), global_shape.end()});
  }

  TF_ASSIGN_OR_RETURN(*output, this->ExpandOp(op));

  // TODO(hthu): Use ToString() instead.
  SetLayoutOnOp(*output, absl::Span<absl::optional<Layout>>(
                             computed_layout.data(), computed_layout.size()));

  // Verify the local shape of the expanded operation matches the shape expected
  // from the layout. Note that this does **not** catch all errors. When tensor
  // dimension is sharded in a wrong mesh with the same device cardinality as
  // the correct/expected mesh, this check will still pass.
  for (const auto& output_layout_and_index :
       llvm::enumerate(llvm::zip((*output)->getResults(), computed_layout))) {
    const int index = output_layout_and_index.index();
    const auto& output_and_layout = output_layout_and_index.value();

    auto output_value = std::get<0>(output_and_layout);
    // Extract the static shape of `output_value` if possible, otherwise ignore
    // this output.
    auto local_expanded_shape_or_status = GetShapeOfValue(output_value);
    if (!local_expanded_shape_or_status.ok()) continue;

    const auto local_expanded_shape =
        local_expanded_shape_or_status.ValueOrDie();
    const auto& layout = std::get<1>(output_and_layout);
    const auto expected_global_shape =
        layout->GlobalShapeFromLocalShape(local_expanded_shape);

    for (const auto& expanded_and_true_global_shape :
         llvm::zip(global_output_shapes[index], expected_global_shape)) {
      const auto expanded_shape = std::get<0>(expanded_and_true_global_shape);
      const auto expected_shape = std::get<1>(expanded_and_true_global_shape);
      // If any of the shape has unknown dimension, do not check/validate the
      // shape.
      if (expanded_shape <= 0 || expected_shape <= 0) continue;

      if (expanded_shape != expected_shape) {
        return errors::Internal(
            "SPMD expansion resulted in op output inconsistent with the "
            "provided layout.");
      }
    }
  }

  return OkStatus();
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutForward API must be implemented via the subclass.");
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  return ComputeLayoutForward(op, input_layouts);
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "ComputeLayoutBackward API must be implemented via the subclass.");
}

StatusOr<llvm::DenseMap<int, Layout>> SPMDExpanderBase::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts,
    const llvm::DenseMap<int, Layout>& output_layouts) {
  return ComputeLayoutBackward(op, output_layouts);
}

Status RunSPMDExpansion(mlir::Operation* op, mlir::Operation** output) {
  SPMDExpanderBase* expander =
      SPMDExpanderRegistry::Global()->GetPropagateFnForOp(op);
  if (expander != nullptr) {
    return expander->ExpandOpAndSetLayout(op, output);
  } else {
    VLOG(1) << "No expansion found for " << OpName(op) << "\n";
    *output = op;
  }
  return OkStatus();
}

}  // namespace dtensor
}  // namespace tensorflow
