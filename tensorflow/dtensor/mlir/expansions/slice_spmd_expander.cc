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

#include "tensorflow/dtensor/mlir/expansions/slice_spmd_expander.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

absl::Status GetSliceOpArguments(mlir::TF::SliceOp slice_op,
                                 llvm::SmallVector<int64_t, 4>& begins,
                                 bool& dynamic_begins,
                                 llvm::SmallVector<int64_t, 4>& sizes) {
  absl::Status begins_result =
      ExtractConstVectorFromValue(slice_op.getBegin(), &begins);
  dynamic_begins = !begins_result.ok();

  TF_RETURN_WITH_CONTEXT(
      ExtractConstVectorFromValue(slice_op.getSize(), &sizes),
      "expected constant argument for SliceOp::size()");

  return absl::OkStatus();
}

StatusOr<Layout> VerifySliceLayout(
    mlir::Operation* slice_op, mlir::Value value, const Layout& layout,
    llvm::ArrayRef<int64_t>* global_shape = nullptr) {
  if (layout.IsFullyReplicated()) return layout;

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> shape,
                      GetShapeOfValue(value, /*fail_on_dynamic=*/true));
  const int64_t rank = shape.size();
  if (global_shape != nullptr) {
    // In ExpandOp, tensor shape is local shape. So, call site needs to provide
    // global shape expliclity.
    shape = *global_shape;
  }

  llvm::SmallVector<int64_t, 4> begins, sizes;
  bool dynamic_begins = false;
  begins.reserve(rank);
  sizes.reserve(rank);

  TF_RETURN_IF_ERROR(GetSliceOpArguments(
      llvm::cast<mlir::TF::SliceOp>(slice_op), begins, dynamic_begins, sizes))

  auto num_shards = layout.num_shards();

  std::vector<std::string> sharding_specs;

  for (int64_t i = 0; i < rank; ++i) {
    const bool begins_starts_at_zero =
        (sizes[i] == shape[i]) || (!dynamic_begins && begins[i] == 0);
    const bool ends_at_full_size =
        (sizes[i] == shape[i]) || (!dynamic_begins && sizes[i] == -1);

    if (begins_starts_at_zero && ends_at_full_size) {
      // We support slicing with dynamic begins when the sharded dimensions are
      // getting a full slice. Since we don't know the begins in this case, we
      // need to rely in the sizes being static and equal to the global shape.
      // In particular sizes[i] == shape[i] implies begins[i] == 0.
      // A full slice over the any dimension can be performed locally.
      sharding_specs.push_back(layout.sharding_spec(i));
    } else {
      // Slicing on sharded dim is not trivial. Propose an unsharded dim for
      // that.
      sharding_specs.push_back(Layout::kUnshardedDim);
    }
  }
  return Layout::GetLayout(sharding_specs, layout.mesh());
}

}  // namespace

StatusOr<mlir::Operation*> SliceSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto slice_op = mlir::cast<mlir::TF::SliceOp>(op);
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(slice_op.getInput()));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));

  if (!output_layout || !input_layout)
    return errors::Unimplemented(
        "layout of Slice op must be known before SPMD expansion.");

  // The dyn_cast will never be nullptr as it is checked in
  // GetLayoutFromOperands.
  auto input_type =
      mlir::dyn_cast<mlir::RankedTensorType>(slice_op.getInput().getType());
  if (!input_type)
    return errors::InvalidArgument(
        "rank of input tensor must be statically known for slice op.");

  TF_ASSIGN_OR_RETURN(auto global_shape,
                      ExtractGlobalInputShape(op->getOpOperand(0)));
  const int64_t input_rank = input_type.getRank();

  llvm::SmallVector<int64_t, 4> begins, sizes;
  bool dynamic_begins = false;
  begins.reserve(input_rank);
  sizes.reserve(input_rank);

  TF_RETURN_IF_ERROR(
      GetSliceOpArguments(slice_op, begins, dynamic_begins, sizes));

  TF_ASSIGN_OR_RETURN(auto proposed_layout,
                      VerifySliceLayout(slice_op, slice_op.getInput(),
                                        *input_layout, &global_shape));

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(auto relayout_input,
                      EmitRelayout(op->getOperand(0), *input_layout,
                                   proposed_layout, &newly_created_ops));
  {
    // Adjusts the sizes when it is full slicing on sharded dimension.
    // Note that proposed layout is unsharded in the cases that:
    // 1) We can't determine the begins and sizes != global shape
    // 2) begins != 0
    // 3) sizes != global shape or -1
    const std::vector<int> num_shards = proposed_layout.num_shards();
    for (int64_t i = 0; i < input_rank; ++i) {
      if (num_shards[i] == 1) continue;

      if (sizes[i] == -1 && !dynamic_begins && begins[i] == 0) continue;

      if (sizes[i] == global_shape[i]) {
        // Set the correct output size. If the input dynamic and this is -1,
        // then shape inference can't tell the output shape.
        sizes[i] = global_shape[i] / num_shards[i];
        continue;
      }

      return errors::InvalidArgument(
          "Non-full-slicing on the sharded dimension is not allowed. "
          "internal bug.");
    }
  }

  mlir::OpBuilder builder(op);
  mlir::Value new_size;
  auto loc = op->getLoc();
  // Both begin and size need to be the same type, so we must match the new
  // size input with the type of begin.
  if (!mlir::isa<mlir::ShapedType>(slice_op.getBegin().getType()))
    return errors::Internal("type of begin is not a ShapedType");
  mlir::ShapedType type =
      mlir::cast<mlir::ShapedType>(slice_op.getBegin().getType());
  if (type.getElementType().isInteger(32))
    new_size = IntConst(
        builder, loc, llvm::SmallVector<int32, 4>(sizes.begin(), sizes.end()));
  else
    new_size = Int64Const(builder, loc, sizes);

  auto new_op = builder
                    .create<mlir::TF::SliceOp>(
                        loc, slice_op.getOutput().getType(), relayout_input,
                        slice_op.getBegin(), new_size)
                    .getOperation();
  new_op = InferSPMDExpandedLocalShape(new_op);

  TF_ASSIGN_OR_RETURN(auto relayout_output,
                      EmitRelayout(new_op->getResult(0), proposed_layout,
                                   *output_layout, &newly_created_ops));

  op->getOpResult(0).replaceAllUsesExcept(relayout_output, newly_created_ops);
  op->erase();
  return relayout_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> SliceSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto slice_op = mlir::cast<mlir::TF::SliceOp>(op);

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(
      auto proposed_layout,
      VerifySliceLayout(slice_op, slice_op.getInput(), input_layout));
  return llvm::DenseMap<int, Layout>({{0, proposed_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> SliceSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto slice_op = mlir::cast<mlir::TF::SliceOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(slice_op.getNumOperands());
  // Set replicated layout for begin and size operands.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // input
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout& output_layout = output_layouts.lookup(0);
    TF_ASSIGN_OR_RETURN(
        auto proposed_layout,
        VerifySliceLayout(slice_op, slice_op.getOutput(), output_layout));
    input_layouts[0] = proposed_layout;
  }

  return input_layouts;
}


}  // namespace dtensor
}  // namespace tensorflow
