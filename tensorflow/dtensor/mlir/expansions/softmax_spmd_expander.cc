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

#include "tensorflow/dtensor/mlir/expansions/softmax_spmd_expander.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Computes a local reduce followed by an EmitAllReduce. This performs a global
// reduction, output will have global shape 1 on the reduced axes if keep dims
// is true otherwise the axes will be removed.
// Assumes builder's insertion point is after input.
StatusOr<mlir::Value> ComputeGlobalReduce(
    mlir::OpBuilder& builder, const mlir::Value& input,
    const Layout& input_layout, const absl::flat_hash_set<int>& reduced_dims,
    absl::string_view reduce_op, bool keep_dims) {
  TF_ASSIGN_OR_RETURN(
      const Layout reduction_layout,
      input_layout.GetLayoutWithReducedDims(reduced_dims,
                                            /*keep_dims=*/true));
  std::vector<int32> reduce_dim_array(reduced_dims.begin(), reduced_dims.end());
  const mlir::Value reduction_indices =
      IntConst(builder, input.getLoc(), reduce_dim_array);
  mlir::Operation* local_reduce;

  // First compute a local reduce
  if (reduce_op == kReduceOpAdd) {
    local_reduce = builder.create<mlir::TF::SumOp>(
        input.getLoc(), input, reduction_indices,
        /*keep_dims=*/builder.getBoolAttr(true));
  } else if (reduce_op == kReduceOpMax) {
    local_reduce = builder.create<mlir::TF::MaxOp>(
        input.getLoc(), input, reduction_indices,
        /*keep_dims=*/builder.getBoolAttr(true));
  } else {
    return errors::Unimplemented("reduction ", reduce_op, " not implemented");
  }

  // Then an all reduce.
  absl::flat_hash_set<std::string> reduced_sharding_specs;
  for (const int dim : reduced_dims)
    if (Layout::IsShardedDimension(input_layout.sharding_spec(dim)))
      reduced_sharding_specs.emplace(input_layout.sharding_spec(dim));
  TF_ASSIGN_OR_RETURN(
      mlir::Operation * global_reduce,
      EmitAllReduce(builder, reduction_layout, reduced_sharding_specs,
                    local_reduce, reduce_op));

  if (!keep_dims) {
    mlir::RankedTensorType output_type = mlir::dyn_cast<mlir::RankedTensorType>(
        global_reduce->getResult(0).getType());
    if (!output_type)
      return errors::Internal(
          "output of EmitAllReduce is not a RankedTensorType");
    std::vector<int64_t> new_shape;
    for (int i = 0; i < output_type.getRank(); ++i)
      if (!reduced_dims.contains(i))
        new_shape.emplace_back(output_type.getDimSize(i));
    mlir::RankedTensorType new_type =
        mlir::RankedTensorType::get(new_shape, output_type.getElementType());
    // Upcast the dimensions to int64_t as SqueezeOp requires this for its
    // dimension attribute type. Everything else is OK with int32_t dimensions.
    std::vector<int64_t> reduce_dim_array_64(reduced_dims.begin(),
                                             reduced_dims.end());
    global_reduce = builder.create<mlir::TF::SqueezeOp>(
        input.getLoc(), new_type, global_reduce->getResult(0),
        builder.getI64ArrayAttr(reduce_dim_array_64));
  }
  return global_reduce->getResult(0);
}

// Takes a sharded logits and compute both the shifted exponentiation of the
// logits and its sum. Assumes that builder's insertion point is after logits.
absl::Status ComputeExpAndSum(mlir::OpBuilder& builder,
                              const mlir::Value& logits,
                              const Layout& logits_layout,
                              mlir::Value& shifted_logits,
                              mlir::Value& exp_of_shifted_logits,
                              mlir::Value& sum_of_exp) {
  auto loc = logits.getLoc();

  if (logits_layout.rank() == 0)
    return errors::Unimplemented("softmax not supported for rank 0 tensors.");

  const int64 class_dimension = logits_layout.rank() - 1;

  // Softmax is exp(input)/sum(exp(input)) and LogSoftmax is
  // logits - log(sum(exp(input)) where the sum takes place on the
  // last axis.
  // For numerical stability, we shift the logits by the max (along
  // the last axis) before doing the above calculation.

  // Construct the max.
  TF_ASSIGN_OR_RETURN(
      const mlir::Value max_logits,
      ComputeGlobalReduce(builder, logits, logits_layout, {class_dimension},
                          kReduceOpMax, /*keep_dims=*/true));

  // Subtract max from local copy of logits.
  shifted_logits =
      builder.create<mlir::TF::SubOp>(loc, logits, max_logits).getResult();
  exp_of_shifted_logits =
      builder.create<mlir::TF::ExpOp>(loc, shifted_logits).getResult();

  // Sum the exponential.
  TF_ASSIGN_OR_RETURN(
      sum_of_exp,
      ComputeGlobalReduce(builder, exp_of_shifted_logits, logits_layout,
                          {class_dimension}, kReduceOpAdd,
                          /*keep_dims=*/true));
  return absl::OkStatus();
}

// Computes softmax from its components. Assumes that builder's insertion point
// is after sum_of_exp and exp_of_shifted_logits.
mlir::Value ComputeSoftmax(mlir::OpBuilder& builder,
                           const mlir::Value& exp_of_shifted_logits,
                           const mlir::Value& sum_of_exp) {
  // For Softmax, we compute exp(shifted_logits)/sum(exp(shifted_logits))
  auto softmax = builder.create<mlir::TF::DivOp>(
      exp_of_shifted_logits.getLoc(), exp_of_shifted_logits, sum_of_exp);
  return softmax.getResult();
}

// Computes softmax from its components. Assumes that builder's insertion point
// is after shifted_logits and sum_of_exp.
mlir::Value ComputeLogSoftmax(mlir::OpBuilder& builder,
                              const mlir::Value& shifted_logits,
                              const mlir::Value& sum_of_exp) {
  // For LogSoftmax, we compute shifted_logs - log(sum(exp(shifted_logits)))
  auto log_of_sum =
      builder.create<mlir::TF::LogOp>(shifted_logits.getLoc(), sum_of_exp);
  auto log_softmax = builder.create<mlir::TF::SubOp>(
      shifted_logits.getLoc(), shifted_logits, log_of_sum.getResult());
  return log_softmax.getResult();
}

// Computes the softmax of the input along the last axis, assuming that the
// input is sharded along that axis.
StatusOr<mlir::Value> ComputeShardedSoftmax(mlir::OpBuilder& builder,
                                            const mlir::Value& logits,
                                            const Layout& logits_layout,
                                            bool log_softmax) {
  mlir::Value shifted_logits;
  mlir::Value exp_of_shifted_logits;
  mlir::Value sum_of_exp;
  TF_RETURN_IF_ERROR(ComputeExpAndSum(builder, logits, logits_layout,
                                      shifted_logits, exp_of_shifted_logits,
                                      sum_of_exp));

  if (log_softmax) {
    return ComputeLogSoftmax(builder, shifted_logits, sum_of_exp);
  } else {
    return ComputeSoftmax(builder, exp_of_shifted_logits, sum_of_exp);
  }
}

// Creates a layout from specs which is
// 1) Left truncated to match the size of global_shape.
// 2) Has unsharded dimensions where ever global_shape is 1.
StatusOr<Layout> GetBroadcastedLayout(llvm::ArrayRef<int64_t> global_shape,
                                      const std::vector<std::string>& specs,
                                      const Mesh& mesh) {
  std::vector<std::string> new_specs(global_shape.size());
  for (int i = 0; i < global_shape.size(); ++i) {
    if (global_shape[i] == 1)
      new_specs[i] = Layout::kUnshardedDim;
    else
      new_specs[i] = specs[i + specs.size() - global_shape.size()];
  }

  return Layout::GetLayout(new_specs, mesh);
}

// Gets a scalar floating point constant with the same element type as the input
// value. Assumes builder's insertion point is after input.
StatusOr<mlir::Value> GetFPConstOfType(mlir::OpBuilder& builder,
                                       const mlir::Value& input, float value) {
  if (mlir::TensorType type =
          mlir::dyn_cast<mlir::TensorType>(input.getType())) {
    return builder
        .create<mlir::TF::ConstOp>(
            input.getLoc(),
            mlir::DenseFPElementsAttr::get<float>(
                mlir::RankedTensorType::get({}, type.getElementType()),
                {value}))
        .getOutput();
  } else {
    return errors::Unimplemented("non tensor type for labels is not supported");
  }
}

// Takes input, which has layout agreeing with the truncation of desired_layout
// and runs OneHot on it to make it 2 dimensions.
// Assumes builder's insertion point is after input and desired_layout is rank
// 2.
//
// OneHot's element type matches that of features and the number of class is
// derived from features last dimension and the number of shards in the last
// dimension of desired layout.
//
// TODO(bfontain): Extract and share with OneHotSPMDExpander
StatusOr<mlir::Value> ComputeOneHot(mlir::OpBuilder& builder,
                                    const mlir::Value& input,
                                    const mlir::Value& features,
                                    const Layout& desired_layout) {
  // Get the number of classes for this onehot. The number of classes is the
  // global size of the last dimension of features.
  mlir::RankedTensorType features_type =
      mlir::dyn_cast<mlir::RankedTensorType>(features.getType());
  if (!features_type)
    return errors::InvalidArgument(
        "feature input shape must be statically known");
  if (features_type.getRank() == 0)
    return errors::InvalidArgument(
        "expected feature input to have at least rank 1, but found rank 0");

  const int64_t local_classes = features_type.getShape().back();
  const int64_t classes = local_classes * desired_layout.num_shards_for_dim(
                                              desired_layout.rank() - 1);

  int64_t num_shards = desired_layout.num_shards_for_dim(1);
  if (classes % num_shards)
    return errors::InvalidArgument("unable to shard onehot with size ", classes,
                                   " over dimension with ", num_shards,
                                   " shards");
  const mlir::Location& loc = input.getLoc();

  mlir::Value depth = CreateIntScalarConst(classes / num_shards, builder, loc,
                                           /*use_int64=*/false);

  // TODO(bfontain): Extract this block (upto and including the SqueezeOp) to
  // a common function.
  mlir::tf_device::ClusterOp cluster =
      depth.getDefiningOp()->getParentOfType<mlir::tf_device::ClusterOp>();

  // `mesh_coordinates` is tensor of size [1, mesh_size] where each
  // element in the tensor refers to shard id for the specified mesh
  // dimension.
  TF_ASSIGN_OR_RETURN(mlir::Value mesh_coordinates,
                      GetMeshCoordinatesFromCluster(cluster));

  const int mesh_dim_index = desired_layout.mesh().GetMeshDimIndexWithName(
      desired_layout.sharding_spec(/*idx=*/1));

  // Slice out the [1,1] for mesh_dim_index.
  mlir::Value shard_id =
      builder
          .create<mlir::TF::SliceOp>(
              loc, mlir::RankedTensorType::get({1, 1}, builder.getI32Type()),
              mesh_coordinates,
              IntConst(builder, input.getLoc(), {0, mesh_dim_index}),
              IntConst(builder, input.getLoc(), {1, 1}))
          .getOutput();

  shard_id = builder
                 .create<mlir::TF::SqueezeOp>(
                     loc, mlir::RankedTensorType::get({}, builder.getI32Type()),
                     shard_id, builder.getI64ArrayAttr({0, 1}))
                 .getOutput();

  // `new_indices` = `input` - `shard_id` * (classes/num_shards)
  mlir::Value id_offset =
      builder.create<mlir::TF::MulOp>(loc, shard_id, depth).getZ();

  // Note that the type of id_offset (int32) may not match the type of input.
  // So we insert a cast in this case.
  mlir::TensorType input_type =
      mlir::dyn_cast<mlir::TensorType>(input.getType());
  if (!input_type) return errors::InvalidArgument("input is not a TensorType");
  if (!input_type.getElementType().isInteger(32))
    id_offset =
        builder
            .create<mlir::TF::CastOp>(
                loc,
                mlir::RankedTensorType::get({}, input_type.getElementType()),
                id_offset)
            .getY();

  mlir::Value indices =
      builder.create<mlir::TF::SubOp>(loc, input, id_offset).getZ();

  TF_ASSIGN_OR_RETURN(mlir::Value on_value,
                      GetFPConstOfType(builder, features, 1.0));
  TF_ASSIGN_OR_RETURN(mlir::Value off_value,
                      GetFPConstOfType(builder, features, 0.0));

  return builder
      .create<mlir::TF::OneHotOp>(input.getLoc(), indices, depth, on_value,
                                  off_value, builder.getI64IntegerAttr(1))
      .getOutput();
}

}  // namespace

// Expander for Softmax and LogSoftmax ops.
StatusOr<mlir::Operation*> SoftmaxOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  TF_ASSIGN_OR_RETURN(auto logits_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  if (!logits_layout) {
    return errors::InvalidArgument("Failed during SPMD expansion of ",
                                   OpName(op),
                                   ". Layout of logits input must be known.");
  }

  // (Log)Softmax's logits are a rank >= 1 tensor. We reduce over the last
  // dimension. If this is replicated, we don't need any cross-replica
  // operations and can just emit the op as is.
  mlir::Value softmax_result;
  if (logits_layout->IsLastDimReplicated()) {
    InferSPMDExpandedLocalShape(op);
    softmax_result = op->getOpResult(0);
  } else {
    mlir::OpBuilder builder(op);
    builder.setInsertionPointAfter(op);

    TF_ASSIGN_OR_RETURN(
        softmax_result,
        ComputeShardedSoftmax(builder, op->getOperand(0), *logits_layout,
                              mlir::isa<mlir::TF::LogSoftmaxOp>(op)));
    op->getOpResult(0).replaceAllUsesWith(softmax_result);
    op->erase();
  }

  // Add a final Relayout in case the output layout is not the same as the
  // layout of input logits.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  auto final_result = EmitRelayout(softmax_result, *logits_layout,
                                   output_layout, &newly_created_ops);
  if (!final_result.ok()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to relayout the output of softmax: ",
                     final_result.status().message()));
  }
  softmax_result.replaceAllUsesExcept(*final_result, newly_created_ops);
  return final_result->getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
SoftmaxOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // We want to use the same layout for the output.
  return input_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
SoftmaxOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // We want to use the same layout for the input.
  return output_layouts;
}

// Takes the input and output layouts and
// 1) Selects a batch and class sharding from the layouts
// 2) Applies relayout to the input
// 3) Sets the new features and loss layout. Takes into account broadcasting.
// 4) Returns the full layout for backprop/loss.
StatusOr<Layout> SoftmaxLossOpSPMDExpander::MaybeRelayoutInputs(
    mlir::Operation* op, bool is_sparse, const Layout& features_layout,
    const Layout& labels_layout, const Layout& loss_layout,
    const Layout& backprop_layout, Layout& new_features_layout,
    Layout& new_labels_layout) {
  // This layout represents the 'internal layout' that the softmax will be
  // operating on. Inputs will be relayout'ed to this layout and outputs will be
  // relayout'ed from this layout to their desired layout.
  std::vector<std::string> internal_layout(2);
  internal_layout[0] = Layout::kUnshardedDim;
  internal_layout[1] = Layout::kUnshardedDim;

  // Choose an internal layout, ideally this layout would be chosen so that
  // the relayout costs for the inputs (from features_layout/labels_layout to
  // internal_layout) and the outputs (from internal_layout to
  // loss_layout/backprop_layout) are minimized, but we will do something more
  // naive for now.

  // Pick a batch sharding, first from features, then labels, loss and backprop.
  // Due to possible broadcasting on features and labels, they will only
  // have a batch dim if they are rank 2.
  if (features_layout.rank() == 2)
    internal_layout[0] = features_layout.sharding_spec(0);
  if (((labels_layout.rank() == 2) ||
       (is_sparse && labels_layout.rank() == 1)) &&
      Layout::IsUnshardedDimension(internal_layout[0]))
    internal_layout[0] = labels_layout.sharding_spec(0);
  if (Layout::IsUnshardedDimension(internal_layout[0]))
    internal_layout[0] = loss_layout.sharding_spec(0);
  if (Layout::IsUnshardedDimension(internal_layout[0]))
    internal_layout[0] = backprop_layout.sharding_spec(0);

  // Pick a class sharding, first from features, then labels and backprop.
  // The class dim for features and labels is always the last dim if it exists.
  // Note that loss and backprop have fixed ranks 1 and 2 respectively where as
  // ranks of features and labels may involved broadcasting.
  if (features_layout.rank() > 0 &&
      (internal_layout[0] !=
       features_layout.sharding_spec(features_layout.rank() - 1)))
    internal_layout[1] =
        features_layout.sharding_spec(features_layout.rank() - 1);
  if (!is_sparse && labels_layout.rank() > 0 &&
      Layout::IsUnshardedDimension(internal_layout[1]) &&
      (internal_layout[0] !=
       labels_layout.sharding_spec(labels_layout.rank() - 1)))
    internal_layout[1] = labels_layout.sharding_spec(labels_layout.rank() - 1);
  if (Layout::IsUnshardedDimension(internal_layout[1]) &&
      (internal_layout[0] != backprop_layout.sharding_spec(1)))
    internal_layout[1] = backprop_layout.sharding_spec(1);

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> features_global_shape,
      GetGlobalShapeOfValueFromDTensorLayout(op->getOperand(0)));

  // At this point we need to compute the new layout of features and labels.
  // Broadcasting makes this more complicated: First we truncate the correct
  // rank and then set any dimensions where the global shape is size 1 to
  // unsharded.
  TF_ASSIGN_OR_RETURN(
      new_features_layout,
      GetBroadcastedLayout(features_global_shape, internal_layout,
                           features_layout.mesh()));

  TF_ASSIGN_OR_RETURN(
      const mlir::Value new_features,
      EmitRelayout(op->getOperand(0), features_layout, new_features_layout));

  op->setOperand(0, new_features);

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> labels_global_shape,
      GetGlobalShapeOfValueFromDTensorLayout(op->getOperand(1)));

  if (is_sparse) {
    // If we are sparse, then the only possible dimension is the batch_dim.
    std::vector<std::string> sparse_specs = {internal_layout[0]};
    TF_ASSIGN_OR_RETURN(new_labels_layout,
                        GetBroadcastedLayout(labels_global_shape, sparse_specs,
                                             labels_layout.mesh()));
  } else {
    TF_ASSIGN_OR_RETURN(
        new_labels_layout,
        GetBroadcastedLayout(labels_global_shape, internal_layout,
                             labels_layout.mesh()));
  }

  TF_ASSIGN_OR_RETURN(
      const mlir::Value new_labels,
      EmitRelayout(op->getOperand(1), labels_layout, new_labels_layout));

  op->setOperand(1, new_labels);

  return Layout::GetLayout(internal_layout, features_layout.mesh());
}

// Takes the given loss, backprop values and relayouts them out to the required
// layouts and pass them through an IdentityN op.
// This assumes that the input have local shape in their type.
StatusOr<mlir::Operation*> SoftmaxLossOpSPMDExpander::MaybeRelayoutOutputs(
    mlir::Operation* op, const mlir::Value& loss, const mlir::Value& backprop,
    const Layout& output_layout, const Layout& loss_layout,
    const Layout& backprop_layout) {
  const Layout current_loss_layout = output_layout.Truncate(/*split_point=*/1);
  const Layout& current_backprop_layout = output_layout;

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  TF_ASSIGN_OR_RETURN(
      const mlir::Value new_loss,
      EmitRelayout(loss, current_loss_layout, loss_layout, &newly_created_ops));

  TF_ASSIGN_OR_RETURN(const mlir::Value new_backprop,
                      EmitRelayout(backprop, current_backprop_layout,
                                   backprop_layout, &newly_created_ops));

  mlir::OpBuilder builder(loss.getContext());

  if (new_loss.getDefiningOp()->isBeforeInBlock(new_backprop.getDefiningOp()))
    builder.setInsertionPointAfterValue(new_backprop);
  else
    builder.setInsertionPointAfterValue(new_loss);

  llvm::SmallVector<mlir::Type, 4> types = {new_loss.getType(),
                                            new_backprop.getType()};
  llvm::SmallVector<mlir::Value, 4> values = {new_loss, new_backprop};

  mlir::TF::IdentityNOp identity_op =
      builder.create<mlir::TF::IdentityNOp>(loss.getLoc(), types, values);

  newly_created_ops.insert(identity_op);

  op->getResult(0).replaceAllUsesExcept(identity_op.getResult(0),
                                        newly_created_ops);
  op->getResult(1).replaceAllUsesExcept(identity_op.getResult(1),
                                        newly_created_ops);

  // If the op we are expanding isn't being used any more, erase it from the
  // program.
  if (op->getResult(0).use_empty() && op->getResult(1).use_empty()) op->erase();

  return identity_op.getOperation();
}

StatusOr<mlir::Operation*> SoftmaxLossOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  if (!mlir::isa<mlir::TF::SoftmaxCrossEntropyWithLogitsOp>(op) &&
      !mlir::isa<mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp>(op))
    return errors::InvalidArgument(
        "unsupported op for in SoftmaxLossOpSPMDExpander");

  TF_ASSIGN_OR_RETURN(const Layout& features_layout,
                      ExtractRequiredLayoutFromOperand(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const Layout& labels_layout,
                      ExtractRequiredLayoutFromOperand(op->getOperand(1)));
  TF_ASSIGN_OR_RETURN(const std::vector<Layout>& output_layouts,
                      ExtractRequiredLayoutFromOp(op));

  const bool is_sparse =
      mlir::isa<mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp>(op);

  Layout new_features_layout;
  Layout new_labels_layout;

  TF_ASSIGN_OR_RETURN(
      const Layout internal_layout,
      MaybeRelayoutInputs(op, is_sparse, features_layout, labels_layout,
                          output_layouts[0], output_layouts[1],
                          new_features_layout, new_labels_layout));

  assert(internal_layout.rank() == 2);

  // If the class dim is unshared, we can emit a local op.
  if (Layout::IsUnshardedDimension(internal_layout.sharding_spec(1))) {
    op = InferSPMDExpandedLocalShape(op);
    return MaybeRelayoutOutputs(op, op->getResult(0), op->getResult(1),
                                internal_layout, output_layouts[0],
                                output_layouts[1]);
  }

  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(op);

  mlir::Value features = op->getOperand(0);
  mlir::Value labels = op->getOperand(1);
  if (is_sparse) {
    // SparseSoftmaxCrossEntropyWithLogits(features, labels) can be rewritten
    // as SoftmaxCrossEntropyWithLogits(features, OneHot(labels)).
    // Note that this is what is done in the XLA kernel for this op.
    TF_ASSIGN_OR_RETURN(
        labels, ComputeOneHot(builder, labels, features, internal_layout));
  }

  if (features_layout.rank() == 0)
    return errors::Unimplemented(
        "scalar values features is not currently supported");

  // SoftmaxCrossEntropyWithLogitsOp is the same as:
  // loss = -tf.reduce_sum(labels*tf.LogSoftmax(features), class_dim)
  // backprop = tf.Softmax(features) - labels

  mlir::Value shifted_logits;
  mlir::Value exp_of_shifted_logits;
  mlir::Value sum_of_exp;

  // Note that its possible that features is shape [x, 1] and is broadcasted
  // to match labels. In this case we are doing a bunch of extra work, since
  // softmax is 1 and log_softmax is 0.
  TF_RETURN_IF_ERROR(ComputeExpAndSum(builder, features, new_features_layout,
                                      shifted_logits, exp_of_shifted_logits,
                                      sum_of_exp));

  const mlir::Value log_softmax =
      ComputeLogSoftmax(builder, shifted_logits, sum_of_exp);
  const mlir::Value softmax =
      ComputeSoftmax(builder, exp_of_shifted_logits, sum_of_exp);

  // Mimic the XLA, which uses where/select to ensure that sub is zero when
  // labels are zero.
  TF_ASSIGN_OR_RETURN(const mlir::Value features_zero,
                      GetFPConstOfType(builder, features, 0.0));
  TF_ASSIGN_OR_RETURN(const mlir::Value labels_zero,
                      GetFPConstOfType(builder, labels, 0.0));

  const mlir::Value is_labels_zero =
      builder
          .create<mlir::TF::EqualOp>(op->getLoc(), labels, labels_zero,
                                     builder.getBoolAttr(true))
          .getZ();
  const mlir::Value safe_softmax =
      builder
          .create<mlir::TF::SelectV2Op>(op->getLoc(), is_labels_zero,
                                        features_zero, log_softmax)
          .getOutput();
  const mlir::Value prod =
      builder.create<mlir::TF::MulOp>(op->getLoc(), labels, safe_softmax)
          .getZ();

  // Compute the reduce sum
  TF_ASSIGN_OR_RETURN(
      mlir::Value positive_loss,
      ComputeGlobalReduce(builder, prod, internal_layout, /*reduced_dims=*/{1},
                          kReduceOpAdd, /*keep_dims=*/false));

  builder.setInsertionPointAfterValue(positive_loss);
  mlir::Value loss =
      builder.create<mlir::TF::NegOp>(op->getLoc(), positive_loss).getY();

  mlir::Value backprop =
      builder.create<mlir::TF::SubOp>(op->getLoc(), softmax, labels);

  return MaybeRelayoutOutputs(op, loss, backprop, internal_layout,
                              output_layouts[0], output_layouts[1]);
}

StatusOr<llvm::DenseMap<int, Layout>>
SoftmaxLossOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  const bool is_sparse =
      mlir::isa<mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp>(op);

  // loss is sum(-labels * logsoftmax(features)), so the layout is batch
  // sharded if labels and features are batch sharded on the same mesh dim or
  // if one is replicated.
  // backprop is softmax(features) - labels

  std::optional<Layout> features_layout;
  if (input_layouts.find(0) != input_layouts.end())
    features_layout.emplace(input_layouts.lookup(0));
  std::optional<Layout> labels_layout;
  if (input_layouts.find(1) != input_layouts.end())
    labels_layout.emplace(input_layouts.lookup(1));

  // We need to compute shardings for two dimensions: batch and class.
  std::vector<std::string> layout_specs(2);
  layout_specs[0] = Layout::kUnshardedDim;
  layout_specs[1] = Layout::kUnshardedDim;

  // First pick the batch dimension, set it to the batch dimension of features
  // if it exists otherwise to the batch dimesion of labels.
  if (features_layout && (features_layout->rank() == 2))
    layout_specs[0] = features_layout->sharding_spec(0);
  if (labels_layout &&
      (labels_layout->rank() == 2 ||
       (is_sparse && labels_layout->rank() == 1)) &&
      Layout::IsUnshardedDimension(layout_specs[0]))
    layout_specs[0] = labels_layout->sharding_spec(0);

  // The class sharding_spec for features and labels is always the last
  // sharding_spec if it exists.
  if (features_layout && (features_layout->rank() > 0) &&
      (layout_specs[0] !=
       features_layout->sharding_spec(features_layout->rank() - 1)))
    layout_specs[1] =
        features_layout->sharding_spec(features_layout->rank() - 1);
  if (!is_sparse && labels_layout && (labels_layout->rank() > 0) &&
      Layout::IsUnshardedDimension(layout_specs[1]) &&
      (layout_specs[0] !=
       labels_layout->sharding_spec(labels_layout->rank() - 1)))
    layout_specs[1] = labels_layout->sharding_spec(labels_layout->rank() - 1);

  TF_ASSIGN_OR_RETURN(const Layout backprop_layout,
                      Layout::GetLayout(layout_specs, mesh));
  const Layout loss_layout = backprop_layout.Truncate(/*split_point=*/1);

  return llvm::DenseMap<int, Layout>({{0, loss_layout}, {1, backprop_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
SoftmaxLossOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  const bool is_sparse =
      mlir::isa<mlir::TF::SparseSoftmaxCrossEntropyWithLogitsOp>(op);

  std::optional<Layout> loss_layout;
  if (output_layouts.find(0) != output_layouts.end())
    loss_layout.emplace(output_layouts.lookup(0));
  std::optional<Layout> backprop_layout;
  if (output_layouts.find(1) != output_layouts.end())
    backprop_layout.emplace(output_layouts.lookup(1));

  // We need to compute two possible shardings:
  // One for the batch dimension and one for the class dimension.
  std::vector<std::string> layout_specs(2);
  layout_specs[0] = Layout::kUnshardedDim;
  layout_specs[1] = Layout::kUnshardedDim;

  // Respect the loss layout if it is set, otherwise use the backprop
  // layout for the batch_dim.
  if (loss_layout) layout_specs[0] = loss_layout->sharding_spec(0);
  if (backprop_layout && Layout::IsUnshardedDimension(layout_specs[0]))
    layout_specs[0] = backprop_layout->sharding_spec(0);

  // Only backprop has class dim so use that if it is available.
  if (backprop_layout && backprop_layout->sharding_spec(1) != layout_specs[0])
    layout_specs[1] = backprop_layout->sharding_spec(1);

  TF_ASSIGN_OR_RETURN(const auto features_shape,
                      GetShapeOfValue(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const auto labels_shape,
                      GetShapeOfValue(op->getOperand(1)));
  TF_ASSIGN_OR_RETURN(const Layout features_layout,
                      GetBroadcastedLayout(features_shape, layout_specs, mesh));
  if (is_sparse) {
    // Drop the class sharding as the labels don't have class dimension in
    // the sparse version.
    layout_specs.resize(1);
  }
  TF_ASSIGN_OR_RETURN(const Layout labels_layout,
                      GetBroadcastedLayout(labels_shape, layout_specs, mesh));

  return llvm::DenseMap<int, Layout>(
      {{0, features_layout}, {1, labels_layout}});
}

}  // namespace dtensor
}  // namespace tensorflow
