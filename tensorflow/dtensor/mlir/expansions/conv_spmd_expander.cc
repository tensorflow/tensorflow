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

#include "tensorflow/dtensor/mlir/expansions/conv_spmd_expander.h"

#include <string>

#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

namespace {

Status VerifyConv2DLayout(const Layout& input_layout,
                          const Layout& kernel_layout,
                          mlir::TF::Conv2DOp conv_op) {
  if (!kernel_layout.IsFullyReplicated())
    return errors::InvalidArgument(
        "Filter for conv2d op must have fully replicated layout.");

  const int channel_index = conv_op.data_format().str() == "NHWC" ? 3 : 1;
  if (input_layout.sharding_spec(channel_index) != Layout::kUnshardedDim)
    return errors::InvalidArgument("Channel dimension must be replicated.");

  if (!input_layout.IsBatchParallel()) {
    int height_dim = 1;
    int width_dim = 2;
    if (channel_index == 1) {
      height_dim = 2;
      width_dim = 3;
    }

    if (input_layout.sharding_spec(height_dim) == Layout::kUnshardedDim ||
        input_layout.sharding_spec(width_dim) == Layout::kUnshardedDim)
      return errors::InvalidArgument(
          "If convolution has spatially partitioned input, both input width "
          "and height dimension must be sharded.");

    const int num_non_default_dilations =
        llvm::count_if(conv_op.dilations(), [](mlir::Attribute dilation) {
          return dilation.cast<mlir::IntegerAttr>().getInt() != 1;
        });
    if (num_non_default_dilations > 0)
      return errors::InvalidArgument(
          "Only dilation rate 1 is supported for convolution with spatial "
          "partition.");
    if (conv_op.padding().str() != "SAME")
      return errors::InvalidArgument(
          "Only SAME padding is allowed for conv2d with spatial partition.");

    mlir::Value filter = conv_op.filter();
    auto filter_type = filter.getType().dyn_cast<mlir::RankedTensorType>();
    if (!filter_type || !filter_type.hasStaticShape())
      return errors::InvalidArgument(
          "Filter must have static shapes for conv2d with spatial "
          "partition.");

    llvm::ArrayRef<int64_t> filter_shape = filter_type.getShape();
    if (filter_shape[0] % 2 != 1 || filter_shape[1] % 2 != 1)
      return errors::InvalidArgument(
          "Filter width and height must be an odd number for conv2d with "
          "spatial partition.");
  }
  return Status::OK();
}

StatusOr<mlir::Operation*> HandleConv2DSPMD(mlir::TF::Conv2DOp conv2d,
                                            mlir::OpBuilder& builder) {
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(conv2d.input()));
  TF_ASSIGN_OR_RETURN(const auto filter_layout,
                      ExtractLayoutFromOperand(conv2d.filter()));

  TF_RETURN_IF_ERROR(VerifyConv2DLayout(input_layout, *filter_layout, conv2d));

  if (!input_layout.IsBatchParallel()) {
    const std::vector<std::string> sharding_spec =
        input_layout.sharding_spec_strs();
    auto kernel_type =
        conv2d.filter().getType().dyn_cast<mlir::RankedTensorType>();
    if (!kernel_type || !kernel_type.hasStaticShape()) {
      return errors::InvalidArgument(
          llvm::formatv("requires kernel type to have statically known rank, "
                        "but got : {0}",
                        kernel_type)
              .str());
    }
    auto kernel_shape = kernel_type.getShape();

    // For non-batch, non-channel dimension sharding, conduct halo exchange.
    for (int i = 1; i < sharding_spec.size(); ++i) {
      if (sharding_spec[i] == Layout::kUnshardedDim) continue;

      const int halo_size = std::floor(kernel_shape[i - 1] / 2);

      builder.setInsertionPoint(conv2d);
      TF_ASSIGN_OR_RETURN(
          mlir::Value halo_exchanged_input,
          EmitHaloExchange(
              halo_size, sharding_spec[i], input_layout, builder,
              conv2d->getParentOfType<mlir::tf_device::ClusterOp>(),
              conv2d->getLoc(), conv2d.input()));

      conv2d->setOperand(0, halo_exchanged_input);
      conv2d.paddingAttr(builder.getStringAttr("VALID"));
    }
  }
  return InferSPMDExpandedLocalShape(conv2d);
}

template <typename ConvBackpropInputOp>
StatusOr<mlir::Operation*> HandleConvBackpropInput(const Layout& output_layout,
                                                   mlir::Operation* op) {
  auto conv_op = llvm::cast<ConvBackpropInputOp>(op);
  llvm::SmallVector<int64_t, 4> global_shape;
  Status extract_status =
      ExtractConstVectorFromValue(conv_op.input_sizes(), &global_shape);

  // Recover local shape in SPMD expansion.
  if (extract_status.ok()) {
    auto local_shape = output_layout.LocalShapeFromGlobalShape(global_shape);
    mlir::OpBuilder builder(conv_op->getBlock(), conv_op->getBlock()->begin());
    auto new_const = IntConst(
        builder, conv_op->getLoc(),
        llvm::SmallVector<int32_t, 4>(local_shape.begin(), local_shape.end()));
    conv_op.input_sizesMutable().assign(new_const);
  }

  return InferSPMDExpandedLocalShape(conv_op);
}

StatusOr<mlir::Operation*> HandleConv2dBackPropFilter(
    const Layout& output_layout,
    mlir::TF::Conv2DBackpropFilterOp conv_2d_backprop_op) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> input_layout,
                      ExtractLayoutFromOperand(conv_2d_backprop_op.input()));

  TF_ASSIGN_OR_RETURN(
      absl::optional<Layout> out_backprop_layout,
      ExtractLayoutFromOperand((conv_2d_backprop_op.out_backprop())));
  // Perform a split on batch dimension so that the each local device performs
  // local operation.
  // TODO(hthu): Make this work on input with rank higher than 4.
  if (input_layout->IsBatchParallel()) {
    mlir::OpBuilder builder(conv_2d_backprop_op);
    if (out_backprop_layout->IsFullyReplicated()) {
      TF_ASSIGN_OR_RETURN(
          const mlir::Value batch_sharded,
          EmitAllScatter(builder, conv_2d_backprop_op.out_backprop(),
                         *out_backprop_layout, *input_layout));
      conv_2d_backprop_op.out_backpropMutable().assign(batch_sharded);
    }

    // Perform all reduce over batch dim.
    builder.setInsertionPointAfter(conv_2d_backprop_op);
    return DT_CTX(EmitAllReduce(builder, output_layout,
                                {input_layout->sharding_spec(0)},
                                conv_2d_backprop_op, kReduceOpAdd));
  } else {
    return errors::InvalidArgument(
        "Conv2d backprop for spatially partitioned input not yet supported.");
  }
  return InferSPMDExpandedLocalShape(conv_2d_backprop_op);
}

StatusOr<mlir::Operation*> HandleMaxPoolGradOp(
    const Layout& output_layout, mlir::TF::MaxPoolGradOp max_pool_grad_op) {
  // MaxPoolGrad has 3 inputs: Original Input to MaxPool, Output of MaxPool and
  // Gradients.
  assert(max_pool_grad_op->getOpOperands().size() == 3);

  // Relayout gradient input to match layout of output of maxpool.
  mlir::OpOperand& max_pool_output = max_pool_grad_op->getOpOperand(1);
  TF_ASSIGN_OR_RETURN(Layout max_pool_output_layout,
                      ExtractRequiredLayoutFromOperand(max_pool_output.get()));

  mlir::OpOperand& grad_input = max_pool_grad_op->getOpOperand(2);
  TF_ASSIGN_OR_RETURN(Layout grad_input_layout,
                      ExtractRequiredLayoutFromOperand(grad_input.get()));
  TF_ASSIGN_OR_RETURN(mlir::Value new_grad_input,
                      EmitRelayout(grad_input.get(), grad_input_layout,
                                   max_pool_output_layout));
  grad_input.set(new_grad_input);

  return InferSPMDExpandedLocalShape(max_pool_grad_op);
}

}  // namespace

StatusOr<mlir::Operation*> ConvSPMDExpander::ExpandOp(mlir::Operation* op) {
  // The first argument to Conv2DBackpropInputOp is the shape of the input we
  // are generating. Since this is almost always the output of a call to
  // `shape`, we lose the ability to infer the original input layout. (c.f if
  // Conv2DBackpropInput accepted the input _tensor_ instead of the shape).
  // Since in eager execution, we cannot look ahead at consumer operations, we
  // instead attach the original input layout as a secondary attribute on the
  // output of the shape operation, and use this to infer the desired layout for
  // this op.

  mlir::OpBuilder builder(op);
  TF_ASSIGN_OR_RETURN(const auto output_layout, ExtractSingleLayoutFromOp(op));

  if (auto conv2d = llvm::dyn_cast<mlir::TF::Conv2DOp>(op))
    return HandleConv2DSPMD(conv2d, builder);

  // For all other ops (other than conv2d), only batch sharded or fully
  // replicated sharding is supported for now.
  if (!output_layout->IsFullyReplicated() && !output_layout->IsBatchParallel())
    return errors::Unimplemented(
        llvm::formatv(
            "Only replicated or batch parallel layout is supported in Conv "
            "expansion, but get layout : {0}",
            output_layout->ToString())
            .str());

  if (llvm::isa<mlir::TF::Conv2DBackpropInputOp>(op))
    return HandleConvBackpropInput<mlir::TF::Conv2DBackpropInputOp>(
        *output_layout, op);

  if (llvm::isa<mlir::TF::Conv3DBackpropInputV2Op>(op))
    return HandleConvBackpropInput<mlir::TF::Conv3DBackpropInputV2Op>(
        *output_layout, op);

  if (auto backprop_filter =
          mlir::dyn_cast<mlir::TF::Conv2DBackpropFilterOp>(op))
    return HandleConv2dBackPropFilter(*output_layout, backprop_filter);

  if (auto max_pool_grad = mlir::dyn_cast<mlir::TF::MaxPoolGradOp>(op))
    return HandleMaxPoolGradOp(*output_layout, max_pool_grad);

  // Local expansion only for all other ops.
  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> ConvSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());

  if (llvm::isa<mlir::TF::Conv2DOp, mlir::TF::Conv3DOp, mlir::TF::MaxPoolOp,
                mlir::TF::MaxPoolGradOp>(op)) {
    // Conv2d/Conv3d and MaxPool ops are grouped together as they all try to
    // propagate layout from input image (operand 0).

    // If requested 'input' layout exist, try to request same layout for output.
    if (input_layouts.find(0) != input_layouts.end()) {
      output_layouts[0] = input_layouts.lookup(0);
    } else {
      // For MaxPoolGrad, request same layout as 'orig_output' or 'grad'
      // whatever is present.
      if (llvm::isa<mlir::TF::MaxPoolGradOp>(op)) {
        if (input_layouts.find(1) != input_layouts.end())
          output_layouts[0] = input_layouts.lookup(1);
        else if (input_layouts.find(2) != input_layouts.end())
          output_layouts[0] = input_layouts.lookup(2);
      }
    }
  } else if (llvm::isa<mlir::TF::Conv2DBackpropInputOp,
                       mlir::TF::Conv3DBackpropInputV2Op,
                       mlir::TF::Conv2DBackpropFilterOp,
                       mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
    // Conv BackProp ops should usually take layout from gradient for both
    // inputs and filters.

    // 'grad' layout
    if (input_layouts.find(2) != input_layouts.end()) {
      if (llvm::isa<mlir::TF::Conv2DBackpropInputOp,
                    mlir::TF::Conv3DBackpropInputV2Op>(op)) {
        // BackProp ops try to respect layout from gradients for inputs.
        output_layouts[0] = input_layouts.lookup(2);
      }

      // For filters, we currently only try to request a replicated output
      // layout.
      if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp,
                    mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
        output_layouts[0] =
            Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOpResult(0)));
      }
    }
  } else {
    return errors::InvalidArgument(
        llvm::formatv(
            "Layout propagation for unrecognized convolution op {0} not "
            "supported.",
            OpName(op))
            .str());
  }

  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>> ConvSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());

  if (llvm::isa<mlir::TF::Conv2DOp, mlir::TF::Conv3DOp, mlir::TF::MaxPoolOp,
                mlir::TF::MaxPoolGradOp>(op)) {
    // If suggested output layout exists, try to request input image to have the
    // same layout so that all computation would be local.
    if (output_layouts.find(0) != output_layouts.end()) {
      const Layout output_layout = output_layouts.lookup(0);

      input_layouts[0] = output_layout;

      // Request replicated for filter input if Conv2D/Conv3D.
      if (llvm::isa<mlir::TF::Conv2DOp, mlir::TF::Conv3DOp>(op)) {
        input_layouts[1] =
            Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)));
      }
      if (llvm::isa<mlir::TF::MaxPoolGradOp>(op)) {
        input_layouts[1] = output_layout;  // 'orig_output'
        input_layouts[2] = output_layout;  // 'grad'
      }
    }
  } else if (llvm::isa<mlir::TF::Conv2DBackpropInputOp,
                       mlir::TF::Conv3DBackpropInputV2Op,
                       mlir::TF::Conv2DBackpropFilterOp,
                       mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
    // If suggested output layout exists, try to request grad to have output
    // layout.
    if (output_layouts.find(0) != output_layouts.end()) {
      input_layouts[2] = output_layouts.lookup(0);
      // Request inputs and filter_sizes to be replicated.
      input_layouts[0] =
          Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)));
      input_layouts[1] =
          Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)));
    }
  } else {
    return errors::InvalidArgument(
        llvm::formatv(
            "Layout propagation for unrecognized convolution op {0} not "
            "supported.",
            OpName(op))
            .str());
  }

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
