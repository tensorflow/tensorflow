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
#include "tensorflow/core/platform/errors.h"
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

template <typename ConvOp>
Status VerifyConvLayout(const Layout& input_layout, const Layout& filter_layout,
                        ConvOp conv_op) {
  if (!filter_layout.IsFullyReplicated())
    return errors::InvalidArgument(
        "Filter for convolution must have fully replicated layout.");

  // Data format "NCHW" or "NCDHW".
  int channel_dim = 1;
  if (conv_op.getDataFormat() == "NHWC")
    channel_dim = 3;
  else if (conv_op.getDataFormat() == "NDHWC")
    channel_dim = 4;

  if (input_layout.sharding_spec(channel_dim) != Layout::kUnshardedDim)
    return errors::InvalidArgument(
        "Conv input's channel dimension must be replicated.");

  if (input_layout.IsBatchParallel())
    // No further checks needed for replicated case.
    return OkStatus();

  if (conv_op.getPadding() == "EXPLICIT")
    return errors::InvalidArgument(
        "Explicit padding not supported for convolution with spatial "
        "partitions.");

  const int num_non_default_dilations =
      llvm::count_if(conv_op.getDilations(), [](mlir::Attribute dilation) {
        return dilation.cast<mlir::IntegerAttr>().getInt() != 1;
      });
  if (num_non_default_dilations > 0)
    return errors::InvalidArgument(
        "Only dilation rate 1 is supported for convolution with spatial "
        "partitions.");

  // TODO(b/208700444): support convolution with strides greater than 1.
  const int num_non_default_strides =
      llvm::count_if(conv_op.getStrides(), [](mlir::Attribute stride) {
        return stride.cast<mlir::IntegerAttr>().getInt() != 1;
      });
  if (num_non_default_strides > 0)
    return errors::InvalidArgument(
        "Only stride 1 is supported for convolution with spatial partitions.");

  mlir::Value input = conv_op.getInput();
  auto input_type = input.getType().dyn_cast<mlir::RankedTensorType>();
  if (!input_type || !input_type.hasStaticShape())
    return errors::InvalidArgument(
        "Input must have static shapes for convolution with spatial "
        "partitions.");

  mlir::Value filter = conv_op.getFilter();
  auto filter_type = filter.getType().dyn_cast<mlir::RankedTensorType>();
  if (!filter_type || !filter_type.hasStaticShape())
    return errors::InvalidArgument(
        "Filter must have static shapes for convolution with spatial "
        "partitions.");

  llvm::ArrayRef<int64_t> filter_shape = filter_type.getShape();
  for (auto it = filter_shape.begin(); it != filter_shape.end() - 2; ++it) {
    if (*it % 2 != 1)
      return errors::InvalidArgument(
          "Filter dimensions must be odd numbers for convolution with "
          "spatial partitions.");
  }

  return OkStatus();
}

mlir::Value PadInputOnUnshardedDim(mlir::OpBuilder& builder,
                                   mlir::Location location,
                                   mlir::Value input_tensor, int curr_input_dim,
                                   int64_t curr_filter_dim_size) {
  auto input_tensor_type =
      input_tensor.getType().dyn_cast<mlir::RankedTensorType>();
  auto input_tensor_shape = input_tensor_type.getShape();

  const size_t paddings_flat_length = input_tensor_type.getRank() * 2;
  llvm::SmallVector<int64_t, 4> paddings_flat_vec(paddings_flat_length, 0);
  int64_t padding_size = curr_filter_dim_size - 1;
  paddings_flat_vec[2 * curr_input_dim] = padding_size / 2;
  paddings_flat_vec[2 * curr_input_dim + 1] = padding_size / 2;

  llvm::SmallVector<int64_t, 4> paddings_shape(input_tensor_shape.begin(),
                                               input_tensor_shape.end());
  paddings_shape[curr_input_dim] += padding_size;

  mlir::Value paddings_flat = Int64Const(builder, location, paddings_flat_vec);
  mlir::RankedTensorType paddings_type = mlir::RankedTensorType::get(
      paddings_shape, input_tensor_type.getElementType());
  mlir::Value paddings = builder.create<mlir::TF::ReshapeOp>(
      location, paddings_flat,
      Int64Const(builder, location, {input_tensor_type.getRank(), 2}));
  return builder.create<mlir::TF::PadOp>(location, paddings_type, input_tensor,
                                         paddings);
}

template <typename ConvOp>
StatusOr<mlir::Operation*> HandleConv(ConvOp conv_op) {
  mlir::OpBuilder builder(conv_op);
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(conv_op.getInput()));
  TF_ASSIGN_OR_RETURN(const Layout filter_layout,
                      ExtractRequiredLayoutFromOperand(conv_op.getFilter()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(conv_op));

  TF_RETURN_IF_ERROR(VerifyConvLayout(input_layout, filter_layout, conv_op));

  if (input_layout.IsBatchParallel())
    // No special handling needed for replicated case.
    return InferSPMDExpandedLocalShape(conv_op);

  mlir::tf_device::ClusterOp cluster =
      conv_op->template getParentOfType<mlir::tf_device::ClusterOp>();
  TF_ASSIGN_OR_RETURN(mlir::Value mesh_coordinates,
                      GetMeshCoordinatesFromCluster(cluster));
  const Mesh& mesh = input_layout.mesh();
  mlir::Location location = conv_op->getLoc();

  const std::vector<std::string> input_sharding_spec =
      input_layout.sharding_spec_strs();
  const std::vector<std::string> output_sharding_spec =
      output_layout.sharding_spec_strs();
  llvm::StringRef format = conv_op.getDataFormat();
  llvm::StringRef padding = conv_op.getPadding();

  const auto input_num_shards = input_layout.num_shards();
  const auto output_num_shards = output_layout.num_shards();

  auto filter_type =
      conv_op.getFilter().getType().template dyn_cast<mlir::RankedTensorType>();
  auto filter_shape = filter_type.getShape();

  int begin_input_dim = -1, end_input_dim = -1;
  if (format == "NCHW") {
    begin_input_dim = 2;
    end_input_dim = 3;
  } else if (format == "NHWC") {
    begin_input_dim = 1;
    end_input_dim = 2;
  } else if (format == "NCDHW") {
    begin_input_dim = 2;
    end_input_dim = 4;
  } else if (format == "NDHWC") {
    begin_input_dim = 1;
    end_input_dim = 3;
  }

  // For non-batch, non-channel dimension sharding, conduct halo exchange.
  for (int curr_input_dim = begin_input_dim; curr_input_dim <= end_input_dim;
       ++curr_input_dim) {
    int curr_filter_dim = curr_input_dim - begin_input_dim;

    auto input_type = conv_op.getInput()
                          .getType()
                          .template dyn_cast<mlir::RankedTensorType>();
    auto input_shape = input_type.getShape();

    if (input_sharding_spec[curr_input_dim] == Layout::kUnshardedDim) {
      if (padding == "SAME") {
        // Since we always emit a Conv op with "VALID" padding, we need to
        // manually pad the input tensor.
        conv_op->setOperand(
            0, PadInputOnUnshardedDim(builder, location, conv_op.getInput(),
                                      curr_input_dim,
                                      filter_shape[curr_filter_dim]));
      }
      // No halo exchange is needed for unsharded dims.
      continue;
    }

    TF_ASSIGN_OR_RETURN(const int mesh_dim_index,
                        mesh.idx_for_dim(input_sharding_spec[curr_input_dim]));
    TF_ASSIGN_OR_RETURN(mlir::Value scalar_mesh_coordinate,
                        SelectScalarValueFromArray(builder, mesh_dim_index,
                                                   location, mesh_coordinates));

    int halo_size;
    if (padding == "SAME") {
      halo_size = std::floor(filter_shape[curr_filter_dim] / 2);
    } else if (padding == "VALID") {
      int input_local_size = input_shape[curr_input_dim];
      int input_size = input_local_size * input_num_shards[curr_input_dim];
      int output_size = input_size - (filter_shape[curr_filter_dim] - 1);
      int output_local_size = output_size / output_num_shards[curr_input_dim];
      halo_size = output_local_size + (filter_shape[curr_filter_dim] - 1) -
                  input_local_size;
    } else {
      return errors::Unimplemented(
          "Spatially partitioned convolution with padding \"", padding.str(),
          "\" is not supported.");
    }

    if (halo_size == 0)
      // No exchange is needed for empty halos.
      continue;

    builder.setInsertionPoint(conv_op);
    TF_ASSIGN_OR_RETURN(
        mlir::Value halo_exchanged_input,
        EmitHaloExchange(builder, halo_size,
                         input_sharding_spec[curr_input_dim], input_layout,
                         mesh_coordinates, cluster, location,
                         conv_op.getInput()));

    if (padding == "SAME") {
      conv_op->setOperand(0, halo_exchanged_input);
    } else if (padding == "VALID") {
      // Slice the halo exchanged tensor to the desired size based on the index
      // of the shard on the current dimension.

      llvm::SmallVector<int32_t, 4> halo_sizes(input_layout.rank(), 0);
      halo_sizes[curr_input_dim] = halo_size;
      mlir::Value halo_sizes_const = IntConst(builder, location, halo_sizes);

      llvm::SmallVector<int32_t, 4> halo_increments(input_layout.rank(), 0);
      halo_increments[curr_input_dim] =
          halo_size / (input_num_shards[curr_input_dim] - 1);
      mlir::Value halo_increments_const =
          IntConst(builder, location, halo_increments);

      mlir::Value offset = builder.create<mlir::TF::MulOp>(
          location, halo_increments_const.getType(), scalar_mesh_coordinate,
          halo_increments_const);
      mlir::Value slice_begin =
          builder.create<mlir::TF::SubOp>(location, halo_sizes_const, offset);

      llvm::SmallVector<int64_t, 4> slice_size(input_shape.begin(),
                                               input_shape.end());
      slice_size[curr_input_dim] += halo_size;
      mlir::Value slice_size_const = Int64Const(builder, location, slice_size);

      mlir::RankedTensorType sliced_input_type =
          mlir::RankedTensorType::get(slice_size, input_type.getElementType());
      mlir::Value sliced_input = builder.create<mlir::TF::SliceOp>(
          location, sliced_input_type, /*input=*/halo_exchanged_input,
          /*begin=*/slice_begin, /*size=*/slice_size_const);
      conv_op->setOperand(0, sliced_input);
    }

    // Spatially partitioned convolution always uses VALID padding after halo
    // exchange.
    conv_op.setPaddingAttr(builder.getStringAttr("VALID"));
  }

  return InferSPMDExpandedLocalShape(conv_op);
}

template <typename ConvBackpropInputOp>
StatusOr<mlir::Operation*> HandleConvBackpropInput(
    const Layout& output_layout, ConvBackpropInputOp conv_op) {
  llvm::SmallVector<int64_t, 4> global_shape;
  Status extract_status =
      ExtractConstVectorFromValue(conv_op.getInputSizes(), &global_shape);

  // Recover local shape in SPMD expansion.
  if (extract_status.ok()) {
    auto local_shape = output_layout.LocalShapeFromGlobalShape(global_shape);
    mlir::OpBuilder builder(conv_op->getBlock(), conv_op->getBlock()->begin());
    auto new_const = IntConst(
        builder, conv_op->getLoc(),
        llvm::SmallVector<int32_t, 4>(local_shape.begin(), local_shape.end()));
    conv_op.getInputSizesMutable().assign(new_const);
  }

  return InferSPMDExpandedLocalShape(conv_op);
}

template <typename ConvBackpropFilterOp>
StatusOr<mlir::Operation*> HandleConvBackpropFilter(
    const Layout& output_layout, ConvBackpropFilterOp conv_op) {
  TF_ASSIGN_OR_RETURN(Layout input_layout,
                      ExtractRequiredLayoutFromOperand(conv_op.getInput()));

  TF_ASSIGN_OR_RETURN(
      Layout out_backprop_layout,
      ExtractRequiredLayoutFromOperand((conv_op.getOutBackprop())));
  // Perform a split on batch dimension so that the each local device performs
  // local operation.
  // TODO(hthu): Make this work on input with rank higher than 4.
  if (input_layout.IsBatchParallel()) {
    mlir::OpBuilder builder(conv_op);
    if (out_backprop_layout.IsFullyReplicated()) {
      TF_ASSIGN_OR_RETURN(const mlir::Value batch_sharded,
                          EmitAllScatter(builder, conv_op.getOutBackprop(),
                                         out_backprop_layout, input_layout));
      conv_op.getOutBackpropMutable().assign(batch_sharded);
    }

    // Perform all reduce over batch dim.
    builder.setInsertionPointAfter(conv_op);
    return DT_CTX(EmitAllReduce(builder, output_layout,
                                {input_layout.sharding_spec(0)}, conv_op,
                                kReduceOpAdd));
  } else {
    return errors::InvalidArgument(
        "Convolution backprop for spatially partitioned input not supported.");
  }
  return InferSPMDExpandedLocalShape(conv_op);
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

  TF_ASSIGN_OR_RETURN(const auto output_layout, ExtractSingleLayoutFromOp(op));

  // Forward prop ops.
  if (llvm::isa<mlir::TF::Conv2DOp>(op))
    return HandleConv<>(llvm::cast<mlir::TF::Conv2DOp>(op));
  if (llvm::isa<mlir::TF::Conv3DOp>(op))
    return HandleConv<>(llvm::cast<mlir::TF::Conv3DOp>(op));

  // Backward prop input ops.
  if (llvm::isa<mlir::TF::Conv2DBackpropInputOp>(op))
    return HandleConvBackpropInput<>(
        *output_layout, llvm::cast<mlir::TF::Conv2DBackpropInputOp>(op));
  if (llvm::isa<mlir::TF::Conv3DBackpropInputV2Op>(op))
    return HandleConvBackpropInput<>(
        *output_layout, llvm::cast<mlir::TF::Conv3DBackpropInputV2Op>(op));

  // Backward prop filter ops.
  if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp>(op))
    return HandleConvBackpropFilter<>(
        *output_layout, llvm::cast<mlir::TF::Conv2DBackpropFilterOp>(op));
  if (llvm::isa<mlir::TF::Conv3DBackpropFilterV2Op>(op))
    return HandleConvBackpropFilter<>(
        *output_layout, llvm::cast<mlir::TF::Conv3DBackpropFilterV2Op>(op));

  // For all other ops, only batch sharded or fully replicated sharding is
  // supported for now.
  if (!output_layout->IsFullyReplicated() && !output_layout->IsBatchParallel())
    return errors::Unimplemented(
        llvm::formatv(
            "Only replicated or batch parallel layout is supported in "
            "expansion of {0}, but got output layout: {1}",
            op->getName().getStringRef().str(), output_layout->ToString())
            .str());

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
