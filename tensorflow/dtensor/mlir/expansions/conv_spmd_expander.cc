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

#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/ValueRange.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

template <typename ConvOp>
absl::Status VerifyConvLayout(const Layout& input_layout,
                              const Layout& filter_layout, ConvOp conv_op) {
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

  if (input_layout.IsBatchParallel() || input_layout.IsFullyReplicated())
    // No further checks needed for replicated case.
    return absl::OkStatus();

  if (conv_op.getPadding() == "EXPLICIT")
    return errors::InvalidArgument(
        "Explicit padding not supported for convolution with spatial "
        "partitions.");

  const int num_non_default_dilations =
      llvm::count_if(conv_op.getDilations(), [](mlir::Attribute dilation) {
        return mlir::cast<mlir::IntegerAttr>(dilation).getInt() != 1;
      });
  if (num_non_default_dilations > 0)
    return errors::InvalidArgument(
        "Only dilation rate 1 is supported for convolution with spatial "
        "partitions.");

  // TODO(b/208700444): support convolution with strides greater than 1.
  const int num_non_default_strides =
      llvm::count_if(conv_op.getStrides(), [](mlir::Attribute stride) {
        return mlir::cast<mlir::IntegerAttr>(stride).getInt() != 1;
      });
  if (num_non_default_strides > 0)
    return errors::InvalidArgument(
        "Only stride 1 is supported for convolution with spatial partitions.");

  mlir::Value input = conv_op.getInput();
  auto input_type = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!input_type || !input_type.hasStaticShape())
    return errors::InvalidArgument(
        "Input must have static shapes for convolution with spatial "
        "partitions.");

  mlir::Value filter = conv_op.getFilter();
  auto filter_type = mlir::dyn_cast<mlir::RankedTensorType>(filter.getType());
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

  return absl::OkStatus();
}

mlir::Value PadInputOnUnshardedDim(mlir::OpBuilder& builder,
                                   mlir::Location location,
                                   mlir::Value input_tensor, int curr_input_dim,
                                   int64_t curr_filter_dim_size) {
  auto input_tensor_type =
      mlir::dyn_cast<mlir::RankedTensorType>(input_tensor.getType());
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

  if (input_layout.IsBatchParallel() || input_layout.IsFullyReplicated())
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
      mlir::dyn_cast<mlir::RankedTensorType>(conv_op.getFilter().getType());
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

    auto input_type =
        mlir::dyn_cast<mlir::RankedTensorType>(conv_op.getInput().getType());
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
      // slice_size_const and slize_begin_int64 has to be same type.
      mlir::Value slice_begin_int64 = builder.create<mlir::TF::CastOp>(
          location,
          mlir::RankedTensorType::get({input_layout.rank()},
                                      builder.getI64Type()),
          slice_begin);

      mlir::RankedTensorType sliced_input_type =
          mlir::RankedTensorType::get(slice_size, input_type.getElementType());
      mlir::Value sliced_input = builder.create<mlir::TF::SliceOp>(
          location, sliced_input_type, /*input=*/halo_exchanged_input,
          /*begin=*/slice_begin_int64, /*size=*/slice_size_const);
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
  TF_ASSIGN_OR_RETURN(std::vector<Layout> input_layouts,
                      ExtractRequiredLayoutFromOperands(conv_op));

  const Layout& input_shape_layout = input_layouts[0];
  const Layout& filter_layout = input_layouts[1];
  const Layout& grad_layout = input_layouts[2];

  // We only support batch sharding for these. In this case the output and input
  // gradient must both be batch sharded. The filter input must be replicated.
  if (!(output_layout.IsBatchParallel() || output_layout.IsFullyReplicated()) ||
      !(grad_layout.IsBatchParallel() || grad_layout.IsFullyReplicated())) {
    return errors::InvalidArgument("{0} only supports batch parallel layouts.",
                                   conv_op->getName().getStringRef().str());
  }
  if (!filter_layout.IsFullyReplicated()) {
    return errors::InvalidArgument("{0} only supports replicated filters.",
                                   conv_op->getName().getStringRef().str());
  }
  if (!input_shape_layout.IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Layout of the input shape (parameter 0) of {0} must be replicated.",
        conv_op->getName().getStringRef().str());
  }

  llvm::SmallVector<int64_t, 4> global_shape;
  absl::Status extract_status =
      ExtractConstVectorFromValue(conv_op.getInputSizes(), &global_shape);

  // If the input is dynamic size, we expect the output is all so dynamic size
  // since they should roughly be the same shape. Don't support this for right
  // now. The easy way to support this is to all gather the gradient input and
  // compute this as a large local convolution and then slice to the output
  // layout.
  if (!extract_status.ok()) {
    return errors::InvalidArgument("{0} requires static shape for input size.",
                                   conv_op->getName().getStringRef().str());
  }

  // Compute the 'true' input/output layout of the operation. E.g. batch sharded
  // vs non-batch sharded. If at least one of the the input gradient or output
  // gradient is batch sharded, use that dimension.
  string batch_sharding_dimension = grad_layout.sharding_spec(0);
  if (batch_sharding_dimension == Layout::kUnshardedDim) {
    batch_sharding_dimension = output_layout.sharding_spec(0);
  } else if ((output_layout.sharding_spec(0) != Layout::kUnshardedDim) &&
             (batch_sharding_dimension != output_layout.sharding_spec(0))) {
    return errors::InvalidArgument(
        "Input and output layout to {2} have incompatible sharding dimensions: "
        "\"{0}\" and \"{1}\".",
        grad_layout.sharding_spec(0), output_layout.sharding_spec(0),
        conv_op->getName().getStringRef().str());
  }

  const Layout desired_input_gradient_layout =
      Layout::BatchShardedLike(grad_layout, batch_sharding_dimension);
  const Layout desired_output_gradient_layout =
      Layout::BatchShardedLike(output_layout, batch_sharding_dimension);

  const Layout desired_input_layout = Layout::BatchShardedOnMesh(
      grad_layout.mesh(), global_shape.size(), batch_sharding_dimension);
  const std::vector<int64_t> local_shape =
      desired_input_layout.LocalShapeFromGlobalShape(global_shape);

  mlir::OpBuilder builder(conv_op.getOperation());
  mlir::Value new_const = IntConst(
      builder, conv_op->getLoc(),
      llvm::SmallVector<int32_t, 4>(local_shape.begin(), local_shape.end()));
  conv_op.getInputSizesMutable().assign(new_const);

  TF_ASSIGN_OR_RETURN(mlir::Value local_input_gradient,
                      EmitRelayout(conv_op.getOutBackprop(), grad_layout,
                                   desired_input_gradient_layout));
  conv_op.getOutBackpropMutable().assign(local_input_gradient);

  InferSPMDExpandedLocalShape(conv_op);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  TF_ASSIGN_OR_RETURN(
      mlir::Value local_output_gradient,
      EmitRelayout(conv_op.getOutput(), desired_output_gradient_layout,
                   output_layout, &newly_created_ops));
  conv_op.getOutput().replaceAllUsesExcept(local_output_gradient,
                                           newly_created_ops);
  return local_output_gradient.getDefiningOp();
}

// This expands backprop ops which take tensor inputs into those which take
// sizes. We first convert the (currently local) input shape to global and use
// the const rather than input. The shape will be converted back to local in
// HandleConvBackpropInput, but this is the correct behavior as
// HandleConvBackpropInput will decided how it wants to expand the op.
template <typename To, typename From>
StatusOr<mlir::Operation*> HandleConvBackpropInputTensor(
    const Layout& output_layout, From conv_op) {
  TF_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> local_shape,
                      GetTFShapeFromType(conv_op.getInput().getType()));

  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(conv_op.getInput()));

  const std::vector<int64_t> global_shape =
      input_layout.GlobalShapeFromLocalShape(local_shape);

  mlir::OpBuilder builder(conv_op);
  mlir::Value global_input_shape = IntConst(
      builder, conv_op->getLoc(),
      llvm::SmallVector<int32_t, 4>(global_shape.begin(), global_shape.end()));

  // Insert a replicated layout along this edge, so that we can call
  // HandleConvBackpropInput which expects there to be a layout here.
  mlir::TF::ShapeAttr global_input_shape_shape = mlir::TF::ShapeAttr::get(
      builder.getContext(),
      mlir::cast<mlir::TensorType>(global_input_shape.getType()));
  mlir::TF::DTensorLayout global_input_shape_with_layout =
      builder.create<mlir::TF::DTensorLayout>(
          conv_op->getLoc(), global_input_shape,
          mlir::dtensor::LayoutAttr::get(
              builder.getContext(),
              Layout::ReplicatedOnMesh(input_layout.mesh(), 1)),
          global_input_shape_shape);

  To new_conv = builder.create<To>(
      conv_op->getLoc(), conv_op->getResultTypes(),
      mlir::ValueRange({global_input_shape_with_layout, conv_op.getFilter(),
                        conv_op.getOutBackprop()}),
      conv_op->getAttrs());

  conv_op.getOutput().replaceAllUsesWith(new_conv.getOutput());
  conv_op.erase();

  return HandleConvBackpropInput(output_layout, new_conv);
}

template <typename ConvBackpropFilterOp>
StatusOr<mlir::Operation*> HandleConvBackpropFilter(
    const Layout& output_layout, ConvBackpropFilterOp conv_op) {
  TF_ASSIGN_OR_RETURN(std::vector<Layout> input_layouts,
                      ExtractRequiredLayoutFromOperands(conv_op));

  const Layout& input_layout = input_layouts[0];
  const Layout& filter_shape_layout = input_layouts[1];
  const Layout& grad_layout = input_layouts[2];

  // We only support batch sharding for these. In this case the input
  // activations and input gradient should both be batch sharded and
  // the output (the filter gradient) should be replicated.
  if (!(output_layout.IsBatchParallel() || output_layout.IsFullyReplicated()) ||
      !(grad_layout.IsBatchParallel() || grad_layout.IsFullyReplicated())) {
    return errors::InvalidArgument("{0} only supports batch parallel layouts.",
                                   conv_op->getName().getStringRef().str());
  }
  if (!output_layout.IsFullyReplicated()) {
    return errors::InvalidArgument("{0} only supports replicated filters.",
                                   conv_op->getName().getStringRef().str());
  }
  if (!filter_shape_layout.IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Filter shape input (parameter 1) for {0} must have replicated layout.",
        conv_op->getName().getStringRef().str());
  }

  // Compute the 'true' input layouts of the operation. E.g. batch sharded
  // vs non-batch sharded. Basically we get the batch sharding dimension from
  // one of the inputs and check that the other is potentially sharded on the
  // same dimension.
  // TODO(b/262417847): if batch_sharding_dimension is Layout::kUnsharded, then
  // we should consider sharding the input here. It may be faster to spread
  // the convolution out and then all reduce after vs running it all locally.
  string batch_sharding_dimension = input_layout.sharding_spec(0);
  if (batch_sharding_dimension == Layout::kUnshardedDim) {
    batch_sharding_dimension = grad_layout.sharding_spec(0);
  } else if ((grad_layout.sharding_spec(0) != Layout::kUnshardedDim) &&
             (batch_sharding_dimension != grad_layout.sharding_spec(0))) {
    return errors::InvalidArgument(
        "Input and gradient layouts for {2} have incompatible batch sharding "
        "dimensions: \"{0}\" and \"{1}\".",
        input_layouts[0].sharding_spec(0), input_layouts[0].sharding_spec(0),
        conv_op->getName().getStringRef().str());
  }

  const Layout desired_input_activation_layout =
      Layout::BatchShardedLike(input_layout, batch_sharding_dimension);
  const Layout desired_input_gradient_layout =
      Layout::BatchShardedLike(grad_layout, batch_sharding_dimension);

  TF_ASSIGN_OR_RETURN(mlir::Value local_input_activation,
                      EmitRelayout(conv_op.getInput(), input_layout,
                                   desired_input_activation_layout));
  conv_op.getInputMutable().assign(local_input_activation);

  TF_ASSIGN_OR_RETURN(mlir::Value local_input_gradient,
                      EmitRelayout(conv_op.getOutBackprop(), grad_layout,
                                   desired_input_gradient_layout));
  conv_op.getOutBackpropMutable().assign(local_input_gradient);

  InferSPMDExpandedLocalShape(conv_op);

  // Output shall be replicated. If we were batch sharded, we need to
  // all-reduce the partial results.

  if (batch_sharding_dimension != Layout::kUnshardedDim) {
    mlir::OpBuilder builder(conv_op.getOperation());
    builder.setInsertionPointAfter(conv_op);
    return DT_CTX(EmitAllReduce(builder, output_layout,
                                {batch_sharding_dimension}, conv_op,
                                kReduceOpAdd));
  }

  return conv_op.getOperation();
}

// This expands backprop ops which take tensor inputs into those which take
// sizes. We check that the filter input shape is global and then make that a
// const, and replace the op with the version taking shapes.
template <typename To, typename From>
StatusOr<mlir::Operation*> HandleConvBackpropFilterTensor(
    const Layout& output_layout, From conv_op) {
  TF_ASSIGN_OR_RETURN(const Layout filter_layout,
                      ExtractRequiredLayoutFromOperand(conv_op.getFilter()));

  if (!filter_layout.IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Convolution backpropation ops only support replicated filters.");
  }

  TF_ASSIGN_OR_RETURN(llvm::SmallVector<int64_t> global_filter_shape,
                      GetTFShapeFromType(conv_op.getFilter().getType()));

  mlir::OpBuilder builder(conv_op);
  mlir::Value global_filter_shape_const =
      IntConst(builder, conv_op->getLoc(),
               llvm::SmallVector<int32_t, 4>(global_filter_shape.begin(),
                                             global_filter_shape.end()));

  // Insert a replicated layout along this edge, so that we can call
  // HandleConvBackpropInput which expects there to be a layout here.
  mlir::TF::ShapeAttr global_filter_shape_shape = mlir::TF::ShapeAttr::get(
      builder.getContext(),
      mlir::cast<mlir::TensorType>(global_filter_shape_const.getType()));
  mlir::TF::DTensorLayout global_filter_shape_with_layout =
      builder.create<mlir::TF::DTensorLayout>(
          conv_op->getLoc(), global_filter_shape_const,
          mlir::dtensor::LayoutAttr::get(
              builder.getContext(),
              Layout::ReplicatedOnMesh(filter_layout.mesh(), 1)),
          global_filter_shape_shape);

  To new_conv = builder.create<To>(
      conv_op->getLoc(), conv_op->getResultTypes(),
      mlir::ValueRange({conv_op.getInput(), global_filter_shape_with_layout,
                        conv_op.getOutBackprop()}),
      conv_op->getAttrs());

  conv_op.getOutput().replaceAllUsesWith(new_conv.getOutput());
  conv_op.erase();

  return HandleConvBackpropFilter(output_layout, new_conv);
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
  if (auto conv_op = llvm::dyn_cast<mlir::TF::Conv2DBackpropInputV2Op>(op))
    return HandleConvBackpropInputTensor<mlir::TF::Conv2DBackpropInputOp>(
        *output_layout, conv_op);
  if (auto conv_op = llvm::dyn_cast<mlir::TF::Conv3DBackpropInputOp>(op))
    return HandleConvBackpropInputTensor<mlir::TF::Conv3DBackpropInputV2Op>(
        *output_layout, conv_op);
  if (llvm::isa<mlir::TF::Conv3DBackpropInputV2Op>(op))
    return HandleConvBackpropInput<>(
        *output_layout, llvm::cast<mlir::TF::Conv3DBackpropInputV2Op>(op));

  // Backward prop filter ops.
  if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp>(op))
    return HandleConvBackpropFilter<>(
        *output_layout, llvm::cast<mlir::TF::Conv2DBackpropFilterOp>(op));
  if (auto conv_op = llvm::dyn_cast<mlir::TF::Conv2DBackpropFilterV2Op>(op))
    return HandleConvBackpropFilterTensor<mlir::TF::Conv2DBackpropFilterOp>(
        *output_layout, conv_op);
  if (auto conv_op = llvm::dyn_cast<mlir::TF::Conv3DBackpropFilterOp>(op))
    return HandleConvBackpropFilterTensor<mlir::TF::Conv3DBackpropFilterV2Op>(
        *output_layout, conv_op);
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
                       mlir::TF::Conv2DBackpropInputV2Op,
                       mlir::TF::Conv3DBackpropInputOp,
                       mlir::TF::Conv3DBackpropInputV2Op>(op)) {
    if (llvm::isa<mlir::TF::Conv2DBackpropInputOp,
                  mlir::TF::Conv3DBackpropInputV2Op>(op)) {
      if (input_layouts.find(2) != input_layouts.end()) {
        // The propagate the gradient layout to the new gradient, e.g. respect
        // the spatial partitioning of the input gradient.
        output_layouts[0] = input_layouts.lookup(2);
      }
    } else {
      if (input_layouts.find(0) != input_layouts.end()) {
        // The propagate the gradient layout to the new gradient, e.g. respect
        // the spatial partitioning of the input.
        output_layouts[0] = input_layouts.lookup(0);
      }
    }
  } else if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp,
                       mlir::TF::Conv2DBackpropFilterV2Op,
                       mlir::TF::Conv3DBackpropFilterOp,
                       mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
    if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp,
                  mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
      // For the ops which take filter shape as input, just return a replicated
      // output shape.
      output_layouts[0] =
          Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOpResult(0)));
    } else if (input_layouts.find(1) != input_layouts.end()) {
      // For the ops taking a real filter, just copy the filter layout.
      output_layouts[0] = input_layouts.lookup(1);
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
                       mlir::TF::Conv2DBackpropInputV2Op,
                       mlir::TF::Conv3DBackpropInputOp,
                       mlir::TF::Conv3DBackpropInputV2Op>(op)) {
    // Generally mark the filter as replicated.
    input_layouts[1] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)));
    if (llvm::isa<mlir::TF::Conv2DBackpropInputOp,
                  mlir::TF::Conv3DBackpropInputV2Op>(op)) {
      // This input is a shape.
      input_layouts[0] =
          Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)));
    }

    if (output_layouts.find(0) != output_layouts.end()) {
      Layout output_layout = output_layouts.lookup(0);
      // Ask for the grad to have the same layout as the output. The reasoning
      // here is that the if the output is spatially partitioned, we expect
      // that the grad is spatially partitioned as well.
      input_layouts[2] = output_layout;
      if (llvm::isa<mlir::TF::Conv2DBackpropInputV2Op,
                    mlir::TF::Conv3DBackpropInputOp>(op)) {
        input_layouts[0] = output_layout;
      }
    }
  } else if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp,
                       mlir::TF::Conv2DBackpropFilterV2Op,
                       mlir::TF::Conv3DBackpropFilterOp,
                       mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
    // Note for Filter op, we generally expect that the output layout would
    // match the variable layout for the filter which is generally replicated.
    // The gradient layout most likely needs to agree with the input layout,
    // e.g. both spatially partitioned or not. This is somewhat similar to
    // MatMul, for now just set both to replicated.

    input_layouts[0] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)));
    input_layouts[2] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(2)));

    if (llvm::isa<mlir::TF::Conv2DBackpropFilterOp,
                  mlir::TF::Conv3DBackpropFilterV2Op>(op)) {
      // For ops taking filter shape as input, just use a replicated input
      // layout.
      input_layouts[1] =
          Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)));
    } else if (output_layouts.find(0) != output_layouts.end()) {
      // For ops taking filer directly as input copy the output layout to the
      // filter layout.
      input_layouts[1] = output_layouts.lookup(0);
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
