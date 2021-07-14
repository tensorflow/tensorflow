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

#include <cstdint>
#include <iostream>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/tpu_rewrite_device_util.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TFTPU {

namespace {

constexpr char kDeviceAttr[] = "device";
typedef std::pair<TF::Conv2DOp, int64_t> Conv2DWithBlockSize;

struct BlockArgumentInfo {
  unsigned arg_num;
  unsigned num_users;
};

// TODO(wangtao): add a pass to check if it is profitable to space to depth
// transform and invoke the transform if it is needed.
struct TPUSpaceToDepthPass
    : public TF::TPUSpaceToDepthPassBase<TPUSpaceToDepthPass> {
  void runOnOperation() override;
};

// Updates func argument type to have the updated input shape.
void UpdateFuncType(FuncOp func) {
  auto arg_types = func.front().getArgumentTypes();
  auto result_types = func.front().getTerminator()->getOperandTypes();
  func.setType(FunctionType::get(func.getContext(), arg_types, result_types));
}

void HandleFuncOp(Operation* op) {
  auto func = llvm::cast<FuncOp>(op);
  UpdateFuncType(func);
}

// Handles cast op between the first convolution and the block argument.
LogicalResult HandleCast(TF::CastOp cast_op, ArrayRef<int64_t> new_shape) {
  auto cast_input = cast_op.x();
  // Update input type.
  auto transform_result_type =
      RankedTensorType::get(new_shape, getElementTypeOrSelf(cast_input));
  cast_input.setType(transform_result_type);
  auto block_arg = cast_input.dyn_cast<mlir::BlockArgument>();
  auto cast_op_input = dyn_cast_or_null<TF::CastOp>(cast_input.getDefiningOp());
  while (block_arg || cast_op_input) {
    if (block_arg) {
      // Change on device function type/shape.
      HandleFuncOp(block_arg.getOwner()->getParentOp());
      block_arg = nullptr;
      cast_op_input = nullptr;
    } else {
      auto cast_input = cast_op_input.x();
      // Update input type.
      auto transform_result_type =
          RankedTensorType::get(new_shape, getElementTypeOrSelf(cast_input));
      cast_input.setType(transform_result_type);
      // Update block arg and cast_op_input.
      block_arg = cast_input.dyn_cast<mlir::BlockArgument>();
      cast_op_input = dyn_cast_or_null<TF::CastOp>(cast_input.getDefiningOp());
    }
  }
  return success();
}

// Handles padding before convolution for space to depth transform.
LogicalResult HandlePad(TF::PadOp op, int32_t kernel_size, int32_t block_size) {
  auto ranked_type = op.input().getType().dyn_cast<RankedTensorType>();
  if (!ranked_type) return failure();
  auto pad_input_shape = ranked_type.getShape();
  Location loc = op.getLoc();
  OpBuilder builder(op);
  builder.setInsertionPoint(op);
  auto padding_type = RankedTensorType::get({4, 2}, builder.getIntegerType(32));

  // Calculate paddings.
  int32_t pad_total = kernel_size - 1;
  int32_t pad_beg = (pad_total / 2 + 1) / block_size;
  int32_t pad_end = (pad_total / 2) / block_size;
  SmallVector<int32_t, 8> values = {0,       0,       pad_beg, pad_end,
                                    pad_beg, pad_end, 0,       0};
  auto paddings = DenseIntElementsAttr::get(padding_type, values);
  // Update pad_op paddings.
  op.setOperand(1, builder.create<TF::ConstOp>(loc, paddings));

  // Set input type.
  auto input = op.getOperand(0);
  SmallVector<int64_t, 4> transform_shape = {
      pad_input_shape[0], pad_input_shape[1] / block_size,
      pad_input_shape[2] / block_size,
      pad_input_shape[3] * block_size * block_size};
  // Input of the pad op could be a cast op.
  if (auto cast_op = dyn_cast_or_null<TF::CastOp>(input.getDefiningOp()))
    if (failed(HandleCast(cast_op, transform_shape))) return failure();

  auto transform_result_type =
      RankedTensorType::get(transform_shape, getElementTypeOrSelf(input));
  input.setType(transform_result_type);
  op.setOperand(0, input);
  return success();
}

// Handles stride for the first convolution for the transform.
void HandleConv2DStride(TF::Conv2DOp conv2d) {
  MLIRContext* context = conv2d.getContext();
  SmallVector<int64_t, 4> values = {1, 1, 1, 1};
  auto attrs = llvm::map_range(values, [context](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(context, 64), v);
  });
  // TODO(b/157276506): change type of strides to DenseElementsAttr
  auto strides = ArrayAttr::get(context, llvm::to_vector<4>(attrs));
  conv2d->setAttr("strides", strides);
}

// Transforms input shape for the first convolution.
void HandleConv2DInput(TF::Conv2DOp conv2d, int64_t block_size) {
  auto input = conv2d.input();
  auto input_shape = input.getType().cast<RankedTensorType>().getShape();
  SmallVector<int64_t, 4> transform_shape = {
      input_shape[0], input_shape[1] / block_size, input_shape[2] / block_size,
      input_shape[3] * block_size * block_size};
  auto transform_result_type =
      RankedTensorType::get(transform_shape, getElementTypeOrSelf(input));
  input.setType(transform_result_type);
}

// Adds padding for convolution filter for space to depth transform.
TF::PadOp GetPadOpForConv2DFilter(ArrayRef<int64_t> filter_shape, Value filter,
                                  OpBuilder* builder, int32_t pad_h,
                                  int32_t pad_w) {
  SmallVector<int32_t, 8> values = {pad_h, 0, pad_w, 0, 0, 0, 0, 0};
  auto padding_type =
      RankedTensorType::get({4, 2}, builder->getIntegerType(32));
  auto paddings = DenseIntElementsAttr::get(padding_type, values);
  auto paddings_value = builder->create<TF::ConstOp>(filter.getLoc(), paddings);
  std::vector<int64_t> pad_shape = {filter_shape[0] + pad_h,
                                    filter_shape[1] + pad_w, filter_shape[2],
                                    filter_shape[3]};
  SmallVector<int64_t, 4> expand_shape(pad_shape.begin(), pad_shape.end());

  auto expand_result_type =
      RankedTensorType::get(expand_shape, getElementTypeOrSelf(filter));
  return builder->create<TF::PadOp>(filter.getLoc(), expand_result_type, filter,
                                    paddings_value);
}

// Creates reshape op for space to depth transform.
TF::ReshapeOp GetReshapeOpForConv2DFilter(ArrayRef<int64_t> new_shape,
                                          Value input, OpBuilder* builder) {
  auto reshape_result_type =
      RankedTensorType::get(new_shape, getElementTypeOrSelf(input));
  auto reshape_type = RankedTensorType::get(
      {static_cast<int64_t>(new_shape.size())}, builder->getIntegerType(64));
  auto reshape_sizes = DenseIntElementsAttr::get(reshape_type, new_shape);
  auto reshape_value =
      builder->create<TF::ConstOp>(input.getLoc(), reshape_sizes);
  return builder->create<TF::ReshapeOp>(input.getLoc(), reshape_result_type,
                                        input, reshape_value);
}

// Creates transpose op for shape to depth transform.
TF::TransposeOp GetTransposeOpForConv2DFilter(OpBuilder* builder, Value input) {
  SmallVector<int32_t, 6> permutation = {0, 2, 1, 3, 4, 5};
  auto permute_type = RankedTensorType::get({6}, builder->getIntegerType(32));
  auto permute_attr = DenseIntElementsAttr::get(permute_type, permutation);
  auto permute_value =
      builder->create<TF::ConstOp>(input.getLoc(), permute_attr);
  return builder->create<TF::TransposeOp>(input.getLoc(), input, permute_value);
}

void HandleConv2DFilter(TF::Conv2DOp conv2d, int64_t block_size) {
  // For example, if filter shape is [7, 7, 3, 64] with block_size 2,
  // will apply below transforms to the filter:
  // 1. Pad the filter to [8, 8, 3, 64]
  // 2. Reshape to [4, 2, 4, 2, 3, 64]
  // 3. Transpose to [4, 4, 2, 2, 3, 64]
  // 4. Reshape to [4, 4, 12, 64]
  auto filter = conv2d.filter();
  OpBuilder builder(conv2d);
  builder.setInsertionPoint(conv2d);
  // Book keeping filter information.
  auto filter_shape = filter.getType().cast<RankedTensorType>().getShape();
  int64_t height = filter_shape[0];
  int64_t width = filter_shape[1];
  int64_t channel = filter_shape[2];
  int64_t out_channel = filter_shape[3];
  // Value/Op before reshape op.
  Value before_reshape_value = filter;
  if (height % block_size != 0 || width % block_size != 0) {
    // Calculate paddings for height and width.
    int32_t pad_h = block_size - height % block_size;
    int32_t pad_w = block_size - width % block_size;
    auto pad_op =
        GetPadOpForConv2DFilter(filter_shape, filter, &builder, pad_h, pad_w);
    // Update op, height and width before reshape.
    before_reshape_value = pad_op;
    height = height + pad_h;
    width = width + pad_w;
  }

  // Reshape.
  SmallVector<int64_t, 6> new_shape = {
      height / block_size, block_size, width / block_size,
      block_size,          channel,    out_channel};
  auto reshape_op =
      GetReshapeOpForConv2DFilter(new_shape, before_reshape_value, &builder);

  // Transpose.
  auto transpose_op = GetTransposeOpForConv2DFilter(&builder, reshape_op);

  // Reshape Back.
  SmallVector<int64_t, 4> final_shape = {
      height / block_size, width / block_size,
      channel * block_size * block_size, out_channel};
  auto final_reshape_op =
      GetReshapeOpForConv2DFilter(final_shape, transpose_op, &builder);
  // Update filter of Conv2D.
  conv2d.setOperand(1, final_reshape_op);
}

// Creates slice op for filter in back prop pass.
TF::SliceOp GetSliceOpForConv2DBackPropFilter(
    ArrayRef<int32_t> old_filter_shape, Value input, OpBuilder* builder) {
  SmallVector<int64_t, 4> slice_size(old_filter_shape.begin(),
                                     old_filter_shape.end());
  auto slice_result_type =
      RankedTensorType::get(slice_size, getElementTypeOrSelf(input));
  auto slice_size_op = builder->create<TF::ConstOp>(
      input.getLoc(),
      DenseIntElementsAttr::get(
          RankedTensorType::get({4}, builder->getIntegerType(32)),
          old_filter_shape));
  SmallVector<int64_t, 4> slice_start_position = {0, 0, 0, 0};
  auto start_position_type =
      RankedTensorType::get({4}, builder->getIntegerType(64));
  auto start_position = builder->create<TF::ConstOp>(
      input.getLoc(),
      DenseIntElementsAttr::get(start_position_type, slice_start_position));
  return builder->create<TF::SliceOp>(input.getLoc(), slice_result_type, input,
                                      start_position, slice_size_op);
}

// Transforms Conv2DBackPropFilter for space to depth.
void HandleConv2DBackPropFilter(TF::Conv2DBackpropFilterOp backprop,
                                ArrayRef<int32_t> old_filter_shape,
                                ArrayRef<int32_t> new_filter_shape,
                                int64_t block_size) {
  OpBuilder builder(backprop);
  builder.setInsertionPoint(backprop);

  auto input = backprop.input();
  // Get new filter size from new_filter_shape.
  auto new_filter_sizes = builder.create<TF::ConstOp>(
      backprop.getLoc(),
      DenseIntElementsAttr::get(
          RankedTensorType::get({4}, builder.getIntegerType(32)),
          new_filter_shape));

  // Set stride to [1, 1, 1, 1].
  MLIRContext* context = backprop.getContext();
  SmallVector<int64_t, 4> values = {1, 1, 1, 1};
  auto attrs = llvm::map_range(values, [context](int64_t v) -> Attribute {
    return IntegerAttr::get(IntegerType::get(context, 64), APInt(64, v));
  });
  auto strides = ArrayAttr::get(context, llvm::to_vector<4>(attrs));

  // new result type.
  SmallVector<int64_t, 4> new_shape(new_filter_shape.begin(),
                                    new_filter_shape.end());
  auto new_result_type =
      RankedTensorType::get(new_shape, getElementTypeOrSelf(input));

  // Build new BackPropFilterOp.
  auto loc = backprop.getLoc();
  auto new_backprop = builder.create<TF::Conv2DBackpropFilterOp>(
      loc, new_result_type, input, new_filter_sizes, backprop.out_backprop(),
      strides, backprop.use_cudnn_on_gpu(), backprop.padding(),
      backprop.explicit_paddings(), backprop.data_format(),
      backprop.dilations());

  // For example, if new filter shape is [4, 4, 12, 64], old filter shape
  // is [7, 7, 3, 64] with block_size 2.
  // Below transforms will be applied to the filter:
  // 1. Reshape to [4, 4, 2, 2, 3, 64];
  // 2. Transpose to [4, 2, 4, 2, 3, 64];
  // 3. Reshape to [8, 8, 3, 64];
  // 4. Slice to [7, 7, 3, 64].
  SmallVector<int64_t, 6> first_reshape_shape = {
      new_filter_shape[0],
      new_filter_shape[1],
      block_size,
      block_size,
      new_filter_shape[2] / (block_size * block_size),
      new_filter_shape[3]};
  auto first_reshape_op =
      GetReshapeOpForConv2DFilter(first_reshape_shape, new_backprop, &builder);

  // Transpose.
  auto transpose_op = GetTransposeOpForConv2DFilter(&builder, first_reshape_op);

  // Last Reshape op.
  SmallVector<int64_t, 4> last_reshape_shape = {
      new_filter_shape[0] * block_size, new_filter_shape[1] * block_size,
      new_filter_shape[2] / (block_size * block_size), new_filter_shape[3]};
  auto final_reshape_op =
      GetReshapeOpForConv2DFilter(last_reshape_shape, transpose_op, &builder);

  // create slice op.
  auto slice_op = GetSliceOpForConv2DBackPropFilter(old_filter_shape,
                                                    final_reshape_op, &builder);

  // Update backprop's user with the slice op.
  backprop.replaceAllUsesWith(slice_op.getResult());
}

// Checks if the input producer op is supported in this transform. Right now, we
// only check if it is a host tf.IteratorGetNext.
bool IsSupportedHostInputOp(Operation* op) {
  TF::IteratorGetNextOp iter = llvm::dyn_cast<TF::IteratorGetNextOp>(op);
  if (!iter) return false;
  auto device = op->getAttrOfType<StringAttr>(kDeviceAttr);
  if (!device) return false;
  tensorflow::DeviceNameUtils::ParsedName parsed_device;
  if (!tensorflow::DeviceNameUtils::ParseFullName(device.getValue().str(),
                                                  &parsed_device)) {
    return false;
  }
  return parsed_device.type == "CPU";
}

// Builds a SpaceToDepthOp with the given get_layout op and input.
TF::SpaceToDepthOp BuildSpaceToDepth(tf_device::ClusterFuncOp cluster_func,
                                     Value input, int32_t block_size,
                                     ArrayRef<int64_t> input_shape) {
  auto input_op = input.getDefiningOp();
  OpBuilder builder(input_op);
  builder.setInsertionPointAfter(input_op);
  SmallVector<int64_t, 4> transform_shape = {
      input_shape[0], input_shape[1] / block_size, input_shape[2] / block_size,
      input_shape[3] * block_size * block_size};
  auto transform_result_type =
      RankedTensorType::get(transform_shape, getElementTypeOrSelf(input));
  return builder.create<TF::SpaceToDepthOp>(
      cluster_func.getLoc(), transform_result_type, input, block_size);
}

// Performs transformation for a non-replicated input.
TF::SpaceToDepthOp HandleHostInput(Value input, int64_t index,
                                   tf_device::ClusterFuncOp cluster_func,
                                   int32_t block_size,
                                   ArrayRef<int64_t> input_shape) {
  auto space_to_depth =
      BuildSpaceToDepth(cluster_func, input, block_size, input_shape);
  cluster_func.setOperand(index, space_to_depth);
  return space_to_depth;
}

// Performs transformation for replicated inputs. Returns true if this is a
// supported case (thus transform happened).
bool HandleHostReplicatedInputs(int64_t index,
                                tf_device::ClusterFuncOp cluster_func,
                                BlockArgument block_arg,
                                tf_device::ReplicateOp replicate,
                                int32_t block_size) {
  // We need to know the devices to copy to.
  if (!replicate.devices()) return false;

  MutableArrayRef<OpOperand> inputs =
      replicate.GetOperandsForBlockArgument(block_arg);
  for (auto& input : inputs) {
    auto input_op = input.get().getDefiningOp();
    if (!input_op || !IsSupportedHostInputOp(input_op)) return false;
  }
  for (auto entry : llvm::enumerate(inputs)) {
    Value input = entry.value().get();
    auto ranked_type = input.getType().dyn_cast<RankedTensorType>();
    if (!ranked_type) return false;
    auto input_shape = ranked_type.getShape();
    auto space_to_depth =
        BuildSpaceToDepth(cluster_func, input, block_size, input_shape);
    entry.value().set(space_to_depth);
    block_arg.setType(space_to_depth.getType());
  }
  return true;
}

// Performs transformation on a pair of execute and compile ops. The compile
// should not have other uses.
void HandleCluster(tf_device::ClusterFuncOp cluster_func, int32_t block_size,
                   unsigned arg_num) {
  auto maybe_replicate =
      llvm::dyn_cast<tf_device::ReplicateOp>(cluster_func->getParentOp());

  llvm::SmallVector<int64_t, 8> transform_input_indices;
  for (auto input : llvm::enumerate(cluster_func.operands())) {
    if (auto block_arg = input.value().dyn_cast<BlockArgument>()) {
      if (block_arg.getArgNumber() != arg_num) continue;
      // For a block argument, consider transforms only when it is a replicated
      // input (defining ops will be outside the replicate node).
      if (maybe_replicate == block_arg.getParentRegion()->getParentOp()) {
        HandleHostReplicatedInputs(input.index(), cluster_func, block_arg,
                                   maybe_replicate, block_size);
      }
    } else {
      // For an op output, consider transforms only when 1) there is no
      // replicateion or 2) it is outside the replicate node that encloses the
      // execute node. (Because if the op is inside replicate, it is probably
      // not on the host.)
      if (input.index() != arg_num) continue;
      auto input_op = input.value().getDefiningOp();
      if (maybe_replicate &&
          maybe_replicate.body().isAncestor(input_op->getParentRegion())) {
        continue;
      }
      if (!IsSupportedHostInputOp(input_op)) continue;
      auto ranked_type = input.value().getType().dyn_cast<RankedTensorType>();
      if (!ranked_type) continue;
      auto input_shape = ranked_type.getShape();
      HandleHostInput(input.value(), input.index(), cluster_func, block_size,
                      input_shape);
    }
  }
}

// Checks if input shape of convolution is good for space to depth transform.
bool Conv2DInputShapeCanTransform(Value input) {
  auto ranked_type = input.getType().dyn_cast<RankedTensorType>();
  if (!ranked_type) return false;
  auto input_shape = ranked_type.getShape();
  int32_t batch_size = input_shape[0];
  int32_t channel = input_shape[3];
  if (batch_size > 8 || channel > 8) {
    return false;
  }
  return true;
}

// Get block argument id and number of users for the input arg.
Optional<BlockArgumentInfo> GetBlockArgNum(Value arg) {
  if (auto block_arg = arg.dyn_cast<mlir::BlockArgument>()) {
    if (!Conv2DInputShapeCanTransform(arg)) return None;
    unsigned num_users =
        std::distance(block_arg.getUsers().begin(), block_arg.getUsers().end());
    BlockArgumentInfo block_arg_info = {block_arg.getArgNumber(), num_users};
    return block_arg_info;
  }
  return None;
}

// Gets input block argument id and number of users for the input recursively.
// Current supported ops between convolution input and the block arguments are
// PadOp and CastOp.
Optional<BlockArgumentInfo> GetInputBlockArgNum(Value input) {
  auto block_arg_num = GetBlockArgNum(input);
  if (block_arg_num.hasValue()) return block_arg_num;

  Value next_input = input;
  auto pad_op = dyn_cast_or_null<TF::PadOp>(next_input.getDefiningOp());
  auto cast_op = dyn_cast_or_null<TF::CastOp>(next_input.getDefiningOp());

  while (pad_op || cast_op) {
    if (pad_op) {
      auto block_arg_num = GetBlockArgNum(pad_op.input());
      if (block_arg_num.hasValue()) return block_arg_num;
      next_input = pad_op.input();
    } else {
      auto block_arg_num = GetBlockArgNum(cast_op.x());
      if (block_arg_num.hasValue()) return block_arg_num;
      next_input = cast_op.x();
    }
    pad_op = dyn_cast_or_null<TF::PadOp>(next_input.getDefiningOp());
    cast_op = dyn_cast_or_null<TF::CastOp>(next_input.getDefiningOp());
  }

  return None;
}

// Checks if a convoluton can apply SpaceToDepth transform.
// Only the first convolution in the graph whose batch size smaller than 8
// and its input feature size smaller than 8 can be transformed.
Optional<BlockArgumentInfo> GetConv2DInputArgNum(TF::Conv2DOp conv2d) {
  if (conv2d.data_format() != "NHWC" || conv2d.strides().size() != 4) {
    return None;
  }
  // Current supported ops between convolution input and the block arguments are
  // PadOp and CastOp.
  return GetInputBlockArgNum(conv2d.input());
}

// Applies space to depth transform for the first convolution on TPU device.
void HandleFirstConvolution(TF::Conv2DOp conv2d, int64_t block_size) {
  // Check if input and filter type are RankedTensorType.
  auto input_tensor_type =
      conv2d.input().getType().dyn_cast<RankedTensorType>();
  auto filter_tensor_type =
      conv2d.filter().getType().dyn_cast<RankedTensorType>();
  if (!input_tensor_type || !filter_tensor_type) return;
  // Book keeping filter shape for padding and backprop filter rewrite.
  auto filter_shape = filter_tensor_type.getShape();
  SmallVector<int32_t, 4> old_filter_shape(filter_shape.begin(),
                                           filter_shape.end());
  // Handles input.
  auto conv2d_input = conv2d.input();
  if (auto block_arg = conv2d_input.dyn_cast<mlir::BlockArgument>()) {
    // Change on device function type/shape.
    HandleFuncOp(block_arg.getOwner()->getParentOp());
  }

  if (auto pad_op = dyn_cast_or_null<TF::PadOp>(conv2d_input.getDefiningOp())) {
    // Rewrite pad_op before Convolutioin.
    if (failed(HandlePad(pad_op, filter_shape[0], block_size))) return;
    auto pad_input = pad_op.input();
    if (auto block_arg = pad_input.dyn_cast<mlir::BlockArgument>()) {
      // Change on device function type/shape.
      HandleFuncOp(block_arg.getOwner()->getParentOp());
    }
  }

  // Handle Conv2D input, stride and filter.
  HandleConv2DInput(conv2d, block_size);
  HandleConv2DStride(conv2d);
  HandleConv2DFilter(conv2d, block_size);

  // Book keeping new filter shape for backprop filter rewrite.
  // Filter shape is defined in HandleConv2DFilter, thus it is RankedTensorType.
  filter_shape = conv2d.filter().getType().cast<RankedTensorType>().getShape();
  SmallVector<int32_t, 4> new_filter_shape(filter_shape.begin(),
                                           filter_shape.end());

  // Rewrite Conv2DBackPropFilter that is the user of first convolution's input.
  if (!conv2d_input.getDefiningOp()) return;
  for (Operation* user : conv2d_input.getDefiningOp()->getUsers()) {
    if (auto backprop = dyn_cast<TF::Conv2DBackpropFilterOp>(user)) {
      HandleConv2DBackPropFilter(backprop, old_filter_shape, new_filter_shape,
                                 block_size);
    }
  }
}

// Gets block size that is equal to stride from spatial dimension
// from convolution.
// Space to depth transform won't be triggered if block size <= 1.
int32_t GetConv2DBlockSize(TF::Conv2DOp conv2d) {
  SmallVector<int32_t, 4> strides(4, 1);
  for (int i = 0; i < 3; ++i) {
    strides[i] = conv2d.strides()[i].cast<mlir::IntegerAttr>().getInt();
  }

  // Space to depth only supports striding at spatial dimension.
  if (strides[0] != 1 || strides[3] != 1) return 1;

  // Space to depth only supports height_stride == width_stride case.
  if (strides[1] != strides[2]) return 1;

  return strides[1];
}

void TPUSpaceToDepthPass::runOnOperation() {
  Optional<tf_device::ClusterFuncOp> cluster_func;
  // Space to depth only supports training loop.
  auto func_result = getOperation().walk([&](tf_device::ClusterFuncOp cluster) {
    cluster_func = cluster;
    return WalkResult::interrupt();
  });

  // Return if there is no tf_device::ClusterFuncOp in training loop.
  if (!func_result.wasInterrupted() || !cluster_func.hasValue()) {
    return;
  }

  // Get the function on device.
  auto device_func = cluster_func->getFunc();
  if (!device_func) return;

  TF::Conv2DOp first_conv;
  // A map maps block argument id to the convolutions consumes them.
  llvm::SmallDenseMap<unsigned, std::vector<Conv2DWithBlockSize>>
      argnum_and_convolutions;
  // A map maps block argument id to the number of users.
  llvm::SmallDenseMap<unsigned, int> argnum_num_users;

  // Find out the qualified convolutions and its block argument ids.
  auto conv2d_result = device_func.walk([&](TF::Conv2DOp conv2d) {
    Optional<BlockArgumentInfo> arg_num_and_num_users =
        GetConv2DInputArgNum(conv2d);
    if (arg_num_and_num_users.hasValue()) {
      // Get block size for the first convolution.
      int64_t block_size = GetConv2DBlockSize(conv2d);
      auto arg_num = arg_num_and_num_users.getValue().arg_num;
      auto num_users = arg_num_and_num_users.getValue().num_users;
      argnum_and_convolutions[arg_num].emplace_back(conv2d, block_size);
      argnum_num_users[arg_num] = num_users;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!conv2d_result.wasInterrupted()) {
    return;
  }

  // Iterate through block argument and its convolution users. Space to depth
  // transform will be applied only if all the below conditions are satisfied:
  //  1. All the users of the block argument will lead to convolutions;
  //  2. block_size of for the space to depth transform for these convolutions
  //     are the same;
  //  3. block_size of for the space to depth transform for these convolutions
  //     are larger than 1.
  for (auto argnum_and_convolution : argnum_and_convolutions) {
    auto arg_num = argnum_and_convolution.getFirst();
    auto conv2d_and_block_sizes = argnum_and_convolution.getSecond();
    // Continue if number of users of the block argment doesn't equal to number
    // of transformable convolutions and there is no qualified convolution
    // for transform or block size is smaller than 2.
    if (argnum_num_users[arg_num] != conv2d_and_block_sizes.size() ||
        conv2d_and_block_sizes.empty()) {
      argnum_and_convolutions.erase(arg_num);
      continue;
    }
    int64_t block_size = conv2d_and_block_sizes[0].second;
    if (block_size < 2) {
      argnum_and_convolutions.erase(arg_num);
      continue;
    }
    // Continue if not all the block sizes for space to depth transform are the
    // same.
    for (auto conv2d_and_block_size : conv2d_and_block_sizes) {
      if (conv2d_and_block_size.second != block_size) {
        argnum_and_convolutions.erase(arg_num);
        break;
      }
    }
  }

  // If there is no qualified space to depth transform.
  if (argnum_and_convolutions.empty()) {
    return;
  }

  // Apply space to depth transform.
  for (auto argnum_and_convolution : argnum_and_convolutions) {
    auto conv2d_and_block_sizes = argnum_and_convolution.getSecond();
    int64_t block_size = conv2d_and_block_sizes[0].second;
    // Apply space to depth transform to the input on the host.
    HandleCluster(cluster_func.getValue(), block_size,
                  argnum_and_convolution.getFirst());
    // Transform the convolution.
    for (auto conv2d_and_block_size : conv2d_and_block_sizes) {
      HandleFirstConvolution(conv2d_and_block_size.first,
                             conv2d_and_block_size.second);
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateTPUSpaceToDepthPass() {
  return std::make_unique<TPUSpaceToDepthPass>();
}

}  // namespace TFTPU
}  // namespace mlir
