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

#include <atomic>
#include <string>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/collectives_common.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/group_assignment.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

namespace ops_util = ::mlir::TF::collection_ops_util;
constexpr int32 kUninitializedGroupKey = 0;

// A counter that is used to generate shift base values for TF collective group
// and instance keys. Every TF collective AllReduce op in a program gets a value
// from this counter. The value increments according to the position of the
// AllReduce op in the program. Different hosts go through exactly the same MLIR
// logic and therefore iterate over AllReduce ops in the same order (even in the
// presence of control flow), so they should indenpendently generate the same
// counter value for matching AllReduce ops across hosts.
static std::atomic<int32> tf_collective_key_base{0};

}  // namespace
}  // namespace dtensor
}  // namespace tensorflow

#ifdef PLATFORM_GOOGLE
// Use the Google internal version of EmitAllReduceForXla.
#include "collective_lowering_google.inc"
#else
namespace tensorflow {
namespace dtensor {
namespace {
constexpr char kCrossReplica[] = "CrossReplica";

mlir::LogicalResult EmitAllReduceForXla(
    mlir::MLIRContext& context, mlir::OpBuilder& builder,
    mlir::TF::DTensorAllReduceOp all_reduce,
    mlir::DenseIntElementsAttr group_assignment_attr, int32 key_base,
    mlir::Operation** final_op) {
  // For TPUs, lower to XlaAllReduce straightforwardly.
  *final_op = builder.create<mlir::TF::XlaAllReduceOp>(
      all_reduce.getLoc(), all_reduce.getResult().getType(), all_reduce.input(),
      all_reduce.group_assignment(), all_reduce.reduce_opAttr(),
      builder.getStringAttr(kCrossReplica));
  return mlir::success();
}
}  // namespace
}  // namespace dtensor
}  // namespace tensorflow
#endif

namespace tensorflow {
namespace dtensor {
namespace {
// Emit a host CollectiveReduce op for the given input.
// `group_assignment` is used to generate an array of group keys.
// `device_id` slices into that array to get the key for a device at runtime.
// `key_base` is the common part shared by all group keys.
// `device_id` is an mlir::Value that will contain the device ID at runtime.
// `host_group_size` sets host collective group size. It should match the number
//   of active devices running the host collective and supplying device IDs,
//   else the host collective will crash or hang.
mlir::Operation* EmitCollectiveReduce(
    mlir::OpBuilder& builder, const mlir::Location& loc, mlir::Value input,
    const std::string& reduce_op_str,
    const mlir::DenseIntElementsAttr& group_assignment, int32 key_base,
    mlir::Value device_id, int32 host_group_size,
    const mlir::StringRef device_type) {
  DCHECK_EQ(group_assignment.getType().getRank(), 2);
  auto shape = group_assignment.getType().getShape();
  const int32 num_groups = shape[0];
  const int32 group_size = shape[1];
  const int32 num_devices = num_groups * group_size;
  const mlir::TensorType input_type =
      input.getType().dyn_cast<mlir::TensorType>();

  const bool need_int32_to_int64_upcast =
      (device_type.endswith("GPU") && input_type &&
       input_type.getElementType().isInteger(32));

  if (need_int32_to_int64_upcast) {
    LOG(WARNING) << "On GPU, collective reduce of int32 is not supported. "
                    "Casting to int64 as a workaround: "
                 << mlir::debugString(loc);

    mlir::TF::CastOp cast_to_int64 = builder.create<mlir::TF::CastOp>(
        loc,
        mlir::RankedTensorType::get(input_type.getShape(),
                                    builder.getIntegerType(64)),
        input);
    input = cast_to_int64.getResult();
  }
  mlir::Value group_key_scalar;
  llvm::SmallVector<int32, 4> device_id_to_group_key(num_devices);
  device_id_to_group_key.resize(num_devices, kUninitializedGroupKey);
  // 21 bits + 11 bits allow roughly 2M all-reduces in one program and up to a
  // full DF pod.
  DCHECK_LT(key_base, 1L << 21) << "Reaching 2^21 all-reduces.";
  DCHECK_LE(num_devices, 1L << 11) << "Exceeding 2048 groups.";
  for (const auto& it :
       llvm::enumerate(group_assignment.getValues<llvm::APInt>())) {
    int32 device_id = it.value().getSExtValue();
    DCHECK_LE(0, device_id);
    DCHECK_LT(device_id, num_devices);
    DCHECK_EQ(device_id_to_group_key[device_id], kUninitializedGroupKey);
    const int32 group_id = static_cast<int32>(it.index()) / group_size;
    device_id_to_group_key[device_id] = (key_base << 11) ^ group_id;
  }

  // Create a scalar group key by slicing device_id_to_group_key with
  // device_id.
  auto group_key_slice = builder.create<mlir::TF::SliceOp>(
      loc, EffectivelyScalarR1Type(builder.getIntegerType(32)),
      /*input=*/IntConst(builder, loc, device_id_to_group_key),
      /*begin=*/device_id,
      /*size=*/IntConst(builder, loc, {1}));
  auto group_key_reshape = builder.create<mlir::TF::ReshapeOp>(
      loc, /*tensor=*/group_key_slice.getResult(),
      /*shape=*/ops_util::GetR1Const({}, builder, loc));
  group_key_scalar = group_key_reshape.getResult();

  // Generate a unique instance key for this collective.
  mlir::Value instance_key_scalar =
      ops_util::CreateScalarConst(static_cast<int32>(key_base), builder, loc);

  const bool is_mean_op = reduce_op_str == kReduceOpMean;
  mlir::Value group_size_scalar =
      ops_util::CreateScalarConst(host_group_size, builder, loc);
  auto collective_reduce = builder.create<mlir::TF::CollectiveReduceV2Op>(
      loc, /*output_type=*/input.getType(), input, group_size_scalar,
      group_key_scalar, instance_key_scalar,
      /*ordering_token=*/mlir::ValueRange({}),
      /*merge_op=*/builder.getStringAttr(is_mean_op ? "Add" : reduce_op_str),
      /*final_op=*/builder.getStringAttr(is_mean_op ? "Div" : "Id"),
      /*communication_hint=*/builder.getStringAttr(""),
      /*timeout_seconds=*/builder.getF32FloatAttr(0.),
      /*max_subdivs_per_device=*/builder.getI64IntegerAttr(16));
  SetSingleLayoutOnOp(collective_reduce, Layout::Empty());
  if (need_int32_to_int64_upcast) {
    return builder.create<mlir::TF::CastOp>(
        loc,
        mlir::RankedTensorType::get(input_type.getShape(),
                                    builder.getIntegerType(32)),
        collective_reduce);
  }
  return collective_reduce;
}

mlir::LogicalResult LowerAllReduceOpImpl(
    mlir::MLIRContext& context, mlir::OpBuilder& builder,
    mlir::TF::DTensorAllReduceOp all_reduce, mlir::Value* value) {
  mlir::Location loc = all_reduce.getLoc();
  StatusOr<Layout> output_layout =
      ExtractRequiredSingleLayoutFromOp(all_reduce);
  if (!output_layout.ok()) {
    return all_reduce.emitOpError(output_layout.status().error_message());
  }
  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(all_reduce.group_assignment(),
                    m_Constant(&group_assignment_attr)))
    return mlir::emitError(loc, "group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return mlir::emitError(loc, "group_assignment should have two dimensions.");
  int32 group_size = group_assignment_attr.getType().getShape()[1];

  // This will become more general when Topology is properly defined.
  const bool is_tpu = all_reduce.device_type().endswith("TPU");
  // Use an atomic counter to generate bases for group and instance keys.
  int32 key_base = tf_collective_key_base++;

  mlir::Operation* final_op;
  if (is_tpu) {
    if (mlir::failed(EmitAllReduceForXla(context, builder, all_reduce,
                                         group_assignment_attr, key_base,
                                         &final_op))) {
      return mlir::failure();
    }
  } else {
    // Generate CPU/GPU collective. CPU/GPU collectives identify groups on
    // the basis of a local group key. We must generate an appropriate group
    // key based on our device ID. This is expressible as an algebraic
    // function of the device id, but we instead encode the
    // device_id->group_key as an explicit map value and lookup the result
    // at runtime. Note that the order we map devices to partitions is not
    // deterministic, and moreover if we have multiple distinct reductions
    // groups in one program reducing over all hosts and reducing over pairs
    // of hosts, we need unique ids for each case.
    mlir::Value device_id = ops_util::ReshapeScalarToSizeType(
        builder, DeviceId(all_reduce.getResult()).ValueOrDie(), loc);
    // TODO(b/188076080): Clean up device id.
    mlir::Value start_device_id = ops_util::GetR1Const(
        {(*output_layout).mesh().min_global_device_id()}, builder, loc);
    mlir::Value relative_device_id =
        builder.create<mlir::TF::SubOp>(loc, device_id, start_device_id);

    final_op = EmitCollectiveReduce(
        builder, loc, all_reduce.input(), all_reduce.reduce_op().str(),
        group_assignment_attr, key_base, relative_device_id,
        /*host_group_size=*/group_size, all_reduce.device_type().str());
  }
  SetSingleLayoutOnOp(final_op, *output_layout);
  *value = final_op->getResult(0);
  return mlir::success();
}

template <class ReduceOpType>
mlir::LogicalResult ConvertBoolReduce(ReduceOpType reduce_op) {
  mlir::OpBuilder builder(reduce_op);
  const mlir::Location loc = reduce_op.getLoc();
  const mlir::Type output_type = reduce_op.getResult().getType();
  const mlir::Type input_type = reduce_op.getOperand(0).getType();

  // Handle bools by first casting to int32 and swapping All/Any for Min/Max.
  const mlir::TensorType& tensor_input_type =
      input_type.dyn_cast<mlir::TensorType>();
  const mlir::TensorType& tensor_output_type =
      output_type.dyn_cast<mlir::TensorType>();
  if (tensor_input_type && tensor_output_type &&
      tensor_input_type.getElementType().isInteger(1)) {
    if (reduce_op.reduce_opAttr().getValue().str() == kReduceOpAll)
      reduce_op.reduce_opAttr(builder.getStringAttr(std::string(kReduceOpMin)));
    else if (reduce_op.reduce_opAttr().getValue().str() == kReduceOpAny)
      reduce_op.reduce_opAttr(builder.getStringAttr(std::string(kReduceOpMax)));
    else
      return reduce_op.emitOpError()
             << "reduce for boolean only supports 'All' or 'Any' reduction. "
             << "Received '" << reduce_op.reduce_opAttr().getValue().str()
             << "'";
    const mlir::Type integer_input_type = mlir::RankedTensorType::get(
        tensor_input_type.getShape(), builder.getIntegerType(32));
    mlir::TF::CastOp cast_to_int32 = builder.create<mlir::TF::CastOp>(
        loc, integer_input_type, reduce_op.input());
    reduce_op.setOperand(0, cast_to_int32.y());
    const mlir::Type integer_output_type = mlir::RankedTensorType::get(
        tensor_output_type.getShape(), builder.getIntegerType(32));
    reduce_op.output().setType(integer_output_type);

    // Add cast back to boolean after reduction.
    mlir::Value result = reduce_op.output();
    builder.setInsertionPointAfter(reduce_op);
    mlir::TF::CastOp cast_to_bool =
        builder.create<mlir::TF::CastOp>(loc, output_type, result);
    StatusOr<Layout> result_layout =
        ExtractRequiredSingleLayoutFromOp(result.getDefiningOp());
    if (!result_layout.ok()) {
      return reduce_op.emitOpError(result_layout.status().error_message());
    }
    SetSingleLayoutOnOp(cast_to_bool, *result_layout);
    reduce_op.output().replaceAllUsesExcept(cast_to_bool.y(), cast_to_bool);
  }

  return mlir::success();
}

mlir::LogicalResult LowerAllReduceOp(mlir::MLIRContext& context,
                                     mlir::TF::DTensorAllReduceOp all_reduce) {
  if (mlir::failed(ConvertBoolReduce<mlir::TF::DTensorAllReduceOp>(all_reduce)))
    return mlir::failure();

  mlir::OpBuilder builder(all_reduce);
  mlir::Value result;
  if (mlir::failed(LowerAllReduceOpImpl(context, builder, all_reduce, &result)))
    return mlir::failure();

  all_reduce.replaceAllUsesWith(result);
  all_reduce.erase();
  return mlir::success();
}

mlir::LogicalResult LowerReduceScatterOp(
    mlir::TF::DTensorReduceScatterOp reduce_scatter) {
  mlir::Location loc = reduce_scatter.getLoc();

  StatusOr<Layout> output_layout =
      ExtractRequiredSingleLayoutFromOp(reduce_scatter);
  if (!output_layout.ok()) {
    return reduce_scatter.emitOpError(output_layout.status().error_message());
  }
  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(reduce_scatter.group_assignment(),
                    m_Constant(&group_assignment_attr)))
    return reduce_scatter.emitOpError("group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return reduce_scatter.emitOpError(
        "group_assignment should have two dimensions.");

  mlir::OpBuilder builder(reduce_scatter);
  if (reduce_scatter.device_type().endswith("TPU")) {
    if (mlir::failed(ConvertBoolReduce<mlir::TF::DTensorReduceScatterOp>(
            reduce_scatter)))
      return mlir::failure();
    // For TPUs, lower to XlaReduceScatter straightforwardly.
    mlir::Operation* xla_reduce_scatter =
        builder.create<mlir::TF::XlaReduceScatterOp>(
            loc, reduce_scatter.getResult().getType(), reduce_scatter.input(),
            reduce_scatter.group_assignment(),
            reduce_scatter.scatter_dimension(), reduce_scatter.reduce_opAttr());
    SetSingleLayoutOnOp(xla_reduce_scatter, *output_layout);
    reduce_scatter.replaceAllUsesWith(xla_reduce_scatter);
  } else {
    // For non TPUs device, decompose to DTensorAllReduce+DTensorAllScatter.
    StatusOr<Layout> input_layout =
        ExtractRequiredLayoutFromOperand(reduce_scatter.input());
    if (!input_layout.ok()) {
      // If input layout is not defined, modify the output_layout based on the
      // scattered dimension.
      mlir::DenseIntElementsAttr scatter_attr;
      if (!matchPattern(reduce_scatter.scatter_dimension(),
                        m_Constant(&scatter_attr))) {
        return reduce_scatter.emitOpError(
            "Scatter dimension not constant integer array.");
      }
      mlir::APInt scatter_dim = *scatter_attr.begin();
      std::vector<string> input_sharding_spec =
          output_layout->sharding_spec_strs();
      input_sharding_spec[scatter_dim.getSExtValue()] = Layout::kUnshardedDim;
      input_layout =
          Layout::GetLayout(input_sharding_spec, output_layout->mesh());
    }

    if (!input_layout.ok()) {
      return reduce_scatter.emitOpError(input_layout.status().error_message());
    }

    auto dtensor_allreduce = builder.create<mlir::TF::DTensorAllReduceOp>(
        reduce_scatter.getLoc(), reduce_scatter.getOperand(0).getType(),
        reduce_scatter.getOperand(0), reduce_scatter.group_assignment(),
        reduce_scatter.reduce_op(), reduce_scatter.device_type());
    SetSingleLayoutOnOp(dtensor_allreduce, *input_layout);

    mlir::Operation* dtensor_all_scatter =
        builder.create<mlir::TF::DTensorAllScatterOp>(
            reduce_scatter.getLoc(), reduce_scatter.getResult().getType(),
            dtensor_allreduce.getResult(),
            mlir::dtensor::LayoutAttr::get(builder.getContext(), *input_layout),
            mlir::dtensor::LayoutAttr::get(builder.getContext(),
                                           *output_layout));
    SetSingleLayoutOnOp(dtensor_all_scatter, *output_layout);
    reduce_scatter.replaceAllUsesWith(dtensor_all_scatter);
  }
  reduce_scatter.erase();
  return mlir::success();
}

mlir::Value CreateZeroScalar(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::RankedTensorType type) {
  const mlir::Value zero_scalar = ops_util::CreateScalarConst(0, builder, loc);
  return builder.create<mlir::TF::CastOp>(
      loc, mlir::RankedTensorType::get({}, type.getElementType()), zero_scalar);
}

// device_id is the relative device_id in a mesh (device id - mesh's 1st device
// id).
mlir::Value SelectElementsBasedOnId(
    mlir::OpBuilder& builder, mlir::Location loc, mlir::Value device_id,
    const llvm::SmallVectorImpl<int64>& candidates_flat, int64 num_devices,
    int64 output_shape_size) {
  // Reshape the flat list to a matrix of shape num_devices * output_shape_size.
  const mlir::Value candidates_flat_const =
      ops_util::GetR1Const(candidates_flat, builder, loc);
  const mlir::Value candidates_shape =
      ops_util::GetR1Const({num_devices, output_shape_size}, builder, loc);
  const mlir::Value candidates = builder.create<mlir::TF::ReshapeOp>(
      loc, candidates_flat_const, candidates_shape);

  // Add a zero after the only value in the 1x1 device_id tensor.
  const mlir::Value device_id_paddings = builder.create<mlir::TF::ReshapeOp>(
      loc, ops_util::GetR1Const({0, 1}, builder, loc),
      ops_util::GetR1Const({1, 2}, builder, loc));
  const mlir::Value device_id_padded = builder.create<mlir::TF::PadOp>(
      loc, candidates_shape.getType(), /*input=*/device_id,
      /*paddings=*/device_id_paddings);

  // Slice a vertical vector out of the 2D candidates matrix.
  const mlir::RankedTensorType chosen_shape_type = mlir::RankedTensorType::get(
      {1, output_shape_size}, builder.getIntegerType(32));
  const mlir::Value chosen_shape_const =
      ops_util::GetR1Const(chosen_shape_type.getShape(), builder, loc);
  const mlir::Value chosen = builder.create<mlir::TF::SliceOp>(
      loc, chosen_shape_type, /*input=*/candidates, /*begin=*/device_id_padded,
      /*size=*/chosen_shape_const);

  // Remove the leading dimension of size 1 before returning the result.
  return builder.create<mlir::TF::ReshapeOp>(
      loc, chosen, ops_util::GetR1Const({output_shape_size}, builder, loc));
}

mlir::LogicalResult LowerAllGatherOp(mlir::TF::DTensorAllGatherOp all_gather) {
  const Layout src_layout = all_gather.input_layout();
  const Layout tgt_layout = all_gather.output_layout();

  llvm::SmallVector<int64, 4> concat_dims;
  for (int64 i = 0; i < src_layout.rank(); ++i)
    if (src_layout.num_shards_for_dim(src_layout.dim(i)) > 1 &&
        Layout::IsUnshardedDimension(tgt_layout.sharding_spec(i)))
      concat_dims.push_back(i);

  mlir::OpBuilder builder(all_gather);
  builder.setInsertionPointAfter(all_gather);

  if (concat_dims.empty()) {
    mlir::TF::IdentityOp identity = builder.create<mlir::TF::IdentityOp>(
        all_gather.getLoc(), all_gather.input().getType(), all_gather.input());
    SetSingleLayoutOnOp(identity, tgt_layout);

    all_gather.output().replaceAllUsesWith(identity);
    all_gather.erase();
    return mlir::success();
  }

  const mlir::RankedTensorType input_type =
      all_gather.input().getType().dyn_cast<mlir::RankedTensorType>();
  const mlir::RankedTensorType output_type =
      all_gather.output().getType().dyn_cast<mlir::RankedTensorType>();

  if (!input_type)
    return all_gather.emitOpError() << "input type is not a RankedTensorType";
  if (!output_type)
    return all_gather.emitOpError() << "output type is not a RankedTensorType";

  const std::vector<int64_t> output_shape = output_type.getShape();

  // Construct an output with zeros of the correct size, and add our
  // local slice into it. We then all reduce to compute a final result.
  const mlir::Location loc = DT_LOC(all_gather.getLoc());
  const mlir::Value output_shape_const = Int64Const(builder, loc, output_shape);
  const mlir::Value zero_scalar = CreateZeroScalar(builder, loc, input_type);
  const mlir::Value zeros =
      builder.create<mlir::TF::FillOp>(loc, output_shape_const, zero_scalar);

  // For every possible device ID, generate its strided slice ranges. Store all
  // ranges---num_devices * output_shape_size * (begin, end, stride)---as three
  // flat lists.
  // Consider making this a generalized N-dimensional helper on Layout.
  const int64 num_devices = src_layout.num_devices();
  const int64 output_shape_size = output_shape.size();
  llvm::SmallVector<int64, 4> device_id_to_begin_flat;
  llvm::SmallVector<int64, 4> device_id_to_end_flat;
  llvm::SmallVector<int64, 4> device_id_to_strides_flat;
  for (int64 device_id = 0; device_id < num_devices; ++device_id) {
    for (int64 i = 0; i < output_shape_size; ++i) {
      if (llvm::find(concat_dims, i) == std::end(concat_dims)) {
        // For unsharded dimensions, the slice range is [0, dim_size).
        device_id_to_begin_flat.push_back(0);
        device_id_to_end_flat.push_back(output_shape[i]);
      } else {
        // For sharded dimensions, the slice range is [step * device_id, step *
        // (device_id + 1)), where step = dim_size / num_of_shards.
        StatusOr<DeviceLocation> device_loc_or_status =
            src_layout.device_location(device_id);
        if (!device_loc_or_status.ok())
          return all_gather.emitOpError()
                 << device_loc_or_status.status().error_message();
        const DeviceLocation device_loc = device_loc_or_status.ValueOrDie();
        const int32 mesh_idx = src_layout.mesh()
                                   .idx_for_dim(src_layout.sharding_spec(i))
                                   .ValueOrDie();
        const int64 device_offset = device_loc[mesh_idx];
        const int64 step = output_shape[i] / src_layout.num_shards()[i];
        device_id_to_begin_flat.push_back(step * device_offset);
        device_id_to_end_flat.push_back(step * device_offset + step);
      }
      // We need to change every element in the selected slice, so stride is 1
      // for every dimension.
      device_id_to_strides_flat.push_back(1);
    }
  }

  // Resize three flat lists to 2D matrices and select one vertical vector out
  // of every matrix based on device ID.
  StatusOr<mlir::Value> device_id_scalar_or_status =
      DeviceId(all_gather.input());
  if (!device_id_scalar_or_status.ok())
    return all_gather.emitOpError()
           << device_id_scalar_or_status.status().error_message();
  const mlir::Value device_id_scalar = device_id_scalar_or_status.ValueOrDie();
  const mlir::Value device_id =
      ops_util::ReshapeScalarToSizeType(builder, device_id_scalar, loc);
  // TODO(b/188076080): Clean up device id.
  const mlir::Value start_device_id = ops_util::GetR1Const(
      {src_layout.mesh().min_global_device_id()}, builder, loc);
  const mlir::Value relative_device_id =
      builder.create<mlir::TF::SubOp>(loc, device_id, start_device_id);
  const mlir::Value begin = SelectElementsBasedOnId(
      builder, loc, relative_device_id, device_id_to_begin_flat, num_devices,
      output_shape_size);
  const mlir::Value end = SelectElementsBasedOnId(
      builder, loc, relative_device_id, device_id_to_end_flat, num_devices,
      output_shape_size);
  const mlir::Value strides = SelectElementsBasedOnId(
      builder, loc, relative_device_id, device_id_to_strides_flat, num_devices,
      output_shape_size);

  // Fill in the local portion by slicing into the correct subrange.
  mlir::Value update_result;
  if (src_layout.mesh().is_tpu_mesh()) {
    if (!tgt_layout.mesh().is_tpu_mesh())
      return all_gather.emitOpError()
             << "source and target layout are not both on tpu";
    update_result = builder.create<mlir::TF::XlaDynamicUpdateSliceOp>(
        loc, zeros.getType(), /*input=*/zeros,
        /*update=*/all_gather.input(), /*indices=*/begin);
  } else {
    update_result = builder.create<mlir::TF::TensorStridedSliceUpdateOp>(
        loc, zeros.getType(),
        /*input=*/zeros, begin, end, strides,
        /*value=*/all_gather.input());
  }

  // All reduce among concatenated dimensions.
  absl::flat_hash_set<std::string> reduced_dims;
  for (int i : concat_dims) reduced_dims.insert(src_layout.sharding_spec(i));

  auto partitions_or_status =
      GetAllReducePartitionsFromReducedDims(src_layout, reduced_dims);
  if (!partitions_or_status.ok())
    return all_gather.emitOpError()
           << partitions_or_status.status().error_message();
  auto partitions = partitions_or_status.ValueOrDie();
  const int32 num_partitions = partitions.size();
  assert(num_partitions <= num_devices);
  if (num_partitions == num_devices) {
    // TODO(unknown): Is this check needed? Since we check that num_shards for
    // each reduced_dims in the src layout is > 1, I think we always need
    // communication.
    // If every device lives in its own partition, we don't need to emit a
    // collective.
    SetSingleLayoutOnOp(update_result.getDefiningOp(), tgt_layout);
    all_gather.output().replaceAllUsesWith(update_result);
    all_gather.erase();
    return mlir::success();
  }

  std::vector<int32> partitions_flat;
  for (auto& p : partitions) {
    if (p.second.size() != partitions.begin()->second.size())
      return all_gather.emitOpError() << "partitions had different sizes -- "
                                         "this is not supported in MLIR.";
    partitions_flat.insert(partitions_flat.end(), p.second.begin(),
                           p.second.end());
  }
  const int32 partition_size = partitions.begin()->second.size();
  const mlir::RankedTensorType shaped_type = mlir::RankedTensorType::get(
      {num_partitions, partition_size},
      mlir::IntegerType::get(builder.getContext(), 32));
  const mlir::DenseIntElementsAttr group_assignment =
      mlir::DenseIntElementsAttr::get(shaped_type, partitions_flat);
  StatusOr<std::string> device_type_or_status =
      DeviceTypeFromMesh(src_layout.mesh());
  if (!device_type_or_status.ok())
    return all_gather.emitOpError()
           << device_type_or_status.status().error_message();
  const std::string device_type = device_type_or_status.ValueOrDie();

  // Support bool types by switching to Any reduce rather than Add. For each
  // position in the tensor, only one task in the reduction group can have a 1.
  // This is sufficient.
  const mlir::TensorType type =
      update_result.getType().dyn_cast<mlir::TensorType>();
  absl::string_view reduce_type = kReduceOpAdd;
  if (type && type.getElementType().isInteger(1)) reduce_type = kReduceOpAny;
  mlir::TF::DTensorAllReduceOp all_reduce =
      builder.create<mlir::TF::DTensorAllReduceOp>(
          loc, update_result.getType(), update_result,
          builder.create<mlir::TF::ConstOp>(loc, group_assignment),
          builder.getStringAttr(std::string(reduce_type)),
          builder.getStringAttr(device_type));
  SetSingleLayoutOnOp(all_reduce, tgt_layout);

  all_gather.output().replaceAllUsesWith(all_reduce.getResult());
  all_gather.erase();
  return mlir::LogicalResult::success();
}

mlir::LogicalResult LowerAllScatterOp(
    mlir::TF::DTensorAllScatterOp all_scatter) {
  const Layout original_layout = all_scatter.input_layout();
  const Layout desired_layout = all_scatter.output_layout();

  mlir::tf_device::ClusterOp cluster =
      all_scatter->getParentOfType<mlir::tf_device::ClusterOp>();
  StatusOr<mlir::Value> mesh_coordinates_status =
      GetMeshCoordinatesFromCluster(cluster);
  if (!mesh_coordinates_status.ok())
    return all_scatter.emitOpError()
           << mesh_coordinates_status.status().error_message();
  mlir::Value mesh_coordinates = mesh_coordinates_status.ValueOrDie();

  // We need to compute the slice offset, which is dynamic based on the id.
  //
  // To compute the offset:
  // For axes where there is no splitting, the offset is simply 0.
  // For axes where there is splitting, say axis a, if new local size of that
  // axis is k, then the offset for the split is
  // mesh_coordinates[sharding_spec[a]]*k where sharding_spec[i] is the
  // mesh_dimension for a. This computation can be encoded in small 2d matrix of
  // shape [mesh.rank(), layout.rank()] where the [i, j]'th entry is k if
  // sharding_spec[j]=i and this is a dimension with split and 0 otherwise.

  mlir::RankedTensorType output_type =
      all_scatter.output().getType().dyn_cast<mlir::RankedTensorType>();
  if (!output_type)
    return all_scatter.emitOpError() << "input must have static rank";

  llvm::ArrayRef<int64_t> output_shape = output_type.getShape();

  // We use a flat list here. The 2D matrix will be of shape
  // [original_layout.mesh().rank(), original_layout.rank()]
  // so the 2D index [i, j] corresponds to the 1D index of
  // [i * original_layout.rank() + j].
  std::vector<int32> matrix(original_layout.mesh().rank() *
                            original_layout.rank());
  for (int i = 0; i < original_layout.rank(); ++i) {
    if (original_layout.sharding_spec(i) != desired_layout.sharding_spec(i)) {
      if (mlir::ShapedType::isDynamic(output_shape[i])) {
        return all_scatter.emitOpError()
               << "EmitAllScatter requires slice on input axis " << i
               << " which is dynamic. This is not supported";
      }

      // We already checked above that original_layout.sharding_spec(i) is
      // unsharded.
      int mesh_dim_index = desired_layout.mesh().GetMeshDimIndexWithName(
          desired_layout.sharding_spec(i));
      matrix[mesh_dim_index * original_layout.rank() + i] = output_shape[i];
    }
  }

  // Produce the constant tensor for the slice shape and the matrix.

  mlir::OpBuilder builder(all_scatter);

  // Slice shape has to be int32_t, as it must match the type of the offset to
  // mlir::TF::SliceOp. The slice offset has to be int32_t as TPU doesn't have
  // int64_t MatMul (which we use to compute the offset).
  llvm::SmallVector<int32_t> output_shape_int32(output_shape.begin(),
                                                output_shape.end());
  mlir::Value slice_shape_value =
      IntConst(builder, all_scatter.getLoc(), output_shape_int32);

  mlir::RankedTensorType matrix_type = mlir::RankedTensorType::get(
      {original_layout.mesh().rank(), original_layout.rank()},
      builder.getIntegerType(32));
  mlir::Attribute matrix_attr =
      mlir::DenseIntElementsAttr::get(matrix_type, matrix);
  mlir::Value matrix_value =
      builder.create<mlir::TF::ConstOp>(all_scatter.getLoc(), matrix_attr)
          .getResult();

  // Compute the offset from mult_matrix_value and mesh_coordinates.
  mlir::TF::MatMulOp offset = builder.create<mlir::TF::MatMulOp>(
      all_scatter.getLoc(),
      mlir::RankedTensorType::get({1, original_layout.rank()},
                                  builder.getIntegerType(32)),
      mesh_coordinates, matrix_value);

  // Input to slice needs to be rank 1, so we need to sequeeze it.
  mlir::TF::SqueezeOp offset_squeezed = builder.create<mlir::TF::SqueezeOp>(
      all_scatter.getLoc(),
      mlir::RankedTensorType::get({original_layout.rank()},
                                  builder.getIntegerType(32)),
      offset.product(), builder.getI64ArrayAttr({0}));

  auto result = builder.create<mlir::TF::SliceOp>(
      all_scatter.getLoc(), output_type, all_scatter.input(),
      offset_squeezed.output(), slice_shape_value);

  SetSingleLayoutOnOp(result, desired_layout);

  all_scatter.output().replaceAllUsesExcept(result.output(), result);
  all_scatter.erase();

  return mlir::LogicalResult::success();
}

struct DTensorAllReduceLowering
    : public DTensorAllReduceLoweringBase<DTensorAllReduceLowering> {
  void runOnOperation() override {
    mlir::MLIRContext& context = getContext();
    mlir::ModuleOp module = getOperation();

    // Find all DTensorAllReduce ops.
    llvm::SmallVector<mlir::TF::DTensorAllReduceOp, 4> all_reduces;
    module.walk([&](mlir::TF::DTensorAllReduceOp all_reduce) {
      all_reduces.emplace_back(all_reduce);
    });

    // Replace every DTensorAllReduce op with device-specific implementations.
    for (auto& all_reduce : all_reduces)
      if (mlir::failed(LowerAllReduceOp(context, all_reduce)))
        return signalPassFailure();
  }
};

struct DTensorReduceScatterLowering
    : public DTensorReduceScatterLoweringBase<DTensorReduceScatterLowering> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Find all DTensorAllReduce ops.
    llvm::SmallVector<mlir::TF::DTensorReduceScatterOp, 4> all_reduces;
    module.walk([&](mlir::TF::DTensorReduceScatterOp all_reduce) {
      all_reduces.emplace_back(all_reduce);
    });

    // Replace every DTensorAllReduce op with device-specific implementations.
    for (auto& all_reduce : all_reduces)
      if (mlir::failed(LowerReduceScatterOp(all_reduce)))
        return signalPassFailure();
  }
};

struct DTensorAllGatherLowering
    : public DTensorAllGatherLoweringBase<DTensorAllGatherLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Process all DTensorAllGather ops.
    llvm::SmallVector<mlir::TF::DTensorAllGatherOp, 4> all_gathers;
    module.walk([&](mlir::TF::DTensorAllGatherOp all_gather) {
      all_gathers.emplace_back(all_gather);
    });

    for (mlir::TF::DTensorAllGatherOp all_gather : all_gathers)
      if (mlir::failed(LowerAllGatherOp(all_gather)))
        return signalPassFailure();
  }
};

struct DTensorAllScatterLowering
    : public DTensorAllScatterLoweringBase<DTensorAllScatterLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Process all DTensorAllScatter ops.
    llvm::SmallVector<mlir::TF::DTensorAllScatterOp, 4> all_scatters;
    module.walk([&](mlir::TF::DTensorAllScatterOp all_scatter) {
      all_scatters.emplace_back(all_scatter);
    });

    for (mlir::TF::DTensorAllScatterOp all_scatter : all_scatters)
      if (mlir::failed(LowerAllScatterOp(all_scatter)))
        return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllReduceLoweringPass() {
  return std::make_unique<DTensorAllReduceLowering>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorReduceScatterLoweringPass() {
  return std::make_unique<DTensorReduceScatterLowering>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllGatherLoweringPass() {
  return std::make_unique<DTensorAllGatherLowering>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllScatterLoweringPass() {
  return std::make_unique<DTensorAllScatterLowering>();
}

}  // namespace dtensor
}  // namespace tensorflow
