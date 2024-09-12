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

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/collection_ops_util.h"
#include "xla/tsl/util/env_var.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/collectives_common.h"
#include "tensorflow/dtensor/mlir/device_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {
#define GEN_PASS_DEF_DTENSORALLREDUCELOWERING
#define GEN_PASS_DEF_DTENSORREDUCESCATTERLOWERING
#define GEN_PASS_DEF_DTENSORALLGATHERLOWERING
#define GEN_PASS_DEF_DTENSORALLSCATTERLOWERING
#define GEN_PASS_DEF_DTENSORALLTOALLLOWERING
#include "tensorflow/dtensor/mlir/dtensor_passes.h.inc"
}  // namespace

namespace internal {

namespace ops_util = ::mlir::TF::collection_ops_util;
constexpr int32 kUninitializedGroupKey = 0;
constexpr char kCpuDevice[] = "/device:CPU:0";
constexpr char kDeviceAttr[] = "device";

std::atomic<int32> tf_collective_instance_key_base{0};

bool HasEnableReuseGroupKey() {
  // FIXME(b/258703996): use tsl::ReadBoolFromEnvVar()
  // Experimental feature. If nonzero, reuse group key when emitting
  // Collectives. Default is 1. This is only allowed to be set before the first
  // use of DTensor.
  static const char* env_str = (std::getenv("DTENSOR_ENABLE_REUSE_GROUP_KEY"));
  if (env_str && strcmp(env_str, "0") == 0) {
    return false;
  }
  return true;
}

bool UseNcclCommunicationOnGpu() {
  // This is the same as gpu_use_nccl_communication() from
  // tensorflow/dtensor/python/config.py.
  static bool is_enabled = [] {
    bool ret = false;
    TF_CHECK_OK(tsl::ReadBoolFromEnvVar("DTENSOR_GPU_USE_NCCL_COMMUNICATION",
                                        /*default_val=*/false, &ret));
    return ret;
  }();
  return is_enabled;
}

mlir::LogicalResult EmitAllReduceForXla(
    mlir::MLIRContext& context, mlir::OpBuilder& builder,
    mlir::TF::DTensorAllReduceOp all_reduce,
    mlir::DenseIntElementsAttr group_assignment_attr, int32 key_base,
    mlir::Operation** final_op) {
  constexpr char kCrossReplica[] = "CrossReplica";

  // For TPUs, lower to XlaAllReduce straightforwardly.
  *final_op = builder.create<mlir::TF::XlaAllReduceOp>(
      all_reduce.getLoc(), all_reduce.getResult().getType(),
      all_reduce.getInput(), all_reduce.getGroupAssignment(),
      all_reduce.getReduceOpAttr(), builder.getStringAttr(kCrossReplica));
  return mlir::success();
}

llvm::SmallVector<int32_t, 4> GetGroupKeyOffsets(
    const mlir::DenseIntElementsAttr& group_assignment, int32_t* group_size) {
  DCHECK_EQ(group_assignment.getType().getRank(), 2);
  auto shape = group_assignment.getType().getShape();
  const int32_t num_groups = shape[0];
  *group_size = shape[1];
  const int32_t num_devices = num_groups * *group_size;

  llvm::SmallVector<int32, 4> device_id_to_group_key(num_devices);
  device_id_to_group_key.resize(num_devices, kUninitializedGroupKey);
  // 21 bits + 11 bits allow roughly 2M all-reduces in one program and up to a
  // full DF pod.
  DCHECK_LE(num_devices, 1L << 11) << "Exceeding 2048 groups.";
  for (const auto& it :
       llvm::enumerate(group_assignment.getValues<llvm::APInt>())) {
    int32 device_id = it.value().getSExtValue();
    DCHECK_LE(0, device_id);
    DCHECK_LT(device_id, num_devices);
    DCHECK_EQ(device_id_to_group_key[device_id], kUninitializedGroupKey);
    const int32 group_offset = static_cast<int32>(it.index()) / *group_size;
    device_id_to_group_key[device_id] = group_offset;
  }
  return device_id_to_group_key;
}

int32_t GetCollectiveKeyBase(
    Mesh mesh, const mlir::DenseIntElementsAttr& group_assignment) {
  // A counter that is used to generate shift base values for TF collective
  // group and instance keys. Every TF collective AllReduce op in a program gets
  // a value from this counter. The value increments according to the position
  // of the AllReduce op in the program. Different hosts go through exactly the
  // same MLIR logic and therefore iterate over AllReduce ops in the same order
  // (even in the presence of control flow), so they should indenpendently
  // generate the same counter value for matching AllReduce ops across hosts.
  static std::atomic<int32> tf_collective_key_base{0};

  if (!HasEnableReuseGroupKey()) {
    return tf_collective_key_base++;
  }
  // Use an atomic counter to generate bases for group and instance keys.
  static tensorflow::mutex* mtx = new tensorflow::mutex();
  static auto* mesh_to_key_base =
      new std::map<std::tuple<std::string, llvm::SmallVector<int32, 4>>,
                   int32_t>();
  int32_t group_size;
  const llvm::SmallVector<int32, 4> group_key_offsets =
      GetGroupKeyOffsets(group_assignment, &group_size);

  const auto iter =
      mesh_to_key_base->find({mesh.ToString(), group_key_offsets});
  tensorflow::mutex_lock lock(*mtx);
  if (iter != mesh_to_key_base->end()) {
    return iter->second;
  }
  int32_t key_base = tf_collective_key_base++;
  mesh_to_key_base->insert({{mesh.ToString(), group_key_offsets}, key_base});
  VLOG(4) << "key base = " << key_base << " mesh = " << mesh.ToString();
  return key_base;
}

mlir::Value GetRelativeDeviceId(mlir::Operation* op,
                                const Layout& output_layout,
                                mlir::OpBuilder& builder,
                                const mlir::Location& loc) {
  // TODO(tmorris): Should this be using op.getResult()?
  mlir::Value device_id =
      ops_util::ReshapeScalarToSizeType(builder, DeviceId(op).value(), loc);
  mlir::Value start_device_id = ops_util::GetR1Const(
      {output_layout.mesh().min_global_device_id()}, builder, loc);
  return builder.create<mlir::TF::SubOp>(loc, device_id, start_device_id);
}

void CreateGroupAndInstanceKey(
    mlir::OpBuilder& builder, const mlir::Location& loc,
    const mlir::DenseIntElementsAttr& group_assignment, int32 key_base,
    mlir::Value device_id, mlir::Value* group_key_scalar,
    mlir::Value* instance_key_scalar) {
  int32_t group_size;
  llvm::SmallVector<int32, 4> device_id_to_group_key =
      GetGroupKeyOffsets(group_assignment, &group_size);
  // 21 bits + 11 bits allow roughly 2M all-reduces in one program and up to a
  // full DF pod.
  DCHECK_LT(key_base, 1L << 21) << "Reaching 2^21 all-reduces.";
  for (int32_t& it : device_id_to_group_key) {
    it += (key_base << 11);
  }

  // Create a scalar group key by slicing device_id_to_group_key with
  // device_id.
  auto group_key_loc = DT_LOC2(loc, "group_key");
  auto group_key_slice = builder.create<mlir::TF::SliceOp>(
      group_key_loc, EffectivelyScalarR1Type(builder.getIntegerType(32)),
      /*input=*/IntConst(builder, loc, device_id_to_group_key),
      /*begin=*/device_id,
      /*size=*/IntConst(builder, loc, {1}));
  auto group_key_reshape = builder.create<mlir::TF::ReshapeOp>(
      group_key_loc, /*tensor=*/group_key_slice.getResult(),
      /*shape=*/ops_util::GetR1Const({}, builder, loc));
  *group_key_scalar = group_key_reshape.getResult();

  // Generate a unique instance key for this collective.
  *instance_key_scalar = ops_util::CreateScalarConst(
      static_cast<int32>(tf_collective_instance_key_base++), builder,
      DT_LOC2(loc, "instance_key"));
}

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
  mlir::Value group_key_scalar;
  mlir::Value instance_key_scalar;
  CreateGroupAndInstanceKey(builder, loc, group_assignment, key_base, device_id,
                            &group_key_scalar, &instance_key_scalar);

  const bool is_mean_op = reduce_op_str == kReduceOpMean;
  mlir::Value group_size_scalar = ops_util::CreateScalarConst(
      host_group_size, builder, DT_LOC2(loc, "group_size"));
  auto collective_reduce = builder.create<mlir::TF::CollectiveReduceV2Op>(
      loc, /*output_type=*/input.getType(), input, group_size_scalar,
      group_key_scalar, instance_key_scalar,
      /*ordering_token=*/mlir::ValueRange({}),
      /*merge_op=*/builder.getStringAttr(is_mean_op ? "Add" : reduce_op_str),
      /*final_op=*/builder.getStringAttr(is_mean_op ? "Div" : "Id"),
      /*communication_hint=*/builder.getStringAttr(""),
      /*timeout_seconds=*/builder.getF32FloatAttr(0.),
      /*is_stateless=*/builder.getBoolAttr(false),
      /*max_subdivs_per_device=*/builder.getI64IntegerAttr(16));
  SetSingleLayoutOnOp(collective_reduce, Layout::Empty());
  return collective_reduce;
}

mlir::Operation* EmitCollectiveReduceScatter(
    mlir::OpBuilder& builder, const mlir::Location& loc, mlir::Value input,
    mlir::Type output_type, const std::string& reduce_op_str,
    const mlir::DenseIntElementsAttr& group_assignment, int32 scatter_dimension,
    int32 key_base, mlir::Value device_id, int32 host_group_size,
    const mlir::StringRef device_type) {
  mlir::TensorType input_type =
      mlir::dyn_cast<mlir::TensorType>(input.getType());

  const bool need_transpose = scatter_dimension != 0;
  std::vector<int64> perm_for_transpose;
  if (need_transpose) {
    perm_for_transpose.reserve(input_type.getRank());
    for (int i = 0; i < input_type.getRank(); i++) {
      perm_for_transpose.push_back(i);
    }
    std::swap(perm_for_transpose[scatter_dimension], perm_for_transpose[0]);
    auto pre_transpose_op =
        EmitTransposeOp(builder, loc, input, perm_for_transpose);
    input = pre_transpose_op->getResult(0);
    input_type = mlir::dyn_cast<mlir::TensorType>(input.getType());
    // Compute transposed output type for CollectiveReduceScatter
    auto output_shape =
        mlir::dyn_cast<mlir::TensorType>(output_type).getShape();
    std::vector<int64> transposed_shape(output_shape.begin(),
                                        output_shape.end());
    for (int i = 0; i < output_shape.size(); i++) {
      transposed_shape[i] = output_shape[perm_for_transpose[i]];
    }
    output_type = mlir::RankedTensorType::get(transposed_shape,
                                              input_type.getElementType());
  }

  mlir::Value group_key_scalar;
  mlir::Value instance_key_scalar;
  CreateGroupAndInstanceKey(builder, loc, group_assignment, key_base, device_id,
                            &group_key_scalar, &instance_key_scalar);

  const bool is_mean_op = reduce_op_str == kReduceOpMean;
  mlir::Value group_size_scalar = ops_util::CreateScalarConst(
      host_group_size, builder, DT_LOC2(loc, "group_size"));
  auto collective_reduce_scatter = builder.create<
      mlir::TF::CollectiveReduceScatterV2Op>(
      loc, output_type, input, group_size_scalar, group_key_scalar,
      instance_key_scalar,
      /*ordering_token=*/mlir::ValueRange({}),
      /*merge_op=*/builder.getStringAttr(is_mean_op ? "Add" : reduce_op_str),
      /*final_op=*/builder.getStringAttr(is_mean_op ? "Div" : "Id"),
      /*communication_hint=*/builder.getStringAttr("nccl"),  // TODO(tmorris):
                                                             // this shouldn't
                                                             // be needed
      /*timeout_seconds=*/builder.getF32FloatAttr(0.),
      /*is_stateless=*/builder.getBoolAttr(false),
      /*max_subdivs_per_device=*/builder.getI64IntegerAttr(16));
  SetSingleLayoutOnOp(collective_reduce_scatter, Layout::Empty());
  if (need_transpose) {
    return EmitTransposeOp(builder, loc,
                           collective_reduce_scatter->getResult(0),
                           perm_for_transpose);
  }
  return collective_reduce_scatter;
}

mlir::Operation* EmitCollectiveAllToAll(
    mlir::OpBuilder& builder, const mlir::Location& loc, mlir::Value input,
    const mlir::DenseIntElementsAttr& group_assignment, int32 concat_dimension,
    int32 split_dimension, int32 key_base, mlir::Value device_id,
    int32 host_group_size, const mlir::StringRef device_type) {
  // This function implements an all-to-all with variable split and concat
  // dimensions using the CollectiveAllToAllV2 which treats the input as a flat
  // buffer. This requires permuting the data before or after the all-to-all
  // using a reshape from rank N to rank N+1 followed by a transpose.
  // Additionally, if neither the split or concat dimensions are rank 0, a pair
  // of transpose, one before and one after all-to-all is needed to split the
  // data correctly. An example relayout that requires this is [y, unsharded, x]
  // -> [y, x, unsharded].
  const mlir::TensorType input_type =
      mlir::dyn_cast<mlir::TensorType>(input.getType());
  auto input_shape = mlir::dyn_cast<mlir::TensorType>(input_type).getShape();

  // TODO(trevor-m): One of the transpose pairs created when requires_transpose
  // is true can be combined with the transpose in permute_data() that lies on
  // the same side of all-to-all.
  const bool permute_before = concat_dimension < split_dimension;
  const bool requires_transpose = concat_dimension != 0 && split_dimension != 0;
  std::vector<int64> transposed_shape(input_shape.begin(), input_shape.end());
  std::vector<int64> original_shape(input_shape);
  int move_dims = std::min(concat_dimension, split_dimension);
  if (requires_transpose) {
    std::vector<int64> perm_for_transpose;
    perm_for_transpose.reserve(input_shape.size());
    // Move all dims before concat/split to end. This will be undone after the
    // all-to-all.
    for (int i = move_dims; i < input_shape.size(); ++i) {
      perm_for_transpose.push_back(i);
    }
    for (int i = 0; i < move_dims; ++i) {
      perm_for_transpose.push_back(i);
    }
    input =
        EmitTransposeOp(builder, loc, input, perm_for_transpose)->getResult(0);
    for (int i = 0; i < input_shape.size(); i++) {
      transposed_shape[i] = input_shape[perm_for_transpose[i]];
    }
    if (permute_before) {
      concat_dimension -= move_dims;
      split_dimension -= move_dims;
      input_shape = transposed_shape;
    }
  }

  auto permute_data = [&](mlir::Value data) {
    // Reshape
    std::vector<int64> new_shape;
    new_shape.reserve(input_shape.size() + 1);
    for (int i = 0; i < input_shape.size(); ++i) {
      if (i == split_dimension) {
        new_shape.push_back(host_group_size);
        new_shape.push_back(input_shape[i] / host_group_size);
      } else {
        new_shape.push_back(input_shape[i]);
      }
    }
    auto reshape_op = builder.create<mlir::TF::ReshapeOp>(
        loc, data, ops_util::GetR1Const(new_shape, builder, loc));

    std::vector<int64> perm_for_permute_transpose;
    perm_for_permute_transpose.reserve(input_shape.size() + 1);
    for (int i = 0; i < input_shape.size(); ++i) {
      if (i == concat_dimension) {
        perm_for_permute_transpose.push_back(split_dimension);
      }
      int dim_after_reshape = i >= split_dimension ? i + 1 : i;
      perm_for_permute_transpose.push_back(dim_after_reshape);
    }
    return EmitTransposeOp(builder, loc, reshape_op->getResult(0),
                           perm_for_permute_transpose);
  };

  if (permute_before) {
    input = permute_data(input)->getResult(0);
  }

  // Flatten input. CPU implementation requires first dim to equal the group
  // size.
  int64 num_elements = std::accumulate(input_shape.begin(), input_shape.end(),
                                       1LL, std::multiplies<int64>());
  std::vector<int64> flatten_shape = {host_group_size,
                                      num_elements / host_group_size};
  auto flatten_reshape_op = builder.create<mlir::TF::ReshapeOp>(
      loc, input, ops_util::GetR1Const(flatten_shape, builder, loc));
  mlir::TensorType output_type =
      mlir::RankedTensorType::get(flatten_shape, input_type.getElementType());

  // All-to-all
  mlir::Value group_key_scalar;
  mlir::Value instance_key_scalar;
  CreateGroupAndInstanceKey(builder, loc, group_assignment, key_base, device_id,
                            &group_key_scalar, &instance_key_scalar);
  mlir::Value group_size_scalar =
      ops_util::CreateScalarConst(host_group_size, builder, loc);
  auto collective_alltoall = builder.create<mlir::TF::CollectiveAllToAllV2Op>(
      loc, /*output_type=*/output_type, flatten_reshape_op->getResult(0),
      group_size_scalar, group_key_scalar, instance_key_scalar,
      /*ordering_token=*/mlir::ValueRange({}),
      /*communication_hint=*/builder.getStringAttr(""),
      /*timeout_seconds=*/builder.getF32FloatAttr(0.),
      /*is_stateless=*/builder.getBoolAttr(false));
  SetSingleLayoutOnOp(collective_alltoall, Layout::Empty());
  mlir::Value prev_op = collective_alltoall->getResult(0);

  if (requires_transpose) {
    // Unflatten after all-to-all.
    auto reshape_op = builder.create<mlir::TF::ReshapeOp>(
        loc, prev_op, ops_util::GetR1Const(transposed_shape, builder, loc));
    // Undo earlier transpose which moved split or concat dim to rank 0.
    std::vector<int64> perm_for_transpose;
    perm_for_transpose.reserve(input_shape.size());
    for (int i = move_dims + 1; i < input_shape.size(); ++i) {
      perm_for_transpose.push_back(i);
    }
    for (int i = 0; i <= move_dims; ++i) {
      perm_for_transpose.push_back(i);
    }
    prev_op = EmitTransposeOp(builder, loc, reshape_op->getResult(0),
                              perm_for_transpose)
                  ->getResult(0);
    if (permute_before) {
      concat_dimension += move_dims;
      split_dimension += move_dims;
      input_shape = original_shape;
    }
  }

  if (!permute_before) {
    prev_op = permute_data(prev_op)->getResult(0);
  }

  // Reshape
  std::vector<int64> output_shape(input_shape.begin(), input_shape.end());
  output_shape[concat_dimension] *= host_group_size;
  output_shape[split_dimension] /= host_group_size;
  auto post_reshape_op = builder.create<mlir::TF::ReshapeOp>(
      loc, prev_op, ops_util::GetR1Const(output_shape, builder, loc));

  return post_reshape_op;
}

mlir::Operation* EmitCollectiveGather(
    mlir::OpBuilder& builder, const mlir::Location& loc, mlir::Value input,
    const mlir::DenseIntElementsAttr& group_assignment, int32 key_base,
    mlir::Value device_id, int32 host_group_size,
    const mlir::StringRef device_type) {
  DCHECK_EQ(group_assignment.getType().getRank(), 2);
  auto shape = group_assignment.getType().getShape();
  const int32 group_size = shape[1];
  const mlir::TensorType input_type =
      mlir::dyn_cast<mlir::TensorType>(input.getType());
  auto input_shape = input_type.getShape();
  auto dim_0_shape = input_shape[0];
  std::vector<int64> output_shape = {input_shape.begin(), input_shape.end()};
  output_shape[0] = dim_0_shape * group_size;
  auto output_type =
      mlir::RankedTensorType::get(output_shape, input_type.getElementType());

  mlir::Value group_key_scalar;
  mlir::Value instance_key_scalar;
  CreateGroupAndInstanceKey(builder, loc, group_assignment, key_base, device_id,
                            &group_key_scalar, &instance_key_scalar);

  mlir::Value group_size_scalar =
      ops_util::CreateScalarConst(host_group_size, builder, loc);
  auto collective_gather = builder.create<mlir::TF::CollectiveGatherV2Op>(
      loc, /*output_type=*/input.getType(), input, group_size_scalar,
      group_key_scalar, instance_key_scalar,
      /*ordering_token=*/mlir::ValueRange({}),
      /*communication_hint=*/builder.getStringAttr(""),
      /*timeout_seconds=*/builder.getF32FloatAttr(0.),
      /*is_stateless=*/builder.getBoolAttr(false));
  SetSingleLayoutOnOp(collective_gather, Layout::Empty());
  collective_gather.getData().setType(output_type);

  return collective_gather;
}

mlir::LogicalResult LowerAllReduceOpImpl(
    mlir::MLIRContext& context, mlir::OpBuilder& builder,
    mlir::TF::DTensorAllReduceOp all_reduce, mlir::Value* value) {
  mlir::Location loc = all_reduce.getLoc();
  StatusOr<Layout> output_layout =
      ExtractRequiredSingleLayoutFromOp(all_reduce);
  if (!output_layout.ok()) {
    return all_reduce.emitOpError(output_layout.status().message());
  }
  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(all_reduce.getGroupAssignment(),
                    m_Constant(&group_assignment_attr)))
    return mlir::emitError(loc, "group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return mlir::emitError(loc, "group_assignment should have two dimensions.");
  int32 group_size = group_assignment_attr.getType().getShape()[1];

  Mesh mesh = output_layout->mesh();
  // This will become more general when Topology is properly defined.
  const bool is_tpu = all_reduce.getDeviceType().ends_with("TPU");

  const int32_t key_base = GetCollectiveKeyBase(mesh, group_assignment_attr);
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
    mlir::Value relative_device_id =
        GetRelativeDeviceId(all_reduce, *output_layout, builder, loc);
    final_op = internal::EmitCollectiveReduce(
        builder, loc, all_reduce.getInput(), all_reduce.getReduceOp().str(),
        group_assignment_attr, key_base, relative_device_id,
        /*host_group_size=*/group_size, all_reduce.getDeviceType().str());
  }
  SetSingleLayoutOnOp(final_op, *output_layout);
  *value = final_op->getResult(0);
  return mlir::success();
}

mlir::LogicalResult LowerAllReduceOp(mlir::MLIRContext& context,
                                     mlir::TF::DTensorAllReduceOp all_reduce) {
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
    return reduce_scatter.emitOpError(output_layout.status().message());
  }
  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(reduce_scatter.getGroupAssignment(),
                    m_Constant(&group_assignment_attr)))
    return reduce_scatter.emitOpError("group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return reduce_scatter.emitOpError(
        "group_assignment should have two dimensions.");
  mlir::DenseIntElementsAttr scatter_attr;
  if (!matchPattern(reduce_scatter.getScatterDimension(),
                    m_Constant(&scatter_attr))) {
    return reduce_scatter.emitOpError(
        "Scatter dimension not constant integer array.");
  }
  int32 scatter_dim = (*scatter_attr.begin()).getSExtValue();

  mlir::OpBuilder builder(reduce_scatter);
  if (reduce_scatter.getDeviceType().ends_with("TPU")) {
    // For TPUs, lower to XlaReduceScatter straightforwardly.
    mlir::Operation* xla_reduce_scatter =
        builder.create<mlir::TF::XlaReduceScatterOp>(
            loc, reduce_scatter.getResult().getType(),
            reduce_scatter.getInput(), reduce_scatter.getGroupAssignment(),
            reduce_scatter.getScatterDimension(),
            reduce_scatter.getReduceOpAttr());
    SetSingleLayoutOnOp(xla_reduce_scatter, *output_layout);
    reduce_scatter.replaceAllUsesWith(xla_reduce_scatter);
  } else if (reduce_scatter.getDeviceType().ends_with("GPU") &&
             UseNcclCommunicationOnGpu()) {
    // Use CollectiveReduceScatterV2 which has a NCCL GPU implementation.
    mlir::Value relative_device_id =
        GetRelativeDeviceId(reduce_scatter, *output_layout, builder, loc);

    int32 group_size = group_assignment_attr.getType().getShape()[1];
    const int32_t key_base =
        GetCollectiveKeyBase((*output_layout).mesh(), group_assignment_attr);

    mlir::Operation* collective_op = EmitCollectiveReduceScatter(
        builder, loc, reduce_scatter.getInput(),
        reduce_scatter.getResult().getType(),
        reduce_scatter.getReduceOp().str(), group_assignment_attr, scatter_dim,
        key_base, relative_device_id,
        /*host_group_size=*/group_size, reduce_scatter.getDeviceType().str());
    SetSingleLayoutOnOp(collective_op, *output_layout);
    reduce_scatter.replaceAllUsesWith(collective_op);
  } else {
    // For CPU and non-NCCL GPU devices, decompose to
    // DTensorAllReduce+DTensorAllScatter.
    // TODO(tmorris): Once CollectiveReduceScatterV2 has a non-NCCL
    // implementation, remove this path.
    StatusOr<Layout> input_layout =
        ExtractRequiredLayoutFromOperand(reduce_scatter.getInput());
    if (!input_layout.ok()) {
      // If input layout is not defined, modify the output_layout based on the
      // scattered dimension.
      std::vector<string> input_sharding_spec =
          output_layout->sharding_spec_strs();
      input_sharding_spec[scatter_dim] = Layout::kUnshardedDim;
      input_layout =
          Layout::GetLayout(input_sharding_spec, output_layout->mesh());
    }

    if (!input_layout.ok()) {
      return reduce_scatter.emitOpError(input_layout.status().message());
    }

    auto dtensor_allreduce = builder.create<mlir::TF::DTensorAllReduceOp>(
        reduce_scatter.getLoc(), reduce_scatter.getOperand(0).getType(),
        reduce_scatter.getOperand(0), reduce_scatter.getGroupAssignment(),
        reduce_scatter.getReduceOp(), reduce_scatter.getDeviceType());
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

StatusOr<const mlir::DenseIntElementsAttr> GetGroupAssignment(
    mlir::OpBuilder builder, const Layout src_layout,
    absl::flat_hash_set<std::string> reduced_dims) {
  std::vector<int32> partitions_flat;
  TF_ASSIGN_OR_RETURN(
      auto all_partitions,
      GetAllReducePartitionsFromReducedDims(src_layout, reduced_dims));

  const int32 num_partitions = all_partitions.size();
  for (auto& p : all_partitions) {
    if (p.second.size() != all_partitions.begin()->second.size()) {
      return errors::InvalidArgument(
          "partitions had different sizes -- "
          "this is not supported in MLIR.");
    }
    partitions_flat.insert(partitions_flat.end(), p.second.begin(),
                           p.second.end());
  }

  const int32 partition_size = all_partitions.begin()->second.size();

  const mlir::RankedTensorType shaped_type = mlir::RankedTensorType::get(
      {num_partitions, partition_size},
      mlir::IntegerType::get(builder.getContext(), 32));
  const mlir::DenseIntElementsAttr group_assignment =
      mlir::DenseIntElementsAttr::get(shaped_type, partitions_flat);
  if (group_assignment.getType().getRank() != 2) {
    return errors::InvalidArgument(
        "group_assignment should have two dimensions.");
  }
  return group_assignment;
}

mlir::LogicalResult LowerAllGatherOpToCollective(
    mlir::TF::DTensorAllGatherOp all_gather) {
  const Layout src_layout = all_gather.getInputLayout();
  const Layout tgt_layout = all_gather.getOutputLayout();
  mlir::OpBuilder builder(all_gather);
  builder.setInsertionPointAfter(all_gather);

  const mlir::Location loc = DT_LOC(all_gather.getLoc());

  mlir::Value relative_device_id =
      GetRelativeDeviceId(all_gather, tgt_layout, builder, loc);

  StatusOr<std::string> device_type_or_status =
      DeviceTypeFromMesh(src_layout.mesh());
  if (!device_type_or_status.ok())
    return all_gather.emitOpError() << device_type_or_status.status().message();
  const std::string device_type = device_type_or_status.value();

  const mlir::RankedTensorType input_type =
      mlir::dyn_cast<mlir::RankedTensorType>(all_gather.getInput().getType());
  const mlir::RankedTensorType output_type =
      mlir::dyn_cast<mlir::RankedTensorType>(all_gather.getOutput().getType());

  if (!input_type)
    return all_gather.emitOpError() << "input type is not a RankedTensorType";
  if (!output_type)
    return all_gather.emitOpError() << "output type is not a RankedTensorType";

  const std::vector<int64_t> output_shape = output_type.getShape();
  const std::vector<int64_t> input_shape = input_type.getShape();

  mlir::Value prev_op_result = all_gather.getInput();

  absl::flat_hash_set<std::string> dims_to_gather;

  std::vector<int32> num_shards_per_dim;
  absl::flat_hash_map<int32, int32> previous_sharded_dim;
  int32 last_sharded_dim = 0;
  std::vector<int64_t> input_shape_after_tr;

  std::vector<int64> perm_for_transpose;
  perm_for_transpose.reserve(src_layout.rank());
  for (int i = 0; i < src_layout.rank(); i++) {
    perm_for_transpose.push_back(i);
  }

  for (int i = 0; i < src_layout.rank(); i++) {
    if (src_layout.num_shards_for_dim(i) == tgt_layout.num_shards_for_dim(i) ||
        src_layout.num_shards_for_dim(i) == 1) {
      continue;
    }

    int64 temp = perm_for_transpose[0];
    perm_for_transpose[0] = perm_for_transpose[i];
    perm_for_transpose[i] = temp;

    num_shards_per_dim.push_back(src_layout.num_shards_for_dim(i));
    previous_sharded_dim[i] = last_sharded_dim;
    last_sharded_dim = i;

    input_shape_after_tr.insert(input_shape_after_tr.begin(), input_shape[i]);
    dims_to_gather.insert(src_layout.sharding_spec(i));
  }
  auto pre_transpose_op =
      EmitTransposeOp(builder, loc, prev_op_result, perm_for_transpose);
  prev_op_result = pre_transpose_op->getResult(0);

  auto group_assignment_or =
      GetGroupAssignment(builder, src_layout, dims_to_gather);
  if (!group_assignment_or.ok()) {
    return all_gather.emitOpError() << group_assignment_or.status().message();
  }
  auto group_assignment = group_assignment_or.value();
  int32 group_size = group_assignment.getType().getShape()[1];
  int32 key_base = GetCollectiveKeyBase(tgt_layout.mesh(), group_assignment);
  auto collective_op =
      EmitCollectiveGather(builder, loc, prev_op_result, group_assignment,
                           key_base, relative_device_id,
                           /*host_group_size=*/group_size, device_type);

  prev_op_result = collective_op->getResult(0);
  if (num_shards_per_dim.size() > 1) {
    std::vector<int64> new_shape;
    new_shape.reserve(input_shape.size() + num_shards_per_dim.size());
    for (int j = 0; j < num_shards_per_dim.size(); j++) {
      new_shape.push_back(num_shards_per_dim[j]);
    }

    for (int j = 0; j < input_shape_after_tr.size(); j++) {
      new_shape.push_back(input_shape_after_tr[j]);
    }

    auto reshape_op = builder.create<mlir::TF::ReshapeOp>(
        loc, /*tensor=*/collective_op->getResult(0),
        /*shape=*/ops_util::GetR1Const(new_shape, builder, loc));

    prev_op_result = reshape_op->getResult(0);
    for (int i = src_layout.rank() - 1; i >= 0; i--) {
      if (src_layout.num_shards_for_dim(i) ==
              tgt_layout.num_shards_for_dim(i) ||
          src_layout.num_shards_for_dim(i) == 1) {
        continue;
      }

      // Transpose based on sharding. Sharded dims are updated in the front
      // before calling collective.
      std::vector<int64> perm_arr = {};
      // for (int j = 0; j <= src_layout.rank(); j++) {
      perm_arr.reserve(new_shape.size());
      for (int j = 0; j < new_shape.size(); j++) {
        perm_arr.push_back(j);
      }

      if (i != previous_sharded_dim[i]) {
        for (int j = i + 1; j < new_shape.size(); j++) {
          perm_arr[j] = j - 1;
        }
        perm_arr[i] = new_shape.size() - 1;
      }
      auto tr_op = EmitTransposeOp(builder, loc, prev_op_result, perm_arr);
      prev_op_result = tr_op->getResult(0);
    }
  } else {
    auto post_transpose_op =
        EmitTransposeOp(builder, loc, prev_op_result, perm_for_transpose);
    prev_op_result = post_transpose_op->getResult(0);
  }

  auto output_reshape_op = builder.create<mlir::TF::ReshapeOp>(
      loc, /*tensor=*/prev_op_result,
      /*shape=*/ops_util::GetR1Const(output_shape, builder, loc));
  SetSingleLayoutOnOp(output_reshape_op, tgt_layout);
  all_gather.replaceAllUsesWith(output_reshape_op->getResult(0));
  all_gather.erase();
  return mlir::success();
}

mlir::LogicalResult LowerAllGatherOp(mlir::TF::DTensorAllGatherOp all_gather) {
  const Layout src_layout = all_gather.getInputLayout();
  const Layout tgt_layout = all_gather.getOutputLayout();

  llvm::SmallVector<int64, 4> concat_dims;
  for (int64 i = 0; i < src_layout.rank(); ++i)
    if (src_layout.num_shards_for_dim(i) > 1 &&
        Layout::IsUnshardedDimension(tgt_layout.sharding_spec(i)))
      concat_dims.push_back(i);

  mlir::OpBuilder builder(all_gather);
  builder.setInsertionPointAfter(all_gather);

  if (concat_dims.empty()) {
    mlir::TF::IdentityOp identity = builder.create<mlir::TF::IdentityOp>(
        all_gather.getLoc(), all_gather.getInput().getType(),
        all_gather.getInput());
    SetSingleLayoutOnOp(identity, tgt_layout);

    all_gather.getOutput().replaceAllUsesWith(identity);
    all_gather.erase();
    return mlir::success();
  }

  const mlir::RankedTensorType input_type =
      mlir::dyn_cast<mlir::RankedTensorType>(all_gather.getInput().getType());
  const mlir::RankedTensorType output_type =
      mlir::dyn_cast<mlir::RankedTensorType>(all_gather.getOutput().getType());

  if (!input_type)
    return all_gather.emitOpError() << "input type is not a RankedTensorType";
  if (!output_type)
    return all_gather.emitOpError() << "output type is not a RankedTensorType";

  if (!LowerCollectiveGatherToCollectiveGatherV2() ||
      src_layout.mesh().is_tpu_mesh()) {
    // Use existing Reduce flow for TPU mesh and when explicitly enabled.
  } else if (input_type.getElementType().isInteger(32) ||
             input_type.getElementType().isInteger(64) ||
             input_type.getElementType().isF16() ||
             input_type.getElementType().isF32() ||
             input_type.getElementType().isF64()) {
    // CollectiveGatherV2 does not support any other data type.
    return LowerAllGatherOpToCollective(all_gather);
  } else {
    // Use existing reduce flow for unsupported data types.
  }

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
            src_layout.mesh().device_location(device_id);
        if (!device_loc_or_status.ok())
          return all_gather.emitOpError()
                 << device_loc_or_status.status().message();
        const DeviceLocation device_loc = device_loc_or_status.value();
        const int32 mesh_idx =
            src_layout.mesh().idx_for_dim(src_layout.sharding_spec(i)).value();
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
  mlir::Value relative_device_id =
      GetRelativeDeviceId(all_gather, src_layout, builder, loc);
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
        /*update=*/all_gather.getInput(), /*indices=*/begin);
  } else {
    update_result = builder.create<mlir::TF::TensorStridedSliceUpdateOp>(
        loc, zeros.getType(),
        /*input=*/zeros, begin, end, strides,
        /*value=*/all_gather.getInput());
  }

  // All reduce among concatenated dimensions.
  absl::flat_hash_set<std::string> reduced_dims;
  for (int i : concat_dims) reduced_dims.insert(src_layout.sharding_spec(i));

  auto partitions_or_status =
      GetAllReducePartitionsFromReducedDims(src_layout, reduced_dims);
  if (!partitions_or_status.ok())
    return all_gather.emitOpError() << partitions_or_status.status().message();
  auto partitions = partitions_or_status.value();
  const int32 num_partitions = partitions.size();
  assert(num_partitions <= num_devices);
  if (num_partitions == num_devices) {
    // TODO(unknown): Is this check needed? Since we check that num_shards for
    // each reduced_dims in the src layout is > 1, I think we always need
    // communication.
    // If every device lives in its own partition, we don't need to emit a
    // collective.
    SetSingleLayoutOnOp(update_result.getDefiningOp(), tgt_layout);
    all_gather.getOutput().replaceAllUsesWith(update_result);
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
    return all_gather.emitOpError() << device_type_or_status.status().message();
  const std::string device_type = device_type_or_status.value();

  // Support bool types by switching to Any reduce rather than Add. For each
  // position in the tensor, only one task in the reduction group can have a 1.
  // This is sufficient.
  const mlir::TensorType type =
      mlir::dyn_cast<mlir::TensorType>(update_result.getType());
  absl::string_view reduce_type = kReduceOpAdd;
  if (type && type.getElementType().isInteger(1)) reduce_type = kReduceOpAny;
  mlir::TF::DTensorAllReduceOp all_reduce =
      builder.create<mlir::TF::DTensorAllReduceOp>(
          loc, update_result.getType(), update_result,
          builder.create<mlir::TF::ConstOp>(loc, group_assignment),
          builder.getStringAttr(std::string(reduce_type)),
          builder.getStringAttr(device_type));
  SetSingleLayoutOnOp(all_reduce, tgt_layout);

  all_gather.getOutput().replaceAllUsesWith(all_reduce.getResult());
  all_gather.erase();
  return mlir::LogicalResult::success();
}

mlir::LogicalResult LowerAllScatterOp(
    mlir::TF::DTensorAllScatterOp all_scatter) {
  const Layout original_layout = all_scatter.getInputLayout();
  const Layout desired_layout = all_scatter.getOutputLayout();

  mlir::tf_device::ClusterOp cluster =
      all_scatter->getParentOfType<mlir::tf_device::ClusterOp>();
  StatusOr<mlir::Value> mesh_coordinates_status =
      GetMeshCoordinatesFromCluster(cluster);
  if (!mesh_coordinates_status.ok())
    return all_scatter.emitOpError()
           << mesh_coordinates_status.status().message();
  mlir::Value mesh_coordinates = mesh_coordinates_status.value();

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
      mlir::dyn_cast<mlir::RankedTensorType>(all_scatter.getOutput().getType());
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

  // We need to softly place the DT_INT32 MatMulOp for GPUs.
  if (original_layout.mesh().is_gpu_mesh() ||
      desired_layout.mesh().is_gpu_mesh()) {
    // TODO(b/303662238): See whether we can replicate soft placement here.
    offset->setAttr(kDeviceAttr, builder.getStringAttr(kCpuDevice));
  }

  // Input to slice needs to be rank 1, so we need to squeeze it.
  mlir::TF::SqueezeOp offset_squeezed = builder.create<mlir::TF::SqueezeOp>(
      all_scatter.getLoc(),
      mlir::RankedTensorType::get({original_layout.rank()},
                                  builder.getIntegerType(32)),
      offset.getProduct(), builder.getI64ArrayAttr({0}));

  auto result = builder.create<mlir::TF::SliceOp>(
      all_scatter.getLoc(), output_type, all_scatter.getInput(),
      offset_squeezed.getOutput(), slice_shape_value);

  SetSingleLayoutOnOp(result, desired_layout);

  all_scatter.getOutput().replaceAllUsesExcept(result.getOutput(), result);
  all_scatter.erase();

  return mlir::LogicalResult::success();
}

mlir::LogicalResult LowerAllToAllOp(mlir::TF::DTensorAllToAllOp all_to_all) {
  mlir::OpBuilder builder(all_to_all);
  mlir::Location loc = all_to_all.getLoc();
  const Layout src_layout = all_to_all.getInputLayout();
  const Layout tgt_layout = all_to_all.getOutputLayout();

  absl::flat_hash_set<std::string> dims_to_gather;
  for (int i = 0; i < src_layout.rank(); i++) {
    if (src_layout.num_shards_for_dim(i) == tgt_layout.num_shards_for_dim(i) ||
        src_layout.num_shards_for_dim(i) == 1) {
      continue;
    }
    dims_to_gather.insert(src_layout.sharding_spec(i));
  }

  auto group_assignment_or =
      GetGroupAssignment(builder, src_layout, dims_to_gather);
  if (!group_assignment_or.ok()) {
    return all_to_all.emitOpError() << group_assignment_or.status().message();
  }
  auto group_assignment = group_assignment_or.value();
  int32 group_size = group_assignment.getType().getShape()[1];

  StatusOr<std::string> device_type_or_status =
      DeviceTypeFromMesh(src_layout.mesh());
  if (!device_type_or_status.ok())
    return all_to_all.emitOpError() << device_type_or_status.status().message();
  const std::string device_type = device_type_or_status.value();

  // Find concat and split dimensions
  int32 split_dimension = -1;
  int32 concat_dimension = -1;
  for (int i = 0; i < src_layout.rank(); ++i) {
    if (src_layout.sharding_spec(i) != tgt_layout.sharding_spec(i)) {
      if (Layout::IsUnshardedDimension(src_layout.sharding_spec(i)) &&
          Layout::IsShardedDimension(tgt_layout.sharding_spec(i))) {
        split_dimension = i;
      } else if (Layout::IsShardedDimension(src_layout.sharding_spec(i)) &&
                 Layout::IsUnshardedDimension(tgt_layout.sharding_spec(i))) {
        concat_dimension = i;
      }
    }
  }
  if (split_dimension == -1 || concat_dimension == -1) {
    return all_to_all.emitOpError();
  }

  if (mlir::StringRef(device_type).ends_with("TPU")) {
    // For TPUs, lower to XlaAllToAll.
    mlir::Operation* xla_all_to_all = builder.create<mlir::TF::AllToAllOp>(
        loc, all_to_all.getResult().getType(), all_to_all.getInput(),
        builder.create<mlir::TF::ConstOp>(loc, group_assignment),
        concat_dimension, split_dimension, group_size);
    SetSingleLayoutOnOp(xla_all_to_all, tgt_layout);
    all_to_all.replaceAllUsesWith(xla_all_to_all);
  } else {
    // Use CollectiveAllToAllV2
    mlir::Value relative_device_id =
        GetRelativeDeviceId(all_to_all, tgt_layout, builder, loc);
    int32 key_base = GetCollectiveKeyBase(tgt_layout.mesh(), group_assignment);

    mlir::Operation* collective_op = EmitCollectiveAllToAll(
        builder, loc, all_to_all.getInput(), group_assignment, concat_dimension,
        split_dimension, key_base, relative_device_id, group_size, device_type);
    SetSingleLayoutOnOp(collective_op, tgt_layout);
    all_to_all.replaceAllUsesWith(collective_op);
  }
  all_to_all.erase();
  return mlir::LogicalResult::success();
}

}  // namespace internal

namespace {
struct DTensorAllReduceLowering
    : public impl::DTensorAllReduceLoweringBase<DTensorAllReduceLowering> {
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
      if (mlir::failed(internal::LowerAllReduceOp(context, all_reduce)))
        return signalPassFailure();
  }
};

struct DTensorReduceScatterLowering
    : public impl::DTensorReduceScatterLoweringBase<
          DTensorReduceScatterLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Find all DTensorAllReduce ops.
    llvm::SmallVector<mlir::TF::DTensorReduceScatterOp, 4> all_reduces;
    module.walk([&](mlir::TF::DTensorReduceScatterOp all_reduce) {
      all_reduces.emplace_back(all_reduce);
    });

    // Replace every DTensorAllReduce op with device-specific implementations.
    for (auto& all_reduce : all_reduces)
      if (mlir::failed(internal::LowerReduceScatterOp(all_reduce)))
        return signalPassFailure();
  }
};

struct DTensorAllGatherLowering
    : public impl::DTensorAllGatherLoweringBase<DTensorAllGatherLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Process all DTensorAllGather ops.
    llvm::SmallVector<mlir::TF::DTensorAllGatherOp, 4> all_gathers;
    module.walk([&](mlir::TF::DTensorAllGatherOp all_gather) {
      all_gathers.emplace_back(all_gather);
    });

    for (mlir::TF::DTensorAllGatherOp all_gather : all_gathers)
      if (mlir::failed(internal::LowerAllGatherOp(all_gather)))
        return signalPassFailure();
  }
};

struct DTensorAllScatterLowering
    : public impl::DTensorAllScatterLoweringBase<DTensorAllScatterLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Process all DTensorAllScatter ops.
    llvm::SmallVector<mlir::TF::DTensorAllScatterOp, 4> all_scatters;
    module.walk([&](mlir::TF::DTensorAllScatterOp all_scatter) {
      all_scatters.emplace_back(all_scatter);
    });

    for (mlir::TF::DTensorAllScatterOp all_scatter : all_scatters)
      if (mlir::failed(internal::LowerAllScatterOp(all_scatter)))
        return signalPassFailure();
  }
};

struct DTensorAllToAllLowering
    : public impl::DTensorAllToAllLoweringBase<DTensorAllToAllLowering> {
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    // Find all DTensorAllToAll ops.
    llvm::SmallVector<mlir::TF::DTensorAllToAllOp, 4> all_to_alls;
    module.walk([&](mlir::TF::DTensorAllToAllOp all_to_all) {
      all_to_alls.emplace_back(all_to_all);
    });

    // Replace every DTensorAllToAll op with device-specific implementations.
    for (auto& all_to_all : all_to_alls)
      if (mlir::failed(internal::LowerAllToAllOp(all_to_all)))
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorAllToAllLoweringPass() {
  return std::make_unique<DTensorAllToAllLowering>();
}

}  // namespace dtensor
}  // namespace tensorflow
