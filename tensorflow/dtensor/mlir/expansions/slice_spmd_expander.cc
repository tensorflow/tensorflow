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

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

Status GetSliceOpArguments(mlir::TF::SliceOp slice_op,
                           llvm::SmallVector<int64_t, 4>& begins,
                           bool& dynamic_begins,
                           llvm::SmallVector<int64_t, 4>& sizes) {
  Status begins_result = ExtractConstVectorFromValue(slice_op.begin(), &begins);
  dynamic_begins = !begins_result.ok();

  TF_RETURN_WITH_CONTEXT(ExtractConstVectorFromValue(slice_op.size(), &sizes),
                         "expected constant argument for SliceOp::size()");

  return OkStatus();
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

  LayoutProto proposed_proto;
  *proposed_proto.mutable_mesh_config() = layout.mesh().ToProto();
  for (int64_t i = 0; i < rank; ++i) {
    // Slice performed on replicated dimension translates to local expansion.
    if (num_shards[i] == 1) {
      proposed_proto.add_sharding_specs()->set_sharding_spec(
          Layout::kUnshardedDim);
      continue;
    }

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
      proposed_proto.add_sharding_specs()->set_sharding_spec(
          layout.sharding_spec(i));
    } else {
      // Slicing on sharded dim is not trivial. Propose an unsharded dim for
      // that.
      proposed_proto.add_sharding_specs()->set_sharding_spec(
          Layout::kUnshardedDim);
    }
  }
  return Layout::FromProto(proposed_proto);
}

llvm::SmallVector<int64_t, 4> CalculateBitVector(const uint64_t mask_value) {
  llvm::SmallVector<int64_t, 4> bit_vector;
  bit_vector.resize(sizeof(uint64_t) * 8, 0);
  for (int i = 0; i < sizeof(uint64_t) * 8; ++i) {
    bit_vector[i] = (mask_value >> i & 1);
  }
  return bit_vector;
}

// The begin/end/stride and the masks are all sized to mach the number of
// entries in the slice specification. E.g. [:, ..., 3] will have a begin/end/
// stride of size 3 and the max set bit in the mask will be the 3rd bit.
// This function converts this specifications into ones relative to the input
// tensor.
// We also output a bool vector of the input indices which are not shrunk away.
// These always must be replicated, since shrinking an index means we took a
// single element along that axis and it must be present on all cores.
// spec_to_input maps the 'spec' dimensions to the input dimensions. This is
// needed so we can create a new 'end' input for the SPMD expanded op.
//
// NOTE: If the begin or ends are dynamic, they will be size 0.
// If strides is dynamic it will be the correct rank but contain 0s (an invalid
// stride).
template <typename T>
Status GetInputOrientedData(T strided_slice,
                            llvm::SmallVectorImpl<int64_t>* begin,
                            uint64_t* begin_mask,
                            llvm::SmallVectorImpl<int64_t>* end,
                            uint64_t* end_mask,
                            llvm::SmallVectorImpl<int64_t>* strides,
                            llvm::SmallVectorImpl<bool>* not_shrunk,
                            llvm::SmallVectorImpl<int64>* spec_to_input) {
  begin->resize(0);
  end->resize(0);
  strides->resize(0);

  llvm::SmallVector<int64_t, 4> spec_begin;
  llvm::SmallVector<int64_t, 4> spec_end;
  llvm::SmallVector<int64_t, 4> spec_strides;

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> strides_shape,
                      GetShapeOfValue(strided_slice.strides(),
                                      /*fail_on_dynamic=*/true));
  if (strides_shape.size() != 1)
    return errors::InvalidArgument(
        "strides input to strided operation is not rank 1");

  int64_t spec_rank = strides_shape[0];
  spec_to_input->resize(spec_rank, -1);

  if (!ExtractConstVectorFromValue(strided_slice.strides(), &spec_strides).ok())
    spec_strides.resize(spec_rank, 0);

  if (ExtractConstVectorFromValue(strided_slice.begin(), &spec_begin).ok())
    if (spec_begin.size() != spec_rank)
      return errors::InvalidArgument(
          "rank of begin input to strided operation does not equal rank of "
          "strides input");

  if (ExtractConstVectorFromValue(strided_slice.end(), &spec_end).ok())
    if (spec_end.size() != spec_rank)
      return errors::InvalidArgument(
          "rank of end input to strided operation does not equal rank of "
          "strides input");

  const uint64_t new_axis_mask = strided_slice.new_axis_mask();
  const uint64_t shink_axis_mask = strided_slice.shrink_axis_mask();
  const uint64_t spec_begin_mask = strided_slice.begin_mask();
  const uint64_t spec_end_mask = strided_slice.end_mask();
  uint64_t ellipsis_mask = strided_slice.ellipsis_mask();

  int64_t input_rank;
  if (mlir::isa<mlir::TF::StridedSliceOp>(strided_slice) ||
      mlir::isa<mlir::TF::TensorStridedSliceUpdateOp>(strided_slice)) {
    // For StridedSlice the first operand is the input.
    input_rank = ValueRank(strided_slice->getOperand(0));
  } else if (mlir::isa<mlir::TF::StridedSliceGradOp>(strided_slice)) {
    // For StridedSliceGrad the first operand is the shape of the input.
    TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> input_shape,
                        GetShapeOfValue(strided_slice->getOperand(0)));
    if (input_shape.size() != 1)
      return errors::InvalidArgument("input shape must be rank 1");
    input_rank = input_shape[0];
  }

  if (absl::popcount(ellipsis_mask) > 1)
    return errors::InvalidArgument(
        "strided slice only supports at most one ellipsis");

  // Count the number of axes after the ellipsis
  bool found_ellipsis = false;
  int64_t num_add_axis_after_ellipsis = 0;
  for (int64_t i = 0; i < spec_rank; ++i) {
    if (found_ellipsis && ((1 << i) & new_axis_mask))
      num_add_axis_after_ellipsis++;
    if ((1 << i) & ellipsis_mask) found_ellipsis = true;
  }
  // Guarantee one ellipsis. If there isn't one, add it at the end of the spec.
  // If we do this, add one to the total rank so that we process the ellipsis as
  // part of the loop below.
  if (!found_ellipsis) ellipsis_mask |= (1 << (spec_rank++));

  // At this point total rank cannot be more than input_rank + number of
  // new axes plus the number of ellipses. Check that condition so that we know
  // the loop below won't have input_index >= input_rank.
  if (spec_rank > input_rank + absl::popcount(new_axis_mask) + 1)
    return errors::InvalidArgument(
        "incompatible input rank, number of new axis and specification rank: ",
        input_rank, ", ", absl::popcount(new_axis_mask), ", ", spec_rank);

  int64_t input_index = 0;
  for (int64_t spec_index = 0; spec_index < spec_rank; ++spec_index) {
    if ((1 << spec_index) & ellipsis_mask) {
      const int64_t next_input_index =
          std::min(input_rank - (spec_rank - spec_index) + 1 +
                       num_add_axis_after_ellipsis,
                   input_rank);
      for (; input_index < next_input_index; input_index++) {
        // For input axes within the ellipsis region, we include the entire axis
        // by setting the begin and end mask.
        not_shrunk->emplace_back(true);
        if (!spec_begin.empty()) begin->emplace_back(0);
        if (!spec_end.empty()) end->emplace_back(0);
        strides->emplace_back(1);
        (*begin_mask) |= 1 << input_index;
        (*end_mask) |= 1 << input_index;
      }
    } else if (((1 << spec_index) & new_axis_mask) == 0) {
      not_shrunk->emplace_back(((1 << spec_index) & shink_axis_mask) == 0);
      if (!spec_begin.empty()) begin->emplace_back(spec_begin[spec_index]);
      if (!spec_end.empty()) end->emplace_back(spec_end[spec_index]);
      strides->emplace_back(spec_strides[spec_index]);
      (*spec_to_input)[spec_index] = input_index;
      (*begin_mask) |= ((spec_begin_mask >> spec_index) & 1) << input_index;
      (*end_mask) |= ((spec_end_mask >> spec_index) & 1) << input_index;
      input_index++;
    }
  }

  // This should not happen.
  if (input_index != input_rank)
    return errors::Internal("strided slice input not totally processed");

  return OkStatus();
}

// Return an intermediate layout for StridedSlice(Grad), where we can lower the
// global StridedSlice(Grad) to a local one.
// All the inputs (begin/end/stride/masks) are sized to match the 'total rank'
// which is the rank of the input rank + number of new dimensions added (e.g
// the number of bits set in the new_axis_mask).
// The values of these inputs on the 'newly added' dimensions are ignored.
// global_input_shape is the global shape for the main input of StridedSlice or
// equivalently the global shape of the output of StridedSliceGrad.
// If new_end is not a nullptr, it will be set to the new ending vector if
// the end was constant, otherwise it will be cleared.
template <typename T>
StatusOr<Layout> GetStridedSliceIntermediateLayout(
    T strided_slice, const Layout& layout,
    const llvm::ArrayRef<int64_t> global_input_shape,
    llvm::SmallVectorImpl<int64_t>* new_end = nullptr) {
  const int64_t rank = global_input_shape.size();

  // Records if the corresponding axis of the input can be sharded.
  llvm::SmallVector<bool, 4> can_shard;
  // Lists the start/end of the slice. Value is otherwise clamped to the correct
  // range.
  llvm::SmallVector<int64_t, 4> begin;
  llvm::SmallVector<int64_t, 4> end;
  // Lists the stride for each tensor dimension. Positive when its constant and
  // 0 when its dynamic.
  llvm::SmallVector<int64_t, 4> strides;
  llvm::SmallVector<int64_t, 4> total_to_input;
  // The current number of shards long each axis;
  const std::vector<int32> shards = layout.num_shards();

  uint64_t begin_mask = 0;
  uint64_t end_mask = 0;

  TF_RETURN_IF_ERROR(GetInputOrientedData(strided_slice, &begin, &begin_mask,
                                          &end, &end_mask, &strides, &can_shard,
                                          &total_to_input));

  bool const_begin = !begin.empty();
  bool const_end = !end.empty();

  if (!const_begin) begin.resize(rank, 0);

  if (!const_end) end.resize(rank, 0);

  for (int i = 0; i < rank; ++i) {
    if ((1 << i) & begin_mask)
      begin[i] = 0;
    else if (begin[i] < 0)
      begin[i] += global_input_shape[i];

    if (begin[i] < 0l) {
      begin[i] = 0l;
    } else if (begin[i] > global_input_shape[i] - 1) {
      begin[i] = global_input_shape[i] - 1;
    }

    if ((1 << i) & end_mask)
      end[i] = global_input_shape[i];
    else if (end[i] < 0)
      end[i] += global_input_shape[i];

    if (end[i] < 1l) {
      end[i] = 1l;
    } else if (end[i] > global_input_shape[i]) {
      end[i] = global_input_shape[i];
    }

    // Negative and dynamic stride requires unsharded axis.
    if (strides[i] < 1) can_shard[i] = false;
    // The local size must be divisible by the stride, otherwise the begin
    // for each local slice would be different.
    if ((global_input_shape[i] / shards[i]) % strides[i] != 0)
      can_shard[i] = false;
    // If start or end are dynamic we can't shard.
    if (!(((1 << i) & begin_mask) || const_begin) ||
        !(((1 << i) & end_mask) || const_end))
      can_shard[i] = false;
    // Finally if amount of space left on 'left' and 'right' of the tensor
    // is more than (or equal to) a stride then we can't shard as there would be
    // an unequal number of outputs per shard.
    // NOTE: the case of end[i] == begin[i] may be a simple optimization since
    // the result is an empty tensor.
    if (global_input_shape[i] - (end[i] - begin[i]) >= strides[i])
      can_shard[i] = false;
    // If there is currently no sharding, it doesn't make sense to shard.
    if (shards[i] == 1) can_shard[i] = false;
  }

  // Compute the new 'end' for the slice. Note that this end needs to be in
  // terms of the 'total' index not the input index (i.e. it needs 'bogus'
  // entries for the new axes).
  if (new_end != nullptr) {
    if (!const_end) {
      // Dynamic end are unchanged. We indicate this by ensuring the passed in
      // is empty;
      new_end->clear();
    } else {
      new_end->resize(total_to_input.size());
      for (int i = 0; i < total_to_input.size(); ++i) {
        const int64_t inp = total_to_input[i];
        if (inp != -1) {
          // If we can keep input axis input_index sharded, we need to update
          // the end. Given the conditions we enforeced above, we can set end to
          // the local size of input.
          if (can_shard[inp])
            (*new_end)[i] = global_input_shape[inp] / shards[inp];
          else
            (*new_end)[i] = end[inp];
        }
      }
    }
  }

  // Compute the new layout, its basically the old layout but replicated on some
  // axis.
  absl::flat_hash_set<int> reduced_dims;
  for (int i = 0; i < rank; ++i)
    if (!can_shard[i]) reduced_dims.emplace(i);
  return layout.GetLayoutWithReducedDims(reduced_dims, /*keep_dims=*/true);
}

enum Direction {
  FORWARD,
  BACKWARD,
};

// Applies the shrink and new masks to a layout. This function works in both the
// forwards and backwards direction as specified in the direction argument.
template <typename SliceOpT>
StatusOr<Layout> ApplyNewAndShrinkMasksToLayout(SliceOpT slice_op,
                                                const int input_rank,
                                                const int output_rank,
                                                const Layout& proposed_layout,
                                                const Direction direction) {
  // Calculate bit mask for shrunk dimensions/newly added dimensions.
  const llvm::SmallVector<int64_t, 4> new_axis_mask =
      CalculateBitVector(slice_op.new_axis_mask());
  const llvm::SmallVector<int64_t, 4> shrink_axis_mask =
      CalculateBitVector(slice_op.shrink_axis_mask());

  std::vector<std::string> sharding_spec;
  int input_dim_index = 0;
  int output_dim_index = 0;
  int current_dimension_index = 0;
  while (current_dimension_index < proposed_layout.rank()) {
    if (input_dim_index < input_rank &&
        shrink_axis_mask[input_dim_index] == 1) {
      input_dim_index++;
      if (direction == BACKWARD)
        sharding_spec.emplace_back(Layout::kUnshardedDim);
      else
        current_dimension_index++;
    } else if (output_dim_index < output_rank &&
               new_axis_mask[output_dim_index] == 1) {
      if (direction == FORWARD)
        sharding_spec.emplace_back(Layout::kUnshardedDim);
      else
        current_dimension_index++;
      output_dim_index++;
    } else {
      sharding_spec.emplace_back(
          proposed_layout.sharding_spec(current_dimension_index));
      input_dim_index++;
      output_dim_index++;
      current_dimension_index++;
    }
  }

  const auto& mask = (direction == FORWARD) ? new_axis_mask : shrink_axis_mask;
  // New dimensions may be added after all dimensions have been sliced.
  while (current_dimension_index < mask.size() &&
         mask[current_dimension_index] == 1) {
    sharding_spec.emplace_back(Layout::kUnshardedDim);
    current_dimension_index++;
  }

  return Layout::GetLayout(sharding_spec, proposed_layout.mesh());
}

mlir::Value IntConstWithMatchingType(mlir::OpBuilder& builder,
                                     mlir::Location loc,
                                     llvm::ArrayRef<int64_t> values,
                                     mlir::Type type) {
  if (type.cast<mlir::RankedTensorType>().getElementType().isInteger(64)) {
    return Int64Const(builder, loc, values);
  } else {
    llvm::SmallVector<int32, 4> values32(values.begin(), values.end());
    return IntConst(builder, loc, values32);
  }
}

}  // namespace

StatusOr<mlir::Operation*> SliceSPMDExpander::ExpandOp(mlir::Operation* op) {
  auto slice_op = mlir::cast<mlir::TF::SliceOp>(op);
  TF_ASSIGN_OR_RETURN(auto input_layout,
                      ExtractLayoutFromOperand(slice_op.input()));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));

  if (!output_layout || !input_layout)
    return errors::Unimplemented(
        "layout of Slice op must be known before SPMD expansion.");

  // The dyn_cast will never be nullptr as it is checked in
  // GetLayoutFromOperands.
  auto input_type =
      slice_op.input().getType().dyn_cast<mlir::RankedTensorType>();
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
                      VerifySliceLayout(slice_op, slice_op.input(),
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
  if (!slice_op.begin().getType().isa<mlir::ShapedType>())
    return errors::Internal("type of begin is not a ShapedType");
  mlir::ShapedType type = slice_op.begin().getType().cast<mlir::ShapedType>();
  if (type.getElementType().isInteger(32))
    new_size = IntConst(
        builder, loc, llvm::SmallVector<int32, 4>(sizes.begin(), sizes.end()));
  else
    new_size = Int64Const(builder, loc, sizes);

  auto new_op =
      builder
          .create<mlir::TF::SliceOp>(loc, slice_op.output().getType(),
                                     relayout_input, slice_op.begin(), new_size)
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
      VerifySliceLayout(slice_op, slice_op.input(), input_layout));
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
        VerifySliceLayout(slice_op, slice_op.output(), output_layout));
    input_layouts[0] = proposed_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> StridedSliceSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto strided_slice_op = mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(Layout input_layout, ExtractRequiredLayoutFromOperand(
                                               strided_slice_op.input()));
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(
      const llvm::ArrayRef<int64_t> global_input_shape,
      GetGlobalShapeOfValueFromDTensorLayout(strided_slice_op.input()));

  llvm::SmallVector<int64_t, 4> end;
  TF_ASSIGN_OR_RETURN(
      Layout intermediate_input_layout,
      GetStridedSliceIntermediateLayout(strided_slice_op, input_layout,
                                        global_input_shape, &end));

  TF_ASSIGN_OR_RETURN(mlir::Value new_input,
                      EmitRelayout(strided_slice_op.input(), input_layout,
                                   intermediate_input_layout));

  strided_slice_op.inputMutable().assign(new_input);

  mlir::OpBuilder builder(op);

  if (!end.empty()) {
    mlir::Value new_end =
        IntConstWithMatchingType(builder, strided_slice_op.getLoc(), end,
                                 strided_slice_op.begin().getType());
    strided_slice_op.endMutable().assign(new_end);
  }

  op = InferSPMDExpandedLocalShape(op);

  // Compute the layout of the output after the local StridedSlice takes place.
  const int input_rank = global_input_shape.size();
  const int output_rank = ValueRank(strided_slice_op.output());

  // Calculate bit mask for shrinked dimensions/newly added dimensions.
  const llvm::SmallVector<int64_t, 4> new_axis_mask =
      CalculateBitVector(strided_slice_op.new_axis_mask());
  const llvm::SmallVector<int64_t, 4> shrink_axis_mask =
      CalculateBitVector(strided_slice_op.shrink_axis_mask());

  TF_ASSIGN_OR_RETURN(
      Layout intermediate_output_layout,
      ApplyNewAndShrinkMasksToLayout(strided_slice_op, input_rank, output_rank,
                                     intermediate_input_layout, FORWARD));

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(
      mlir::Value output,
      EmitRelayout(strided_slice_op.output(), intermediate_output_layout,
                   output_layout, &newly_created_ops));

  strided_slice_op.output().replaceAllUsesExcept(output, newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  mlir::TF::StridedSliceOp strided_slice_op =
      mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.input(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = global_input_shape.size();
  const int output_rank = ValueRank(strided_slice_op.output());

  const Layout& input_layout = input_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(Layout proposed_layout,
                      GetStridedSliceIntermediateLayout(
                          strided_slice_op, input_layout, global_input_shape));
  // If dimension was added or removed, create a new proposed output layout
  // with dimensions added/skipped.
  TF_ASSIGN_OR_RETURN(
      proposed_layout,
      ApplyNewAndShrinkMasksToLayout(strided_slice_op, input_rank, output_rank,
                                     proposed_layout, FORWARD));
  return llvm::DenseMap<int, Layout>({{0, proposed_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::StridedSliceOp strided_slice_op =
      mlir::cast<mlir::TF::StridedSliceOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.input(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = global_input_shape.size();
  const int output_rank = ValueRank(strided_slice_op.output());

  llvm::DenseMap<int, Layout> input_layouts(strided_slice_op.getNumOperands());
  // Set replicated layout for begin, end, and strides operands.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // input
  if (output_layouts.find(0) != output_layouts.end()) {
    // This layout must exist (as there is only one output).
    const Layout& output_layout = output_layouts.lookup(0);
    // If dimension was added or removed, take the current output layout, and
    // add/skip dimensions in it as needed to get an input layout.
    TF_ASSIGN_OR_RETURN(
        Layout proposed_layout,
        ApplyNewAndShrinkMasksToLayout(strided_slice_op, input_rank,
                                       output_rank, output_layout, BACKWARD));
    TF_ASSIGN_OR_RETURN(proposed_layout, GetStridedSliceIntermediateLayout(
                                             strided_slice_op, proposed_layout,
                                             global_input_shape));
    input_layouts[0] = proposed_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> TensorStridedSliceUpdateSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      llvm::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_op.input()));
  TF_ASSIGN_OR_RETURN(
      const Layout value_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_op.value()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  TF_ASSIGN_OR_RETURN(
      const llvm::ArrayRef<int64_t> global_input_shape,
      GetGlobalShapeOfValueFromDTensorLayout(strided_slice_op.input()));

  const int input_rank = global_input_shape.size();
  const int value_rank = ValueRank(strided_slice_op.value());

  llvm::SmallVector<int64_t, 4> end;
  TF_ASSIGN_OR_RETURN(
      Layout intermediate_input_layout,
      GetStridedSliceIntermediateLayout(strided_slice_op, input_layout,
                                        global_input_shape, &end));

  TF_ASSIGN_OR_RETURN(
      Layout intermediate_value_layout,
      ApplyNewAndShrinkMasksToLayout(strided_slice_op, input_rank, value_rank,
                                     intermediate_input_layout, FORWARD));

  TF_ASSIGN_OR_RETURN(mlir::Value new_input,
                      EmitRelayout(strided_slice_op.input(), input_layout,
                                   intermediate_input_layout));

  TF_ASSIGN_OR_RETURN(mlir::Value new_value,
                      EmitRelayout(strided_slice_op.value(), value_layout,
                                   intermediate_value_layout));

  strided_slice_op.inputMutable().assign(new_input);
  strided_slice_op.valueMutable().assign(new_value);

  mlir::OpBuilder builder(op);

  if (!end.empty()) {
    mlir::Value new_end =
        IntConstWithMatchingType(builder, strided_slice_op.getLoc(), end,
                                 strided_slice_op.begin().getType());
    strided_slice_op.endMutable().assign(new_end);
  }

  op = InferSPMDExpandedLocalShape(op);

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(
      mlir::Value output,
      EmitRelayout(strided_slice_op.output(), intermediate_input_layout,
                   output_layout, &newly_created_ops));

  strided_slice_op.output().replaceAllUsesExcept(output, newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorStridedSliceUpdateSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout and value layout are missing, don't return an output
  // layout.
  if (input_layouts.find(0) == input_layouts.end() &&
      input_layouts.find(4) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      mlir::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.input(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = global_input_shape.size();
  const int value_rank = ValueRank(strided_slice_op.value());

  // We have a choice to determine the output layout, we will default to use
  // input_layout if available, otherwise we will expand value_layout and use
  // that.
  Layout input_layout;
  if (input_layouts.find(0) != input_layouts.end()) {
    input_layout = input_layouts.lookup(0);
  } else {
    // When we don't have the input layout, use value layout to 'create' the
    // input layout. We do this by applying the new and shrink masks backwards.
    // This is because in the case of a normal strided slice the layout of
    // value would be output layout.
    const Layout& value_layout = input_layouts.lookup(4);
    TF_ASSIGN_OR_RETURN(input_layout, ApplyNewAndShrinkMasksToLayout(
                                          strided_slice_op, input_rank,
                                          value_rank, value_layout, BACKWARD));
  }
  TF_ASSIGN_OR_RETURN(Layout proposed_output_layout,
                      GetStridedSliceIntermediateLayout(
                          strided_slice_op, input_layout, global_input_shape));

  return llvm::DenseMap<int, Layout>({{0, proposed_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorStridedSliceUpdateSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::TensorStridedSliceUpdateOp strided_slice_op =
      mlir::cast<mlir::TF::TensorStridedSliceUpdateOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_input_shape,
                      GetShapeOfValue(strided_slice_op.input(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = global_input_shape.size();
  const int value_rank = ValueRank(strided_slice_op.value());

  llvm::DenseMap<int, Layout> input_layouts(strided_slice_op.getNumOperands());
  // Set replicated layout for begin, end, and strides operands.
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // input and value layouts
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout& output_layout = output_layouts.lookup(0);
    TF_ASSIGN_OR_RETURN(
        const Layout proposed_input_layout,
        GetStridedSliceIntermediateLayout(strided_slice_op, output_layout,
                                          global_input_shape));
    input_layouts[0] = proposed_input_layout;

    // We also need a layout for value as well, and for that we just take the
    // input layout and apply the masks.
    // The layout of value is determined from the input layout by applying the
    // new and shrink masks in the forwards direction as value would have been
    // the output layout for a normal strided slice operation.
    TF_ASSIGN_OR_RETURN(
        const Layout proposed_value_layout,
        ApplyNewAndShrinkMasksToLayout(strided_slice_op, input_rank, value_rank,
                                       proposed_input_layout, FORWARD));
    input_layouts[4] = proposed_value_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> StridedSliceGradSPMDExpander::ExpandOp(
    mlir::Operation* op) {
  auto strided_slice_grad_op = llvm::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(
      const Layout input_layout,
      ExtractRequiredLayoutFromOperand(strided_slice_grad_op.dy()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  TF_ASSIGN_OR_RETURN(
      const llvm::ArrayRef<int64_t> global_output_shape,
      GetGlobalShapeOfValueFromDTensorLayout(strided_slice_grad_op.output()));

  const int input_rank = ValueRank(strided_slice_grad_op.dy());
  const int output_rank = global_output_shape.size();

  llvm::SmallVector<int64_t, 4> end;
  TF_ASSIGN_OR_RETURN(
      Layout intermediate_output_layout,
      GetStridedSliceIntermediateLayout(strided_slice_grad_op, output_layout,
                                        global_output_shape, &end));

  TF_ASSIGN_OR_RETURN(Layout intermediate_input_layout,
                      ApplyNewAndShrinkMasksToLayout(
                          strided_slice_grad_op, output_rank, input_rank,
                          intermediate_output_layout, FORWARD));

  TF_ASSIGN_OR_RETURN(mlir::Value new_dy,
                      EmitRelayout(strided_slice_grad_op.dy(), input_layout,
                                   intermediate_input_layout));

  strided_slice_grad_op.dyMutable().assign(new_dy);

  mlir::OpBuilder builder(op);

  if (!end.empty()) {
    mlir::Value new_end =
        IntConstWithMatchingType(builder, strided_slice_grad_op.getLoc(), end,
                                 strided_slice_grad_op.begin().getType());
    strided_slice_grad_op.endMutable().assign(new_end);
  }

  // The shape input to StridedSliceGrad will still be global, so we need to
  // compute the local shape update it.
  std::vector<int64_t> computed_output_shape =
      intermediate_output_layout.LocalShapeFromGlobalShape(global_output_shape);
  mlir::Value new_shape = IntConstWithMatchingType(
      builder, strided_slice_grad_op.getLoc(), computed_output_shape,
      strided_slice_grad_op.begin().getType());
  strided_slice_grad_op.shapeMutable().assign(new_shape);

  op = InferSPMDExpandedLocalShape(op);

  // Do a final relayout to the correct output layout in case there are any
  // differences between intermediate_output_layout and output_layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(
      mlir::Value output,
      EmitRelayout(strided_slice_grad_op.output(), intermediate_output_layout,
                   output_layout, &newly_created_ops));

  strided_slice_grad_op.output().replaceAllUsesExcept(output,
                                                      newly_created_ops);

  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceGradSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If the input layout is missing, don't return an output layout.
  if (input_layouts.find(4) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  mlir::TF::StridedSliceGradOp strided_slice_grad_op =
      mlir::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_output_shape,
                      GetShapeOfValue(strided_slice_grad_op.output(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = ValueRank(strided_slice_grad_op.dy());
  const int output_rank = global_output_shape.size();

  const Layout& input_layout = input_layouts.lookup(4);
  // If dimension was added or removed, take the current output layout, and
  // add/skip dimensions in it as needed to get an input layout.
  TF_ASSIGN_OR_RETURN(
      Layout proposed_layout,
      ApplyNewAndShrinkMasksToLayout(strided_slice_grad_op, output_rank,
                                     input_rank, input_layout, BACKWARD));
  TF_ASSIGN_OR_RETURN(
      proposed_layout,
      GetStridedSliceIntermediateLayout(strided_slice_grad_op, proposed_layout,
                                        global_output_shape));
  return llvm::DenseMap<int, Layout>({{0, proposed_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
StridedSliceGradSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  mlir::TF::StridedSliceGradOp strided_slice_grad_op =
      mlir::cast<mlir::TF::StridedSliceGradOp>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(const llvm::ArrayRef<int64_t> global_output_shape,
                      GetShapeOfValue(strided_slice_grad_op.output(),
                                      /*fail_on_dynamic=*/true));
  const int input_rank = ValueRank(strided_slice_grad_op.dy());
  const int output_rank = global_output_shape.size();

  llvm::DenseMap<int, Layout> input_layouts(
      strided_slice_grad_op.getNumOperands());
  // Set replicated layout for shape, begin, end, stride operands.
  input_layouts[0] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  input_layouts[3] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);

  // dy
  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout& output_layout = output_layouts.lookup(0);
    TF_ASSIGN_OR_RETURN(
        Layout proposed_layout,
        GetStridedSliceIntermediateLayout(strided_slice_grad_op, output_layout,
                                          global_output_shape));

    // If dimension was added or removed, create a new proposed output layout
    // with dimensions added/skipped.
    TF_ASSIGN_OR_RETURN(
        proposed_layout,
        ApplyNewAndShrinkMasksToLayout(strided_slice_grad_op, output_rank,
                                       input_rank, proposed_layout, FORWARD));
    input_layouts[4] = proposed_layout;
  }

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
