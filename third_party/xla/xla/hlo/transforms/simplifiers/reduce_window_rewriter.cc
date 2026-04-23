/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/reduce_window_rewriter.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_util.h"
#include "xla/literal_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Returns true if all the shapes in the computation are scalars or tuples of
// scalars. Nested tuples are not supported.
static bool IsAlreadyScalar(const HloComputation* comp) {
  for (const HloInstruction* inst : comp->instructions()) {
    if (inst->shape().IsTuple()) {
      for (const Shape& subshape : inst->shape().tuple_shapes()) {
        if (!ShapeUtil::IsScalar(subshape)) {
          return false;
        }
      }
    } else if (!ShapeUtil::IsScalar(inst->shape())) {
      return false;
    }
  }
  return true;
}

static absl::StatusOr<HloComputation*> ScalarizeComputation(
    HloComputation* comp, HloComputation* parent_for_embedded) {
  if (IsAlreadyScalar(comp)) {
    return comp;
  }

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> replacements;
  HloComputation::Builder builder(absl::StrCat(comp->name(), "_scalarized"));

  auto get_scalar_shape = [](const Shape& shape) -> absl::StatusOr<Shape> {
    if (!shape.IsTuple()) {
      return ShapeUtil::MakeScalarShape(shape.element_type());
    }
    std::vector<Shape> subshapes;
    subshapes.reserve(shape.tuple_shapes().size());
    for (const Shape& subshape : shape.tuple_shapes()) {
      TF_RET_CHECK(!subshape.IsTuple())
          << "Only one level of nesting is supported.";
      subshapes.push_back(ShapeUtil::MakeScalarShape(subshape.element_type()));
    }
    return ShapeUtil::MakeTupleShape(subshapes);
  };

  auto get_mapped_operands = [&](const HloInstruction* inst) {
    std::vector<HloInstruction*> operands;
    operands.reserve(inst->operand_count());
    for (HloInstruction* op : inst->operands()) {
      operands.push_back(replacements[op]);
    }
    return operands;
  };

  for (HloInstruction* inst : comp->MakeInstructionPostOrder()) {
    HloInstruction* new_inst = nullptr;
    switch (inst->opcode()) {
      case HloOpcode::kParameter: {
        TF_ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
        new_inst = builder.AddInstruction(HloInstruction::CreateParameter(
            inst->parameter_number(), shape, inst->name()));
        break;
      }
      case HloOpcode::kTuple:
        new_inst = builder.AddInstruction(
            HloInstruction::CreateTuple(get_mapped_operands(inst)));
        break;
      case HloOpcode::kGetTupleElement: {
        TF_ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
        new_inst = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
            shape, replacements[inst->operand(0)], inst->tuple_index()));
        break;
      }
      case HloOpcode::kConstant:
        if (!inst->shape().IsArray()) {
          return absl::InvalidArgumentError("Constant is not an array.");
        }
        if (ShapeUtil::IsScalar(inst->shape())) {
          new_inst = builder.AddInstruction(
              HloInstruction::CreateConstant(inst->literal().Clone()));
        } else if (inst->literal().IsAllFirst()) {
          new_inst = builder.AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::GetFirstScalarLiteral(inst->literal())));
        } else {
          return absl::InvalidArgumentError("Constant is not uniform.");
        }
        break;
      case HloOpcode::kBroadcast:
        if (!ShapeUtil::IsScalar(inst->operand(0)->shape())) {
          return absl::InvalidArgumentError(
              "Broadcast operand is not a scalar.");
        }
        new_inst = replacements[inst->operand(0)];
        break;
      case HloOpcode::kBitcast:
        if (inst->operand(0)->shape().element_type() !=
            inst->shape().element_type()) {
          return absl::InvalidArgumentError("Bitcast changes element type.");
        }
        new_inst = replacements[inst->operand(0)];
        break;
      case HloOpcode::kReshape:
        new_inst = replacements[inst->operand(0)];
        break;
      default: {
        if (!inst->IsElementwise()) {
          return absl::InvalidArgumentError(
              absl::StrCat("Instruction is not elementwise: ",
                           HloOpcodeString(inst->opcode())));
        }
        TF_ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
        new_inst = builder.AddInstruction(
            inst->CloneWithNewOperands(shape, get_mapped_operands(inst)));
        break;
      }
    }
    replacements[inst] = new_inst;
  }
  return parent_for_embedded->parent()->AddEmbeddedComputation(
      builder.Build(replacements[comp->root_instruction()]));
}

static absl::StatusOr<HloInstruction*> GetScalarInitValue(
    HloInstruction* init, HloComputation* parent) {
  while (HloPredicateIsOp<HloOpcode::kBroadcast, HloOpcode::kReshape,
                          HloOpcode::kBitcast>(init)) {
    if (init->opcode() == HloOpcode::kBitcast &&
        init->shape().element_type() !=
            init->operand(0)->shape().element_type()) {
      return absl::InvalidArgumentError(
          "Bitcast changes element type, cannot extract scalar init value.");
    }
    init = init->mutable_operand(0);
  }
  if (ShapeUtil::IsScalar(init->shape())) {
    return init;
  }
  if (init->opcode() != HloOpcode::kConstant) {
    return absl::InvalidArgumentError("Init value is not a constant.");
  }
  if (!init->literal().IsAllFirst()) {
    return absl::InvalidArgumentError("Init value is a non-uniform constant.");
  }
  return parent->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::GetFirstScalarLiteral(init->literal())));
}

static size_t FlattenShapeIndex(const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return 0;
  }
  CHECK_EQ(shape_index.size(), 1);
  return shape_index.back();
}

std::vector<int64_t> ReduceWindowRewriter::GetTransposedInputs(
    HloComputation* hlo_computation, std::vector<HloInstruction*>& inputs,
    int64_t rank, int64_t scan_dim, int64_t last_dim) {
  std::vector<int64_t> permutation(rank);
  absl::c_iota(permutation, 0);
  if (scan_dim != last_dim) {
    // permute the dimensions.
    permutation[scan_dim] = last_dim;
    permutation[last_dim] = scan_dim;

    // add transpose for each input.
    for (size_t i = 0; i < inputs.size(); ++i) {
      inputs[i] =
          hlo_computation->AddInstruction(HloInstruction::CreateTranspose(
              ShapeUtil::PermuteDimensions(permutation, inputs[i]->shape()),
              inputs[i], permutation));
    }
  }

  return permutation;
}

int64_t ReduceWindowRewriter::PreparePaddingForRewrite(
    HloComputation* hlo_computation,
    absl::Span<HloInstruction* const> init_values,
    std::vector<HloInstruction*>& inputs, int64_t scan_length,
    int64_t last_dim) {
  Shape shape = inputs.front()->shape();
  int64_t rank = shape.dimensions().size();

  // getting round up to the base length to ensure that the padded length is a
  // multiple of the base length.
  const int64_t padded_length = RoundUpTo(scan_length, base_length_);

  if (scan_length != padded_length) {
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
      HloInstruction* input = inputs[input_index];

      // We already moved scan dimensions to last dimension always -> rank - 1
      Shape padded_shape = input->shape();
      padded_shape.set_dimensions(last_dim, padded_length);
      UpdateLayout(&padded_shape);

      // Padding config for only the last dimension.
      std::vector<std::pair<int64_t, int64_t>> padding(rank);
      padding.back() = {0, padded_length - scan_length};

      // Pad the input with the init value.
      inputs[input_index] =
          hlo_computation->AddInstruction(HloInstruction::CreatePad(
              padded_shape, input, init_values[input_index],
              MakeEdgePaddingConfig(padding)));
    }
  }
  return padded_length;
}

// [x, y] -> [x, y/base, base]
int64_t ReduceWindowRewriter::ExpandToNewMajorDimension(
    HloComputation* hlo_computation, std::vector<HloInstruction*>& inputs,
    std::vector<HloInstruction*>& tiled_inputs,
    std::vector<Shape>& tiled_shapes, int64_t padded_length, int64_t last_dim) {
  const int64_t num_columns = padded_length / base_length_;
  for (auto* input : inputs) {
    Shape tiled_shape = input->shape();
    tiled_shape.set_dimensions(last_dim, num_columns);

    UpdateLayout(&tiled_shape);
    ShapeUtil::AppendMajorDimension(base_length_, &tiled_shape);
    tiled_shapes.push_back(tiled_shape);
    tiled_inputs.push_back(hlo_computation->AddInstruction(
        HloInstruction::CreateReshape(tiled_shape, input)));
  }

  return num_columns;
}

// reduce_window ( [x, y/base, base] window [1, 1, base] )
HloInstruction* ReduceWindowRewriter::GenerateNewReduceWindowWithTiledInputs(
    HloComputation* hlo_computation, std::vector<HloInstruction*>& tiled_inputs,
    absl::Span<HloInstruction* const> init_values, HloComputation* to_apply,
    std::vector<Shape>& tiled_shapes, bool forward_scan, bool is_tuple_result) {
  const int64_t rank = tiled_inputs.front()->shape().dimensions().size() - 1;

  Window outer_window =
      window_util::MakeWindow(std::vector<int64_t>(rank + 1, 1));
  outer_window.mutable_dimensions(rank)->set_size(base_length_);

  if (forward_scan) {
    outer_window.mutable_dimensions(rank)->set_padding_low(base_length_ - 1);
  } else {
    outer_window.mutable_dimensions(rank)->set_padding_high(base_length_ - 1);
  }

  return hlo_computation->AddInstruction(HloInstruction::CreateReduceWindow(
      is_tuple_result ? ShapeUtil::MakeTupleShape(tiled_shapes)
                      : tiled_shapes[0],
      tiled_inputs, init_values, outer_window, to_apply));
}

// slices [x, y/base, base] -> [x, y/base, 1] slice {x, y/base}
// reshape [x, y/base, 1] -> [x, y/base]
void ReduceWindowRewriter::SliceOutLastColumn(
    HloComputation* hlo_computation, const Shape& subshape,
    HloInstruction* outer_shape, int64_t rank, int64_t last_dim,
    bool forward_scan, int64_t num_columns, std::vector<Shape>& column_shapes,
    std::vector<HloInstruction*>& last_cols) {
  // creating slices [x, y/base, base] -> [x, y/base, 1]
  Shape column_shape = subshape;
  column_shape.set_dimensions(rank, 1);
  UpdateLayout(&column_shape);

  std::vector<int64_t> col_slice_starts(rank + 1, 0);
  std::vector<int64_t> col_slice_limits(SpanToVector(subshape.dimensions()));
  if (forward_scan) {
    col_slice_starts[rank] = base_length_ - 1;
  } else {
    col_slice_limits[rank] = 1;
  }
  auto last_col = hlo_computation->AddInstruction(HloInstruction::CreateSlice(
      column_shape, outer_shape, col_slice_starts, col_slice_limits,
      std::vector<int64_t>(rank + 1, 1)));

  // we delete the last dimension, it is a simplification because it is 1
  // anyway. reshape [x, y/base, 1] -> [x, y/base]
  column_shape.DeleteDimension(rank);
  last_col = hlo_computation->AddInstruction(
      HloInstruction::CreateReshape(column_shape, last_col));
  last_cols.push_back(last_col);

  column_shape.set_dimensions(last_dim, num_columns + 1);
  UpdateLayout(&column_shape);
  column_shapes.push_back(column_shape);
}

absl::StatusOr<HloInstruction*>
ReduceWindowRewriter::RewriteScanAsTreeReduction(
    HloComputation* parent, std::vector<HloInstruction*> sources,
    absl::Span<HloInstruction* const> init_values, HloComputation* to_apply,
    const Shape& result_shape, int64_t rank, int64_t scan_dim,
    int64_t scan_length, bool forward_scan, bool is_exclusive) {
  const int64_t last_dim = rank - 1;

  // Since we need to tile the scan dimension, it's convenient to have it
  // logically last. If the scan dimension is not the last dimension, we need to
  // transpose the inputs by permuting the dimensions.
  std::vector<int64_t> permutation =
      GetTransposedInputs(parent, sources, rank, scan_dim, last_dim);

  // Break the scan into an "inner" and an "outer" scan - this is basically a
  // tree reduction:
  // (The explanation below assumes an R1 scan for simplicity. For Rk scan, all
  // shapes have k-1 "batch" dimensions that need to be preserved.)
  //
  // 1) If necessary, pad input from {N} to {K}, where K is a multiple of 128.
  // 2) Reshape from {K} to {K / base, base}.
  // 3) Scan each base dimension.
  // 4) Slice out the last column.
  // 5) Exclusive scan across the last column.
  // 6) Broadcast it back into {K / base, base}
  // 7) Add up the results of (3) and (6).
  // 8) Reshape back into {K}
  // 9) Slice off the padding.
  // For reverse scans, we perform the same as forward scans, except: we perform
  // a reverse scan at (3), slice out the first column at (4), and perform an
  // exclusive reverse scan of the first column at (5).

  // For example, consider a cumulative sum over an R1 of length 9, with a base
  // case of 3 instead of 128. Let the input be: [0 1 2 3 4 5 6 7 8]

  // 1) If necessary, pad input from {N} to {K}, where K is a multiple of 128.
  const int64_t padded_length = PreparePaddingForRewrite(
      parent, init_values, sources, scan_length, last_dim);

  // 2) Reshape to R(k+1).
  // [x, y] -> [x, y/base, base]
  // In the example above
  // [0 1 2 3 4 5 6 7 8] -> [0 1 2
  //                         3 4 5
  //                         6 7 8]
  std::vector<HloInstruction*> tiled_sources;
  std::vector<Shape> tiled_shapes;
  const int64_t num_columns = ExpandToNewMajorDimension(
      parent, sources, tiled_sources, tiled_shapes, padded_length, last_dim);

  // 3) Outer scan - Scan each "base" dimension.
  // reduce_window ( [x, y/base, base] window [1, 1, base] )
  // scan for each window of {1, base}
  // [0 1 2     [0  1  3
  //  3 4 5  ->  3  7 12
  //  6 7 8]     6 13 21]
  HloInstruction* outer_reduce_window = GenerateNewReduceWindowWithTiledInputs(
      parent, tiled_sources, init_values, to_apply, tiled_shapes, forward_scan,
      result_shape.IsTuple());

  // 4) Slice out the last column.
  // Slice out the last (first if reverse scan) column.
  // [0  1  3     [ 3
  //  3  7 12  ->  12
  //  6 13 21]     21]
  std::vector<Shape> column_shapes;
  std::vector<HloInstruction*> last_cols;
  ShapeUtil::ForEachSubshape(
      outer_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return;
        }

        // slices [x, y/base, base] -> [x, y/base, 1] slice {x, y/base}
        // reshape [x, y/base, 1] -> [x, y/base]
        SliceOutLastColumn(
            parent, subshape,
            /*outer_shape=*/
            reduce_window_util::GetAtIndex(outer_reduce_window, shape_index),
            rank, last_dim, forward_scan, num_columns, column_shapes,
            last_cols);
      });

  // 5) Inner scan - Exclusive scan for the last column.
  //  [ 3       [ 0
  //   12   ->    3
  //   21]       15]
  Window inner_window = window_util::MakeWindow(std::vector<int64_t>(rank, 1));
  inner_window.mutable_dimensions(last_dim)->set_size(num_columns);
  if (forward_scan) {
    inner_window.mutable_dimensions(last_dim)->set_padding_low(num_columns);
  } else {
    inner_window.mutable_dimensions(last_dim)->set_padding_high(num_columns);
  }
  auto inner_reduce_window =
      parent->AddInstruction(HloInstruction::CreateReduceWindow(
          result_shape.IsTuple() ? ShapeUtil::MakeTupleShape(column_shapes)
                                 : column_shapes[0],
          last_cols, init_values, inner_window, to_apply));
  std::vector<int64_t> exclusive_slice_starts(rank, 0);
  std::vector<int64_t> exclusive_slice_limits =
      SpanToVector(column_shapes[0].dimensions());
  if (forward_scan) {
    exclusive_slice_limits[last_dim] = num_columns;
  } else {
    exclusive_slice_starts[last_dim] = 1;
    exclusive_slice_limits[last_dim] = num_columns + 1;
  }

  // 6) Broadcast it back into {K / 128, 128}
  // [ 0      [ 0  0  0
  //   3  ->    3  3  3
  //  15]      15 15 15]
  std::vector<HloInstruction*> inner_scan_components;
  ShapeUtil::ForEachSubshape(
      inner_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(inner_reduce_window->shape(),
                                    shape_index)) {
          return;
        }
        size_t idx = FlattenShapeIndex(shape_index);
        auto last_col = last_cols[idx];
        auto* inner_slice = parent->AddInstruction(HloInstruction::CreateSlice(
            last_col->shape(),
            reduce_window_util::GetAtIndex(inner_reduce_window, shape_index),
            exclusive_slice_starts, exclusive_slice_limits,
            std::vector<int64_t>(rank, 1)));

        std::vector<int64_t> rank_iota(rank);
        absl::c_iota(rank_iota, 0);
        auto* inner_scan_component =
            parent->AddInstruction(HloInstruction::CreateBroadcast(
                tiled_shapes[idx], inner_slice, rank_iota));
        inner_scan_components.push_back(inner_scan_component);
      });

  // Combine inner and outer scans.
  std::vector<HloInstruction*> map_operands;
  ShapeUtil::ForEachSubshape(
      outer_reduce_window->shape(),
      [&](const Shape& subshape, const ShapeIndex& shape_index) {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return;
        }
        map_operands.push_back(
            reduce_window_util::GetAtIndex(outer_reduce_window, shape_index));
      });
  map_operands.insert(map_operands.end(), inner_scan_components.begin(),
                      inner_scan_components.end());

  // Finally, we add up the two scans (3) and (6), getting (7):
  // [ 0  1  3
  //   6 10 15
  //  21 28 36]
  //
  // And reshape back into [0 1 3 6 10 15 21 28 36].
  std::vector<HloInstruction*> scans;

  // Reshape back to Rk and slice out the padding.
  auto status = ShapeUtil::ForEachSubshapeWithStatus(
      outer_reduce_window->shape(),
      [&](const Shape& subshape,
          const ShapeIndex& shape_index) -> absl::Status {
        if (!ShapeUtil::IsLeafIndex(outer_reduce_window->shape(),
                                    shape_index)) {
          return absl::OkStatus();
        }
        size_t idx = FlattenShapeIndex(shape_index);
        auto source = sources[idx];
        HloComputation* map_computation;
        auto reduce_function_root = to_apply->root_instruction();
        if (reduce_function_root->shape().IsTuple()) {
          // This corresponds to step 7: combining the inner scan with the outer
          // scan using a map function.
          // We add (apply map function) up the two scans (3) and (6), getting
          // (7):
          //      ( [0  1  3    [ 0  0  0  )     [ 0  1  3
          // map  (  3  7 12      3  3  3  )  ->   6 10 15
          // (+)  (  6 13 21]    15 15 15] )      21 28 36]
          auto* map_computation_root = reduce_function_root->operand(idx);
          absl::flat_hash_map<const HloInstruction*,
                              std::unique_ptr<HloInstruction>>
              replacements;
          replacements[reduce_function_root] = nullptr;
          map_computation = parent->parent()->AddEmbeddedComputation(
              to_apply->CloneWithReplacements(&replacements,
                                              /*extra_parameters=*/{}, nullptr,
                                              "clone", map_computation_root));
        } else {
          map_computation = to_apply;
        }
        auto scan = parent->AddInstruction(HloInstruction::CreateMap(
            reduce_window_util::ShapeAtIndex(outer_reduce_window->shape(),
                                             shape_index),
            map_operands, map_computation));
        scan = parent->AddInstruction(
            HloInstruction::CreateReshape(source->shape(), scan));

        // If necessary, transpose back to the original order.
        if (scan_dim != last_dim) {
          scan = parent->AddInstruction(HloInstruction::CreateTranspose(
              ShapeUtil::PermuteDimensions(permutation, source->shape()), scan,
              permutation));
        }

        // Remove the padding to the base length.
        if (padded_length != scan_length) {
          Shape slice_shape =
              reduce_window_util::ShapeAtIndex(result_shape, shape_index);
          slice_shape.set_dimensions(scan_dim, scan_length);
          scan = parent->AddInstruction(HloInstruction::CreateSlice(
              slice_shape, scan, std::vector<int64_t>(rank, 0),
              slice_shape.dimensions(), std::vector<int64_t>(rank, 1)));
        }

        if (is_exclusive) {
          auto padding_config = MakeNoPaddingConfig(rank);
          if (forward_scan) {
            padding_config.mutable_dimensions(scan_dim)->set_edge_padding_low(
                1);
          } else {
            padding_config.mutable_dimensions(scan_dim)->set_edge_padding_high(
                1);
          }
          scan = parent->AddInstruction(HloInstruction::CreatePad(
              reduce_window_util::ShapeAtIndex(result_shape, shape_index), scan,
              init_values[idx], padding_config));
        }
        scans.push_back(scan);
        return absl::OkStatus();
      });
  TF_RETURN_IF_ERROR(status);

  HloInstruction* scan;
  if (result_shape.IsTuple()) {
    scan = parent->AddInstruction(HloInstruction::CreateTuple(scans));
  } else {
    CHECK_EQ(scans.size(), 1);
    scan = scans[0];
  }
  return scan;
}

absl::StatusOr<bool> ReduceWindowRewriter::TryOptimizeCumSumOrProd(
    HloReduceWindowInstruction* reduce_window) {
  const Shape& operand_shape = reduce_window->inputs().front()->shape();

  // Try to find the scan axis. We expect all window dimensions to be trivial,
  // except for one.
  const int64_t rank = operand_shape.dimensions().size();
  const Window& window = reduce_window->window();
  std::vector<int64_t> non_trivial_window_dimensions =
      reduce_window->non_trivial_window_dimensions();

  // If there are multiple non-trivial window dimensions or no non-trivial
  // window dimensions, we cannot optimize.
  if (non_trivial_window_dimensions.size() != 1) {
    VLOG(2) << "ReduceWindowRewriter: Cannot optimize the reduce window "
               "because of the number of non-trivial window dimensions: "
            << reduce_window->ToString();
    return false;
  }
  const int64_t scan_dim = non_trivial_window_dimensions.front();
  const int64_t scan_length = operand_shape.dimensions(scan_dim);

  // Early checks to avoid unnecessary work.
  if (scan_length <= base_length_) {
    return false;
  }
  if (reduce_window->to_apply()->root_instruction()->shape().IsTuple() &&
      reduce_window->to_apply()->root_instruction()->opcode() !=
          HloOpcode::kTuple) {
    return false;
  }

  const WindowDimension& scan_window_dim = window.dimensions(scan_dim);
  bool forward_scan = (scan_window_dim.padding_low() == scan_length - 1 ||
                       scan_window_dim.padding_low() == scan_length) &&
                      scan_window_dim.padding_high() == 0;
  bool reverse_scan = (scan_window_dim.padding_high() == scan_length - 1 ||
                       scan_window_dim.padding_high() == scan_length) &&
                      scan_window_dim.padding_low() == 0;
  // We accept two values for low padding: the input length for exclusive scan,
  // and scan_length - 1 for inclusive scan.
  if (scan_window_dim.stride() != 1 || scan_window_dim.size() != scan_length ||
      (!forward_scan && !reverse_scan) || scan_window_dim.window_reversal() ||
      scan_window_dim.base_dilation() != 1 ||
      scan_window_dim.window_dilation() != 1) {
    return false;
  }
  bool is_exclusive = forward_scan
                          ? (scan_window_dim.padding_low() == scan_length)
                          : (scan_window_dim.padding_high() == scan_length);

  VLOG(2) << "Rewriting Scan: " << reduce_window->ToString();
  HloComputation* parent = reduce_window->parent();
  std::vector<HloInstruction*> sources(reduce_window->inputs().begin(),
                                       reduce_window->inputs().end());

  // We don't actually need to match the computation - this transformation will
  // work for a commutative/associative reducer, which is what we assume for
  // ReduceWindow anyway.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * scan,
      RewriteScanAsTreeReduction(parent, sources, reduce_window->init_values(),
                                 reduce_window->to_apply(),
                                 reduce_window->shape(), rank, scan_dim,
                                 scan_length, forward_scan, is_exclusive));
  TF_RETURN_IF_ERROR(reduce_window->ReplaceAllUsesWith(scan));
  TF_RETURN_IF_ERROR(parent->RemoveInstruction(reduce_window));
  return true;
}

absl::StatusOr<bool> ReduceWindowRewriter::TryOptimizeAssociativeScan(
    HloScanInstruction* scan) {
  if (scan->is_associative() != TRI_STATE_TRUE) {
    return false;
  }

  const Shape& operand_shape = scan->inputs()[0]->shape();
  int64_t rank = operand_shape.dimensions().size();
  int64_t scan_dim = scan->scan_dimension();
  int64_t scan_length = operand_shape.dimensions(scan_dim);

  if (scan_length <= base_length_) {
    return false;
  }

  VLOG(2) << "Rewriting associative scan: " << scan->ToString();
  HloComputation* parent = scan->parent();
  std::vector<HloInstruction*> sources(scan->inputs().begin(),
                                       scan->inputs().end());
  std::vector<HloInstruction*> inits;
  inits.reserve(scan->inits().size());
  for (HloInstruction* init : scan->inits()) {
    absl::StatusOr<HloInstruction*> scalar_init =
        GetScalarInitValue(init, parent);
    if (!scalar_init.ok()) {
      return false;
    }
    inits.push_back(*scalar_init);
  }

  int64_t num_carries = scan->num_carries();
  int64_t num_outputs = scan->shape().IsTuple()
                            ? scan->shape().tuple_shapes().size() - num_carries
                            : 1 - num_carries;

  absl::StatusOr<HloComputation*> scan_to_apply_scalar =
      ScalarizeComputation(scan->to_apply(), parent);
  if (!scan_to_apply_scalar.ok()) {
    return false;
  }
  HloComputation* scan_to_apply = *scan_to_apply_scalar;

  HloComputation::Builder builder(
      absl::StrCat(scan_to_apply->name(), "_rw_wrapper"));
  int64_t num_inputs = sources.size();

  std::vector<HloInstruction*> wrapper_params;
  wrapper_params.resize(num_carries + num_inputs);

  for (int64_t i = 0; i < num_carries; ++i) {
    wrapper_params[i] = builder.AddInstruction(HloInstruction::CreateParameter(
        i, scan_to_apply->parameter_instruction(num_inputs + i)->shape(),
        absl::StrCat("carry_", i)));
  }
  for (int64_t i = 0; i < num_inputs; ++i) {
    wrapper_params[num_carries + i] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            num_carries + i, scan_to_apply->parameter_instruction(i)->shape(),
            absl::StrCat("input_", i)));
  }

  std::vector<HloInstruction*> call_operands;
  call_operands.reserve(num_inputs + num_carries);
  for (int64_t i = 0; i < num_inputs; ++i) {
    call_operands.push_back(wrapper_params[num_carries + i]);
  }
  for (int64_t i = 0; i < num_carries; ++i) {
    call_operands.push_back(wrapper_params[i]);
  }

  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(scan_to_apply->root_instruction()->shape(),
                                 call_operands, scan_to_apply));

  if (num_carries == 1) {
    builder.AddInstruction(
        HloInstruction::CreateGetTupleElement(call, num_outputs));
  } else {
    std::vector<HloInstruction*> carry_results;
    carry_results.reserve(num_carries);
    for (int64_t i = 0; i < num_carries; ++i) {
      carry_results.push_back(builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(call, num_outputs + i)));
    }
    builder.AddInstruction(HloInstruction::CreateTuple(carry_results));
  }

  HloComputation* rw_to_apply =
      parent->parent()->AddEmbeddedComputation(builder.Build());

  Shape outputs_shape;
  if (num_outputs == 1 && !scan->shape().IsTuple()) {
    outputs_shape = scan->shape();
  } else if (num_outputs == 1) {
    outputs_shape = scan->shape().tuple_shapes(0);
  } else {
    std::vector<Shape> output_shapes;
    output_shapes.reserve(num_outputs);
    for (int i = 0; i < num_outputs; ++i) {
      output_shapes.push_back(scan->shape().tuple_shapes(i));
    }
    outputs_shape = ShapeUtil::MakeTupleShape(output_shapes);
  }

  bool forward_scan = !scan->is_reverse();
  TF_ASSIGN_OR_RETURN(HloInstruction * result,
                      RewriteScanAsTreeReduction(
                          parent, sources, inits, rw_to_apply, outputs_shape,
                          rank, scan_dim, scan_length, forward_scan,
                          /*is_exclusive=*/false));

  std::vector<HloInstruction*> final_results;
  for (int i = 0; i < num_outputs; ++i) {
    if (outputs_shape.IsTuple()) {
      final_results.push_back(parent->AddInstruction(
          HloInstruction::CreateGetTupleElement(result, i)));
    } else {
      final_results.push_back(result);
    }
  }

  for (int i = 0; i < num_outputs; ++i) {
    HloInstruction* out = final_results[i];
    std::vector<int64_t> starts(rank, 0);
    std::vector<int64_t> limits(out->shape().dimensions().begin(),
                                out->shape().dimensions().end());
    std::vector<int64_t> strides(rank, 1);
    if (forward_scan) {
      starts[scan_dim] = scan_length - 1;
    } else {
      limits[scan_dim] = 1;
    }
    Shape slice_shape = out->shape();
    slice_shape.set_dimensions(scan_dim, 1);
    HloInstruction* carry = parent->AddInstruction(
        HloInstruction::CreateSlice(slice_shape, out, starts, limits, strides));
    Shape carry_shape = carry->shape();
    carry_shape.DeleteDimension(scan_dim);
    carry = parent->AddInstruction(
        HloInstruction::CreateReshape(carry_shape, carry));
    final_results.push_back(carry);
  }

  HloInstruction* final_result;
  if (scan->shape().IsTuple()) {
    final_result =
        parent->AddInstruction(HloInstruction::CreateTuple(final_results));
  } else {
    final_result = final_results[0];
  }

  TF_RETURN_IF_ERROR(scan->ReplaceAllUsesWith(final_result));
  TF_RETURN_IF_ERROR(parent->RemoveInstruction(scan));
  return true;
}

absl::StatusOr<bool> ReduceWindowRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto& computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (auto* scan = DynCast<HloScanInstruction>(instruction)) {
        auto result = TryOptimizeAssociativeScan(scan);
        TF_RETURN_IF_ERROR(result.status());
        if (*result) {
          changed = true;
        }
        continue;
      }

      if (auto* reduce_window =
              DynCast<HloReduceWindowInstruction>(instruction)) {
        auto result = TryOptimizeCumSumOrProd(reduce_window);
        TF_RETURN_IF_ERROR(result.status());
        if (*result) {
          changed = true;
          continue;
        }
        if (reduce_window->inputs().front()->shape().dimensions().size() == 1) {
          TF_RETURN_IF_ERROR(
              reduce_window_util::Replace1DReduceWindowWithReshape(
                  reduce_window));
          changed = true;
        }
        continue;
      }
    }
  }
  return changed;
}

}  // namespace xla
