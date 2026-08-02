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
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/transforms/simplifiers/reduce_window_util.h"
#include "xla/hlo/utils/hlo_query.h"
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
        ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
        new_inst = builder.AddInstruction(HloInstruction::CreateParameter(
            inst->parameter_number(), shape, inst->name()));
        break;
      }
      case HloOpcode::kTuple:
        new_inst = builder.AddInstruction(
            HloInstruction::CreateTuple(get_mapped_operands(inst)));
        break;
      case HloOpcode::kGetTupleElement: {
        ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
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
        ASSIGN_OR_RETURN(Shape shape, get_scalar_shape(inst->shape()));
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

// Walks through broadcasts, reshapes, and bitcasts to the value that seeds a
// scan. Returns the scalar instruction or the uniform constant behind the
// init, or nullptr when no scalar init value can be derived. Does not modify
// the module.
static HloInstruction* FindScalarInitSource(HloInstruction* init) {
  while (HloPredicateIsOp<HloOpcode::kBroadcast, HloOpcode::kReshape,
                          HloOpcode::kBitcast>(init)) {
    if (init->opcode() == HloOpcode::kBitcast &&
        init->shape().element_type() !=
            init->operand(0)->shape().element_type()) {
      // Bitcast changes element type; cannot extract a scalar init value.
      return nullptr;
    }
    init = init->mutable_operand(0);
  }
  if (ShapeUtil::IsScalar(init->shape())) {
    return init;
  }
  if (init->opcode() == HloOpcode::kConstant && init->literal().IsAllFirst()) {
    return init;
  }
  return nullptr;
}

static size_t FlattenShapeIndex(const ShapeIndex& shape_index) {
  if (shape_index.empty()) {
    return 0;
  }
  CHECK_EQ(shape_index.size(), 1);
  return shape_index.back();
}

// Transposes the inputs if the scan dimension is not the last dimension.
// Returns the permutation of the dimensions.
static std::vector<int64_t> GetTransposedInputs(
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

// Adds padding (if necessary) to enable further rewrites working properly.
static int64_t PreparePaddingForRewrite(
    HloModulePass* pass, int64_t base_length, HloComputation* hlo_computation,
    absl::Span<HloInstruction* const> init_values,
    std::vector<HloInstruction*>& inputs, int64_t scan_length,
    int64_t last_dim) {
  Shape shape = inputs.front()->shape();
  int64_t rank = shape.dimensions().size();

  // getting round up to the base length to ensure that the padded length is a
  // multiple of the base length.
  const int64_t padded_length = RoundUpTo(scan_length, base_length);

  if (scan_length != padded_length) {
    for (size_t input_index = 0; input_index < inputs.size(); ++input_index) {
      HloInstruction* input = inputs[input_index];

      // We already moved scan dimensions to last dimension always -> rank - 1
      Shape padded_shape = input->shape();
      padded_shape.set_dimensions(last_dim, padded_length);
      pass->UpdateLayout(&padded_shape);

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
static int64_t ExpandToNewMajorDimension(
    HloModulePass* pass, int64_t base_length, HloComputation* hlo_computation,
    std::vector<HloInstruction*>& inputs,
    std::vector<HloInstruction*>& tiled_inputs,
    std::vector<Shape>& tiled_shapes, int64_t padded_length, int64_t last_dim) {
  const int64_t num_columns = padded_length / base_length;
  for (auto* input : inputs) {
    Shape tiled_shape = input->shape();
    tiled_shape.set_dimensions(last_dim, num_columns);

    pass->UpdateLayout(&tiled_shape);
    ShapeUtil::AppendMajorDimension(base_length, &tiled_shape);
    tiled_shapes.push_back(tiled_shape);
    tiled_inputs.push_back(hlo_computation->AddInstruction(
        HloInstruction::CreateReshape(tiled_shape, input)));
  }

  return num_columns;
}

// reduce_window ( [x, y/base, base] window [1, 1, base] )
static HloInstruction* GenerateNewReduceWindowWithTiledInputs(
    int64_t base_length, HloComputation* hlo_computation,
    std::vector<HloInstruction*>& tiled_inputs,
    absl::Span<HloInstruction* const> init_values, HloComputation* to_apply,
    std::vector<Shape>& tiled_shapes, bool forward_scan, bool is_tuple_result) {
  const int64_t rank = tiled_inputs.front()->shape().dimensions().size() - 1;

  Window outer_window =
      window_util::MakeWindow(std::vector<int64_t>(rank + 1, 1));
  outer_window.mutable_dimensions(rank)->set_size(base_length);

  if (forward_scan) {
    outer_window.mutable_dimensions(rank)->set_padding_low(base_length - 1);
  } else {
    outer_window.mutable_dimensions(rank)->set_padding_high(base_length - 1);
  }

  return hlo_computation->AddInstruction(HloInstruction::CreateReduceWindow(
      is_tuple_result ? ShapeUtil::MakeTupleShape(tiled_shapes)
                      : tiled_shapes[0],
      tiled_inputs, init_values, outer_window, to_apply));
}

// slices [x, y/base, base] -> [x, y/base, 1] slice {x, y/base}
// reshape [x, y/base, 1] -> [x, y/base]
static void SliceOutLastColumn(HloModulePass* pass, int64_t base_length,
                               HloComputation* hlo_computation,
                               const Shape& subshape,
                               HloInstruction* outer_shape, int64_t rank,
                               int64_t last_dim, bool forward_scan,
                               int64_t num_columns,
                               std::vector<Shape>& column_shapes,
                               std::vector<HloInstruction*>& last_cols) {
  // creating slices [x, y/base, base] -> [x, y/base, 1]
  Shape column_shape = subshape;
  column_shape.set_dimensions(rank, 1);
  pass->UpdateLayout(&column_shape);

  std::vector<int64_t> col_slice_starts(rank + 1, 0);
  std::vector<int64_t> col_slice_limits(SpanToVector(subshape.dimensions()));
  if (forward_scan) {
    col_slice_starts[rank] = base_length - 1;
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
  pass->UpdateLayout(&column_shape);
  column_shapes.push_back(column_shape);
}

static absl::StatusOr<HloInstruction*> RewriteScanAsTreeReduction(
    HloModulePass* pass, int64_t base_length, HloComputation* parent,
    std::vector<HloInstruction*> sources,
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
      pass, base_length, parent, init_values, sources, scan_length, last_dim);

  // 2) Reshape to R(k+1).
  // [x, y] -> [x, y/base, base]
  // In the example above
  // [0 1 2 3 4 5 6 7 8] -> [0 1 2
  //                         3 4 5
  //                         6 7 8]
  std::vector<HloInstruction*> tiled_sources;
  std::vector<Shape> tiled_shapes;
  const int64_t num_columns = ExpandToNewMajorDimension(
      pass, base_length, parent, sources, tiled_sources, tiled_shapes,
      padded_length, last_dim);

  // 3) Outer scan - Scan each "base" dimension.
  // reduce_window ( [x, y/base, base] window [1, 1, base] )
  // scan for each window of {1, base}
  // [0 1 2     [0  1  3
  //  3 4 5  ->  3  7 12
  //  6 7 8]     6 13 21]
  HloInstruction* outer_reduce_window = GenerateNewReduceWindowWithTiledInputs(
      base_length, parent, tiled_sources, init_values, to_apply, tiled_shapes,
      forward_scan, result_shape.IsTuple());

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
            pass, base_length, parent, subshape,
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
  RETURN_IF_ERROR(status);

  HloInstruction* scan;
  if (result_shape.IsTuple()) {
    scan = parent->AddInstruction(HloInstruction::CreateTuple(scans));
  } else {
    CHECK_EQ(scans.size(), 1);
    scan = scans[0];
  }
  return scan;
}

static absl::StatusOr<bool> TryOptimizeCumSumOrProd(
    HloModulePass* pass, int64_t base_length,
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
  if (scan_length <= base_length) {
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
  ASSIGN_OR_RETURN(
      HloInstruction * scan,
      RewriteScanAsTreeReduction(
          pass, base_length, parent, sources, reduce_window->init_values(),
          reduce_window->to_apply(), reduce_window->shape(), rank, scan_dim,
          scan_length, forward_scan, is_exclusive));
  RETURN_IF_ERROR(reduce_window->ReplaceAllUsesWith(scan));
  RETURN_IF_ERROR(parent->RemoveInstruction(reduce_window));
  return true;
}

// Returns true if it is safe to rewrite the scan as a tree reduction with
// `init_source` as the seed. RewriteScanAsTreeReduction folds the init into
// both tree levels, which changes the result unless the extra fold is a
// no-op: the init is the combiner's identity, or the combiner is idempotent
// in the init (min/max), or the init is absorbing.
static bool IsTreeRewriteSafeInit(const HloScanInstruction* scan,
                                  const HloInstruction* init_source) {
  const HloInstruction* root = scan->to_apply()->root_instruction();
  if (root->opcode() != HloOpcode::kTuple || root->operand_count() != 2 ||
      root->operand(0) != root->operand(1) ||
      root->operand(0)->operand_count() != 2) {
    return false;
  }
  switch (root->operand(0)->opcode()) {
    case HloOpcode::kMinimum:
    case HloOpcode::kMaximum:
      // Idempotent: folding any init again is a no-op.
      return true;
    case HloOpcode::kAdd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
      return init_source->IsConstant() && init_source->literal().IsAll(0);
    case HloOpcode::kMultiply:
      // One is the identity, zero is absorbing.
      return init_source->IsConstant() && (init_source->literal().IsAll(1) ||
                                           init_source->literal().IsAll(0));
    default:
      return false;
  }
}

static absl::StatusOr<bool> TryOptimizeAssociativeScan(
    HloModulePass* pass, int64_t base_length, HloScanInstruction* scan) {
  if (!hlo_query::IsStandardAssociativeScan(scan)) {
    return false;
  }
  // The reduce-window rewrite emits a forward cumulative sum and drops the
  // final carry, so reverse scans and scans whose carry is read keep their
  // other lowerings. Non get-tuple-element users are dead
  // (IsStandardAssociativeScan).
  if (scan->is_reverse()) {
    return false;
  }
  for (const HloInstruction* user : scan->users()) {
    if (user->opcode() == HloOpcode::kGetTupleElement &&
        user->tuple_index() == 1 &&
        (user->user_count() > 0 || user->IsRoot())) {
      return false;
    }
  }

  const Shape& operand_shape = scan->inputs()[0]->shape();
  int64_t rank = operand_shape.dimensions().size();
  int64_t scan_dim = scan->scan_dimension();
  int64_t scan_length = operand_shape.dimensions(scan_dim);

  VLOG(2) << "Rewriting associative scan: " << scan->ToString();
  HloComputation* parent = scan->parent();

  // Scans whose init is not a broadcast scalar (e.g. the vector carry seeds
  // the SPMD partitioner builds) cannot be expressed as a reduce-window
  // cumsum: reduce-window inits are scalars. Skip them; ScanExpander lowers
  // them instead. This classification does not modify the module, so gate
  // rejections below leave the module untouched.
  HloInstruction* init_source = FindScalarInitSource(scan->inits()[0]);
  if (init_source == nullptr) {
    return false;
  }

  const bool use_single_reduce_window =
      base_length == 0 || scan_length <= base_length;
  if (!use_single_reduce_window && !IsTreeRewriteSafeInit(scan, init_source)) {
    // The tree rewrite folds the init into both tree levels, which is only
    // correct when the extra fold is a no-op (an identity, idempotent, or
    // absorbing init for the combiner). A single reduce-window handles any
    // scalar init but is quadratic in the scan length, so past base_length
    // the scan is left to ScanExpander.
    return false;
  }

  absl::StatusOr<HloComputation*> scan_to_apply_or =
      ScalarizeComputation(scan->to_apply(), parent);
  if (absl::IsInvalidArgument(scan_to_apply_or.status())) {
    // Bodies that are not elementwise cannot be scalarized into a
    // reduce-window combiner; leave them to ScanExpander. ScalarizeComputation
    // does not modify the module when it fails.
    return false;
  }
  ASSIGN_OR_RETURN(HloComputation * scan_to_apply, std::move(scan_to_apply_or));

  // Every gate has passed; from here on the module is modified. Materialize
  // the scalar init: the scalar instruction itself, or a scalar constant
  // extracted from a uniform higher-rank constant.
  HloInstruction* init =
      ShapeUtil::IsScalar(init_source->shape())
          ? init_source
          : parent->AddInstruction(HloInstruction::CreateConstant(
                LiteralUtil::GetFirstScalarLiteral(init_source->literal())));
  HloComputation::Builder builder(
      absl::StrCat(scan_to_apply->name(), "_rw_wrapper"));

  HloInstruction* carry_param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          0, scan_to_apply->parameter_instruction(1)->shape(), "carry_0"));
  HloInstruction* input_param =
      builder.AddInstruction(HloInstruction::CreateParameter(
          1, scan_to_apply->parameter_instruction(0)->shape(), "input_0"));
  HloInstruction* call = builder.AddInstruction(
      HloInstruction::CreateCall(scan_to_apply->root_instruction()->shape(),
                                 {input_param, carry_param}, scan_to_apply));
  builder.AddInstruction(HloInstruction::CreateGetTupleElement(call, 1));
  HloComputation* rw_to_apply =
      parent->parent()->AddEmbeddedComputation(builder.Build());

  HloInstruction* result = nullptr;
  HloInstruction* input = scan->inputs()[0];
  if (use_single_reduce_window) {
    Window window = window_util::MakeWindow(std::vector<int64_t>(rank, 1));
    window.mutable_dimensions(scan_dim)->set_size(scan_length);
    window.mutable_dimensions(scan_dim)->set_padding_low(scan_length - 1);

    result = parent->AddInstruction(HloInstruction::CreateReduceWindow(
        input->shape(), input, init, window, rw_to_apply));
  } else {
    Shape outputs_shape = scan->shape().tuple_shapes(0);
    ASSIGN_OR_RETURN(
        result, RewriteScanAsTreeReduction(pass, base_length, parent, {input},
                                           {init}, rw_to_apply, outputs_shape,
                                           rank, scan_dim, scan_length,
                                           /*forward_scan=*/true,
                                           /*is_exclusive=*/false));
  }

  // Replace carry with init value, users are guaranteed to be dead.
  HloInstruction* tuple = parent->AddInstruction(
      HloInstruction::CreateTuple({result, scan->inits()[0]}));
  RETURN_IF_ERROR(parent->ReplaceInstruction(scan, tuple));

  return true;
}

absl::StatusOr<bool> ReduceWindowRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  if (base_length_ == 0) {
    return false;
  }

  for (const auto& computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (auto* reduce_window =
              DynCast<HloReduceWindowInstruction>(instruction)) {
        ASSIGN_OR_RETURN(bool result, TryOptimizeCumSumOrProd(
                                          this, base_length_, reduce_window));
        if (result) {
          changed = true;
          continue;
        }
        if (reduce_window->inputs().front()->shape().dimensions().size() == 1) {
          RETURN_IF_ERROR(reduce_window_util::Replace1DReduceWindowWithReshape(
              reduce_window));
          changed = true;
        }
      }
    }
  }
  return changed;
}

absl::StatusOr<bool> AssociativeScanRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (const auto& computation : module->computations(execution_threads)) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (auto* scan = DynCast<HloScanInstruction>(instruction)) {
        ASSIGN_OR_RETURN(bool result,
                         TryOptimizeAssociativeScan(this, base_length_, scan));
        changed |= result;
      }
    }
  }
  return changed;
}

}  // namespace xla
