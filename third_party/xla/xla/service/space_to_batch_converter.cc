/* Copyright 2018 The OpenXLA Authors.

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
#include "xla/service/space_to_batch_converter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/algorithm.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/bitmap.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"

namespace xla {

namespace {

namespace m = match;

// ConvolutionVisitor traverses the HLO computation and rewrites Convolution
// operations with small batch counts into convolutions with larger batch
// counts by moving space to batch.
class ConvolutionVisitor {
 public:
  // Top-level function to begin space-to-batch conversion.
  absl::Status PerformSpaceToBatchOnConvolution(HloInstruction* convolution);

  // Struct containing details about a convolution.
  struct ConvDetails {
    std::vector<int64_t> spatial_dimensions_to_split;
    int64_t inherent_low_padding, inherent_high_padding, stride, spatial_size,
        base_dilation_factor, halo_size, high_padding_for_conv,
        low_padding_for_conv, kernel_spatial_dim_size, input_dim_size;
  };

  // Return a struct containing various necessary information pieces for
  // performing space-to-batch on a convolution.
  ConvDetails GetConvolutionDetails(HloInstruction* convolution,
                                    ConvolutionDimensionNumbers& dim_numbers);

  // Returns the set of old and new spatial dimensions respectively.
  std::pair<std::vector<int64_t>, std::vector<int64_t>> GetSpatialDimsToSplit(
      HloInstruction* old_operand);

  // Returns if the convolution is a forward window dilated convolution.
  bool IsForwardWindowDilatedConv(HloInstruction* convolution,
                                  ConvolutionDimensionNumbers& dim_numbers);

  // Function that determines if space-to-batch can be propagated into the
  // consumer. Such propagation is only possible when all required operands are
  // space-to-batch'ed.
  bool CanPropagate(HloInstruction* consumer, HloInstruction* producer);

  // Returns true if the op has all its direct and indirect operands being
  // created via broadcasts. Consumer uses op, and is space-to-batched.
  // instructions_to_transform returns the reverse post order instruction graph.
  bool IsBroadcastTree(HloInstruction* op, HloInstruction* consumer,
                       std::vector<HloInstruction*>& instructions_to_transform);

  // Replicates the broadcast tree with space-to-batched instructions.
  void RewriteBroadcastTree(
      HloInstruction* producer,
      std::vector<HloInstruction*>& instructions_to_transform);

  // Propagate space-to-batch on a broadcast instruction.
  void PropagateOnBroadcast(HloInstruction* consumer, HloInstruction* producer);

  // Returns false if the opcode should definitely not be propagated upon.
  bool IsOpcodeNonPropagatable(HloInstruction* consumer);

  // This function checks if the HLO instruction supports propagation.
  bool SupportedOpForPropagation(HloInstruction* consumer,
                                 HloInstruction* producer);
  bool SupportedDotForPropagation(HloInstruction* consumer,
                                  HloInstruction* producer);

  // Method that checks validity of Broadcast propagation.
  bool IsBroadcastPropagatable(HloInstruction* broadcast,
                               HloInstruction* old_other_op);

  // Propagates space-to-batch on the op, and returns a bool that indicates if
  // the users of the op need to be propagated through.
  absl::StatusOr<bool> Propagate(HloInstruction* consumer,
                                 HloInstruction* producer);

  // Splits the given spatial dimension on the activations and returns the
  // new instructions, and the dimension permutation of the new shape.
  absl::StatusOr<std::pair<HloInstruction*, std::vector<int64_t>>> SplitSpace(
      HloInstruction* activations, ConvolutionDimensionNumbers& dim_numbers,
      int64_t& activations_batch_dim, int64_t high_padding, int64_t low_padding,
      int64_t spatial_split_size, int64_t num_splits,
      std::vector<int64_t>* spatial_dimensions_to_split,
      bool is_backprop = false, bool is_rhs = false);

  // Performs the actual dimension splitting.
  absl::StatusOr<HloInstruction*> PerformSplitSpace(
      HloInstruction* activations,
      absl::Span<const int64_t> spatial_dimensions_to_split,
      int64_t activations_batch_dim, int64_t spatial_split_size,
      int64_t num_splits);

  // Helper function that puts individually split dimensions together, and
  // merges the batch(es).
  // The input activations dimensions are ... B, B0, S0, B1, S1, ... Bn, Sn, ...
  // The output dimensions will be ..., B, S0, S1,.. Sn, ...
  absl::StatusOr<HloInstruction*> TransposeAndMergeBatch(
      HloInstruction* activations,
      absl::Span<const int64_t> final_split_spatial_dim_positioning,
      int64_t activations_batch_dim, int64_t old_batch_size);

  // Helper function for the SplitSpace function above. Handles padding and
  // reshaping to generate space-to-batched shape.
  absl::StatusOr<HloInstruction*> PadAndSplitSpace(
      HloInstruction* activations,
      absl::Span<const int64_t> spatial_dimensions_to_split,
      int64_t activations_batch_dim, int64_t high_padding, int64_t low_padding,
      int64_t spatial_split_size, int64_t num_splits);

  // Perform space-to-batch propagation on constants.
  absl::StatusOr<HloInstruction*> PropagateOnConstant(HloInstruction* consumer,
                                                      HloInstruction* producer);

  // Perform space-to-batch propagation on the convolution. Assumes the
  // activations were already space-to-batched.
  absl::Status PropagateOnConv(HloInstruction* convolution);

  // Perform space-to-batch propagation on concatenate.
  absl::Status PropagateOnConcat(HloInstruction* concat);

  // Perform space-to-batch propagation on reverse.
  absl::Status PropagateOnReverse(HloInstruction* reverse);

  // Perform space-to-batch propagation on pad.
  absl::Status PropagateOnPad(HloInstruction* pad);

  // Perform space-to-batch propagation on slice.
  absl::Status PropagateOnSlice(HloInstruction* slice);

  // Perform space-to-batch propagation on the backprop filter convolution.
  // Assumes the activations and kernel were already space-to-batched.
  absl::Status PropagateOnBackpropFilterConv(HloInstruction* convolution);

  // Method that checks validity of space-to-batch on a given convolution.
  bool IsConvSuitableForSpaceToBatch(HloInstruction* convolution);

  // Method that returns true if this is a backprop filter convolution.
  bool IsThisBackPropFilterConv(HloInstruction* convolution);

  // Once a convolution has been space-to-batch'ed, this function will
  // transitively propagate the space-to-batch-ness on rest of the graph.
  absl::Status PropagateOnUsers(HloInstruction* old_conv);

  // Generates masked output with valid data. This is useful when larger shapes
  // are generated due to space-to-batch.
  absl::StatusOr<HloInstruction*> SelectValidPortion(
      HloInstruction* new_instr, HloInstruction* old_instr,
      HloInstruction* select_val, int64_t new_batch_dim,
      absl::Span<const int64_t> new_space_dims, int64_t old_batch_dim,
      absl::Span<const int64_t> old_space_dims);

  struct SpaceNextToBatchDetails {
    HloInstruction* instr;
    std::vector<int64_t> transpose_dims;
  };

  // Performs tranposition so that space dimension follows the batch dimension.
  absl::StatusOr<SpaceNextToBatchDetails> BringSpaceNextToBatch(
      HloInstruction* activations, ConvolutionDimensionNumbers& dim_numbers,
      int64_t& activations_batch_dim,
      std::vector<int64_t>* spatial_dimensions_to_split,
      bool is_backprop = false, bool is_rhs = false);

  // Decreases the spatial dimension size in an already space-to-batched shape
  // so that the new size is new_spatial_dim_size.
  absl::StatusOr<HloInstruction*> ChangeSpatialSizeOnSpaceToBatchedShape(
      HloInstruction* activations, int64_t batch_dimension,
      int64_t old_batch_size,
      absl::Span<const int64_t> spatial_dimensions_to_split,
      int64_t new_spatial_dim_size, bool increase_spatial_size = false);

  // Turns B, S0, S1, ..., Sn into B, B0, S0, B1, S1,... Bn, Sn.
  absl::StatusOr<HloInstruction*> SplitAndTransposeMergedBatch(
      HloInstruction* activations, int64_t batch_dimension,
      int64_t old_batch_size, absl::Span<const int64_t> spatial_dimensions);

  // Function that converts spaced-to-batch shape back to the original.
  absl::StatusOr<HloInstruction*> BatchToSpace(HloInstruction* old_instr);

  // Duplicates elements at boundaries.
  absl::StatusOr<HloInstruction*> HaloDuplicateWithSlice(
      HloInstruction* activations,
      absl::Span<const int64_t> spatial_dimensions_to_split,
      int64_t activations_batch_dim, int64_t low_padding, int64_t halo_size,
      HloInstruction* pad_val = nullptr);

  // Runs the visitor on a computation.
  absl::StatusOr<bool> Run();

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() = default;

  explicit ConvolutionVisitor(SpaceToBatchController ctrl,
                              HloComputation* computation);

  int64_t GetFirstChosenSpatialDim(HloInstruction* convolution) {
    const int64_t dim_count = ctrl_.count_of_dimensions_to_convert;
    const int64_t end_point = convolution->convolution_dimension_numbers()
                                  .input_spatial_dimensions_size() -
                              ctrl_.dimension_from_end_to_convert;
    return end_point - dim_count + 1;
  }

  std::vector<int64_t> GetChosenSpatialDims(HloInstruction* convolution) {
    const int64_t dim_count = ctrl_.count_of_dimensions_to_convert;
    const int64_t first_dim = GetFirstChosenSpatialDim(convolution);
    std::vector<int64_t> dims(dim_count);
    for (int i = 0; i < dim_count; ++i) {
      dims[i] =
          convolution->convolution_dimension_numbers().input_spatial_dimensions(
              first_dim + i);
    }
    return dims;
  }

  int64_t DimLookUp(absl::Span<const int64_t> permute_dims, int64_t id) {
    return permute_dims[id];
  }

  int DimMapper(SpaceToBatchDimMap s) { return static_cast<int>(s); }

  int64_t ReverseDimLookUp(absl::Span<const int64_t> permute_dims, int64_t id) {
    return std::distance(permute_dims.begin(), absl::c_find(permute_dims, id));
  }

  HloInstruction* DoesConvolutionFeedReduceWindowOrSelectAndScatter(
      HloInstruction* instr, int64_t depth);

  // Returns true if instr feeds an unpropagatable op before it feeds 'depth'
  // number of convolutions.
  bool DoesConvolutionFeedUnpropagatableOp(
      HloInstruction* instr, int64_t depth = kUnpropagatableOpSearchDepth);

  // Checks that the space-to-batched shape has not rendered the new spatial
  // dimension to be smaller than the window's size.
  bool IsSpaceToBatchedSpaceSizeSuitable(HloInstruction* instr);

 private:
  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  absl::flat_hash_set<HloInstruction*> convs_to_visit_;
  std::vector<HloInstruction*> conv_visitor_list_;
  HloInstructionSet non_propagatable_instrs_;
  // Map from a given spaced-to-batch instruction to its batched-to-space
  // version.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> batch_to_space_map_;

  // Map from old (non space-to-batch) instructions to space-to-batch'ed
  // instructions.
  absl::flat_hash_map<HloInstruction*, HloInstruction*> old_to_new_instrs_;

  // Map from instruction to dimensions of the shape. This is with respect to
  // the old instruction.
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>> instr_to_dim_map_;

  // Map from space-to-batch'ed instruction to its permute dims.
  absl::flat_hash_map<HloInstruction*, std::vector<int64_t>>
      instr_to_dim_permute_map_;

  // Map maintaining previously space-to-batched broadcasts.
  absl::flat_hash_map<HloInstruction*, absl::flat_hash_set<HloInstruction*>>
      broadcast_map_;

  // Whether rewrite has occurred.
  bool changed_ = false;

  // Depth for searching reduce window
  static constexpr int64_t kReduceWindowSearchDepth = 10;

  // Depth for searching unpropagatable op.
  static constexpr int64_t kUnpropagatableOpSearchDepth = 3;

  // Penalty on size for base dilated convs
  static constexpr int64_t kMultiplierOnSpaceForBaseDilation = 3;

  // Cache for <instruction, depth> ==> unpropagatablilty decision.
  absl::flat_hash_map<std::pair<HloInstruction*, int64_t>, bool>
      unpropagatability_cache_;

  // Controller for various knobs.
  SpaceToBatchController ctrl_;
};

ConvolutionVisitor::ConvolutionVisitor(SpaceToBatchController ctrl,
                                       HloComputation* computation) {
  ctrl_ = ctrl;
  computation_ = computation;
  for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
    if (inst->opcode() != HloOpcode::kConvolution) {
      continue;
    }

    auto convolution = inst;
    // Perform legality checks.
    if (!IsConvSuitableForSpaceToBatch(convolution)) {
      VLOG(1) << "Conv not suitable for space-to-batch "
              << convolution->ToString();
      continue;
    }
    VLOG(1) << "Conv added to space-to-batch worklist "
            << convolution->ToString();
    convs_to_visit_.insert(convolution);
    conv_visitor_list_.push_back(convolution);
  }
}

std::pair<std::vector<int64_t>, std::vector<int64_t>>
ConvolutionVisitor::GetSpatialDimsToSplit(HloInstruction* old_operand) {
  auto new_operand = old_to_new_instrs_[old_operand];
  auto dim_map_val = instr_to_dim_map_[old_operand];
  auto permute_dims = instr_to_dim_permute_map_[new_operand];
  std::vector<int64_t> old_dims(ctrl_.count_of_dimensions_to_convert),
      new_dims(ctrl_.count_of_dimensions_to_convert);

  old_dims[0] = dim_map_val[DimMapper(SpaceToBatchDimMap::kSpace0)];
  new_dims[0] = DimLookUp(permute_dims, old_dims[0]);
  for (int i = 1; i < ctrl_.count_of_dimensions_to_convert; ++i) {
    old_dims[i] = old_dims[0] + i;
    new_dims[i] = new_dims[0] + i;
  }
  return std::make_pair(old_dims, new_dims);
}

bool ConvolutionVisitor::IsForwardWindowDilatedConv(
    HloInstruction* convolution, ConvolutionDimensionNumbers& dim_numbers) {
  const int64_t window_dilation_factor =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .window_dilation();

  if (window_dilation_factor == 1) {
    return false;
  }

  const int64_t output_spatial_dim = dim_numbers.output_spatial_dimensions(
      GetFirstChosenSpatialDim(convolution));
  const int64_t kernel_spatial_dim = dim_numbers.kernel_spatial_dimensions(
      GetFirstChosenSpatialDim(convolution));

  // If convolution's spatial dim size is larger than that of RHS, this is a
  // forward RHS dilated convolution.
  return convolution->operand(1)->shape().dimensions(kernel_spatial_dim) <
         convolution->shape().dimensions(output_spatial_dim);
}

bool ConvolutionVisitor::IsConvSuitableForSpaceToBatch(
    HloInstruction* convolution) {
  ConvolutionDimensionNumbers dim_numbers =
      convolution->convolution_dimension_numbers();

  // If there are no specified spatial dims, we return.
  if (GetFirstChosenSpatialDim(convolution) < 0) {
    return false;
  }

  // Batch in batch_group_count has different semantics (it isn't true batch).
  // Consider supporting this case in future if needed.
  if (convolution->batch_group_count() != 1) {
    return false;
  }

  if (convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .window_dilation() != 1) {
    if (!IsForwardWindowDilatedConv(convolution, dim_numbers)) {
      return false;
    }
  }

  const ConvDetails c = GetConvolutionDetails(convolution, dim_numbers);

  const int64_t low_pad = convolution->window()
                              .dimensions(GetFirstChosenSpatialDim(convolution))
                              .padding_low();

  // TODO(b/168316428): Support base dilations more generically.
  if (c.base_dilation_factor != 1) {
    if (!ctrl_.enable_propagations_on_base_dilations) {
      return false;
    }
    if (c.stride != 1) {
      return false;
    }
    // For low pad of 0, only support a pointwise kernel.
    if (low_pad == 0) {
      if (c.kernel_spatial_dim_size != 1) {
        return false;
      }
    } else if (low_pad != c.base_dilation_factor - 1 &&
               low_pad != c.base_dilation_factor) {
      // Only support dilations such that base dilation factor and low pad are
      // compatible with kernel_spatial_dim_size to be compatible with
      // HaloDuplicateWithSlice.
      return false;
    }
  }

  int64_t activations_batch_dim = dim_numbers.input_batch_dimension();

  const int64_t old_batch_size =
      convolution->operand(0)->shape().dimensions(activations_batch_dim);

  if (old_batch_size > ctrl_.limit_on_batch_size) {
    return false;
  }

  VLOG(1) << "spatial size " << c.spatial_size << " halo size " << c.halo_size;

  // If the ratio is not within the 2X range, we can't Halo Pad from the next
  // split.
  if (c.halo_size > CeilOfRatio(c.spatial_size, ctrl_.number_of_splits)) {
    return false;
  }

  // TODO(b/201444224): The following cost model is needed to escape slowing
  // down ssd batch 4.
  if (c.base_dilation_factor > 1 &&
      c.inherent_low_padding == c.base_dilation_factor) {
    if (c.spatial_size <
        kMultiplierOnSpaceForBaseDilation * ctrl_.number_of_splits) {
      return false;
    }
  }

  VLOG(1) << "Legal space-to-batch convolution " << convolution->ToString();
  return true;
}

bool ConvolutionVisitor::IsThisBackPropFilterConv(HloInstruction* convolution) {
  auto activations = convolution->mutable_operand(0);
  auto kernel = convolution->mutable_operand(1);
  auto dim_numbers = convolution->convolution_dimension_numbers();

  if (!old_to_new_instrs_.contains(kernel) &&
      !old_to_new_instrs_.contains(activations)) {
    return false;
  }

  if (old_to_new_instrs_.contains(kernel)) {
    auto dim_map_val_op_0 = instr_to_dim_map_[kernel];
    const int64_t old_batch_dim =
        dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)];
    if (convolution->convolution_dimension_numbers()
            .kernel_input_feature_dimension() != old_batch_dim) {
      return false;
    }
  }

  if (old_to_new_instrs_.contains(activations)) {
    auto dim_map_val_op_0 = instr_to_dim_map_[activations];
    const int64_t old_batch_dim =
        dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)];
    if (dim_numbers.input_feature_dimension() != old_batch_dim) {
      return false;
    }
  }

  return true;
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::HaloDuplicateWithSlice(
    HloInstruction* activations,
    absl::Span<const int64_t> spatial_dimensions_to_split,
    int64_t activations_batch_dim, int64_t low_padding, int64_t halo_size,
    HloInstruction* pad_val) {
  const int64_t spatial_dim_count = spatial_dimensions_to_split.size();
  const int64_t additional_batch_size =
      IPow<int64_t>(ctrl_.number_of_splits, spatial_dim_count);
  const int64_t original_batch_size =
      activations->shape().dimensions(activations_batch_dim) /
      additional_batch_size;

  const int64_t spatial_split_size =
      activations->shape().dimensions(spatial_dimensions_to_split[0]);
  const int64_t batch_size = ctrl_.number_of_splits;

  TF_ASSIGN_OR_RETURN(
      activations, SplitAndTransposeMergedBatch(
                       activations, activations_batch_dim, original_batch_size,
                       spatial_dimensions_to_split));

  const int64_t rank = activations->shape().rank();

  VLOG(1) << "In HaloDuplicateWithSlice with activations "
          << activations->ToString() << " batch_size " << batch_size
          << " spatial_split_size " << spatial_split_size << " low_padding "
          << low_padding << " halo size " << halo_size;

  CHECK_LE(std::abs(halo_size - low_padding), spatial_split_size);

  for (int64_t i = 0; i < spatial_dimensions_to_split.size(); ++i) {
    int64_t spatial_dimension_to_split = activations_batch_dim + 2 * (i + 1);
    int64_t remapped_batch_dimension = spatial_dimension_to_split - 1;
    HloInstruction* first_slice = nullptr;

    std::vector<int64_t> strides(rank, 1);
    HloInstruction* padding =
        pad_val == nullptr
            ? activations->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(activations->shape().element_type())))
            : pad_val;

    if (low_padding > 0) {
      std::vector<int64_t> start_indices(rank, 0),
          end_indices(activations->shape().dimensions().begin(),
                      activations->shape().dimensions().end());
      start_indices[spatial_dimension_to_split] =
          spatial_split_size - low_padding;
      end_indices[remapped_batch_dimension] = batch_size - 1;
      end_indices[spatial_dimension_to_split] = spatial_split_size;

      TF_ASSIGN_OR_RETURN(first_slice,
                          MakeSliceHlo(activations, start_indices, end_indices,
                                       strides, &activations->metadata(),
                                       &activations->frontend_attributes()));
      VLOG(1) << "first slice " << first_slice->ToString();

      PaddingConfig padding_config =
          MakeNoPaddingConfig(first_slice->shape().dimensions_size());
      padding_config.mutable_dimensions(remapped_batch_dimension)
          ->set_edge_padding_low(1);

      TF_ASSIGN_OR_RETURN(first_slice,
                          MakePadHlo(first_slice, padding, padding_config,
                                     &first_slice->metadata(),
                                     &first_slice->frontend_attributes()));
    }

    HloInstruction* halo_region = nullptr;
    if (halo_size - low_padding > 0) {
      std::vector<int64_t> start_indices_halo(rank, 0),
          end_indices_halo(activations->shape().dimensions().begin(),
                           activations->shape().dimensions().end());

      start_indices_halo[remapped_batch_dimension] = 1;
      end_indices_halo[spatial_dimension_to_split] = halo_size - low_padding;
      TF_ASSIGN_OR_RETURN(
          halo_region,
          MakeSliceHlo(activations, start_indices_halo, end_indices_halo,
                       strides, &activations->metadata(),
                       &activations->frontend_attributes()));
      VLOG(1) << "halo_region " << halo_region->ToString();
      PaddingConfig padding_config_halo =
          MakeNoPaddingConfig(halo_region->shape().dimensions_size());
      padding_config_halo.mutable_dimensions(remapped_batch_dimension)
          ->set_edge_padding_high(1);
      TF_ASSIGN_OR_RETURN(halo_region,
                          MakePadHlo(halo_region, padding, padding_config_halo,
                                     &halo_region->metadata(),
                                     &halo_region->frontend_attributes()));
    }

    if ((halo_size == 0 && low_padding != 0) || low_padding < 0) {
      std::vector<int64_t> start_indices_activations_cut(rank, 0),
          end_indices_activations_cut(activations->shape().dimensions().begin(),
                                      activations->shape().dimensions().end());
      // When no halo is needed, we must slice out activations.
      if (low_padding > 0) {
        end_indices_activations_cut[spatial_dimension_to_split] =
            spatial_split_size - low_padding;
      } else {
        start_indices_activations_cut[spatial_dimension_to_split] =
            0 - low_padding;
        end_indices_activations_cut[spatial_dimension_to_split] =
            spatial_split_size;
      }
      TF_ASSIGN_OR_RETURN(
          activations, MakeSliceHlo(activations, start_indices_activations_cut,
                                    end_indices_activations_cut, strides,
                                    &activations->metadata(),
                                    &activations->frontend_attributes()));
    }

    if (first_slice != nullptr) {
      TF_ASSIGN_OR_RETURN(
          activations,
          MakeConcatHlo({first_slice, activations}, spatial_dimension_to_split,
                        &activations->metadata(),
                        &activations->frontend_attributes()));
    }

    if (halo_region != nullptr) {
      TF_ASSIGN_OR_RETURN(
          activations,
          MakeConcatHlo({activations, halo_region}, spatial_dimension_to_split,
                        &activations->metadata(),
                        &activations->frontend_attributes()));
    }
  }

  TF_ASSIGN_OR_RETURN(
      activations,
      TransposeAndMergeBatch(
          activations,
          /*final_split_spatial_dim_positioning=*/spatial_dimensions_to_split,
          activations_batch_dim, original_batch_size));

  VLOG(1) << "HaloDuplicated activations " << activations->ToString();
  return activations;
}

absl::StatusOr<ConvolutionVisitor::SpaceNextToBatchDetails>
ConvolutionVisitor::BringSpaceNextToBatch(
    HloInstruction* activations, ConvolutionDimensionNumbers& dim_numbers,
    int64_t& activations_batch_dim,
    std::vector<int64_t>* spatial_dimensions_to_split, bool is_backprop,
    bool is_rhs) {
  for (int64_t i = 1; i < spatial_dimensions_to_split->size(); ++i) {
    CHECK_EQ(spatial_dimensions_to_split->at(i),
             spatial_dimensions_to_split->at(i - 1) + 1)
        << "Spatial dimensions are not contiguous";
  }

  int64_t spatial_dimension_to_split = spatial_dimensions_to_split->at(0);

  std::vector<int64_t> transpose_dims(activations->shape().rank());
  if (spatial_dimension_to_split == activations_batch_dim + 1) {
    absl::c_iota(transpose_dims, 0);
  } else {
    ConvolutionDimensionNumbers new_dim_numbers = dim_numbers;
    int64_t pushed_counter = 0;
    int64_t new_batch_dim, new_spatial_dim;
    int64_t dim_counter = 0;
    if (is_rhs) {
      CHECK(is_backprop);
      for (int i = 0; i < activations->shape().rank(); ++i) {
        if (i == activations_batch_dim) {
          continue;
        }
        if (i == spatial_dimension_to_split) {
          transpose_dims[dim_counter++] = activations_batch_dim;
          new_batch_dim = pushed_counter;
          pushed_counter++;
          new_spatial_dim = pushed_counter;
        }

        if (i == dim_numbers.kernel_output_feature_dimension()) {
          new_dim_numbers.set_kernel_output_feature_dimension(pushed_counter);
        } else {
          auto it = absl::c_find(dim_numbers.kernel_spatial_dimensions(), i);
          if (it != dim_numbers.kernel_spatial_dimensions().end()) {
            int64_t j = it - dim_numbers.kernel_spatial_dimensions().begin();
            new_dim_numbers.set_kernel_spatial_dimensions(j, pushed_counter);
          }
        }
        transpose_dims[dim_counter++] = i;
        pushed_counter++;
      }

      activations_batch_dim = new_batch_dim;
      spatial_dimension_to_split = new_spatial_dim;
      TF_ASSIGN_OR_RETURN(activations,
                          MakeTransposeHlo(activations, transpose_dims));

      new_dim_numbers.set_kernel_input_feature_dimension(activations_batch_dim);

    } else {
      for (int i = 0; i < activations->shape().rank(); ++i) {
        if (i == activations_batch_dim) {
          continue;
        }
        if (i == spatial_dimension_to_split) {
          transpose_dims[dim_counter++] = activations_batch_dim;
          new_batch_dim = pushed_counter;
          pushed_counter++;
          new_spatial_dim = pushed_counter;
        }

        if (is_backprop && i == dim_numbers.input_batch_dimension()) {
          new_dim_numbers.set_input_batch_dimension(pushed_counter);
        } else if (i == dim_numbers.input_feature_dimension()) {
          new_dim_numbers.set_input_feature_dimension(pushed_counter);
        } else {
          auto it = absl::c_find(dim_numbers.input_spatial_dimensions(), i);
          if (it != dim_numbers.input_spatial_dimensions().end()) {
            int64_t j = it - dim_numbers.input_spatial_dimensions().begin();
            new_dim_numbers.set_input_spatial_dimensions(j, pushed_counter);
          }
        }
        transpose_dims[dim_counter++] = i;
        pushed_counter++;
      }

      activations_batch_dim = new_batch_dim;
      spatial_dimension_to_split = new_spatial_dim;
      TF_ASSIGN_OR_RETURN(activations,
                          MakeTransposeHlo(activations, transpose_dims));

      if (is_backprop) {
        new_dim_numbers.set_input_feature_dimension(activations_batch_dim);
      } else {
        new_dim_numbers.set_input_batch_dimension(activations_batch_dim);
      }
    }

    dim_numbers = new_dim_numbers;
  }

  // Note that the spatial dimensions are in a sequential increasing order.
  for (int64_t i = 0; i < spatial_dimensions_to_split->size(); ++i) {
    (*spatial_dimensions_to_split)[i] = spatial_dimension_to_split + i;
  }

  return SpaceNextToBatchDetails{activations, transpose_dims};
}

absl::StatusOr<HloInstruction*>
ConvolutionVisitor::SplitAndTransposeMergedBatch(
    HloInstruction* activations, int64_t batch_dimension,
    int64_t old_batch_size, absl::Span<const int64_t> spatial_dimensions) {
  CHECK_EQ(batch_dimension + 1, spatial_dimensions[0]);
  std::vector<int64_t> new_dimensions(activations->shape().dimensions().begin(),
                                      activations->shape().dimensions().end());

  const int64_t new_batch_size =
      activations->shape().dimensions(batch_dimension);

  VLOG(3) << "Decreasing the spatial size while propagating new_batch_size "
          << new_batch_size << " old_batch_size " << old_batch_size;

  new_dimensions[batch_dimension] = old_batch_size;

  const int64_t spatial_dim_count = spatial_dimensions.size();
  // Create additional batch dimensions.
  for (int64_t i = 0; i < spatial_dim_count; ++i) {
    new_dimensions.insert(new_dimensions.begin() + spatial_dimensions[0],
                          ctrl_.number_of_splits);
  }

  // Reshape the output of the new conv into the old convolutions shape.
  TF_ASSIGN_OR_RETURN(HloInstruction * batch_split_activations,
                      MakeReshapeHlo(new_dimensions, activations));

  if (spatial_dim_count > 1) {
    // Transpose such that we get // B, B0, S0, B1, S1,...
    std::vector<int64_t> transpose_dims(new_dimensions.size());
    absl::c_iota(transpose_dims, 0);
    // Transpose such that we get B, B0, S0, B1, S1,...
    std::vector<int64_t> trans_dims(new_dimensions.size());
    absl::c_iota(trans_dims, 0);

    int64_t start_batch_dim_position = batch_dimension + 1;
    int64_t start_space_dim_position = batch_dimension + 2;

    for (int i = 0; i < spatial_dim_count; ++i) {
      transpose_dims[start_batch_dim_position + 2 * i] =
          batch_dimension + spatial_dim_count - i;
      transpose_dims[start_space_dim_position + 2 * i] =
          batch_dimension + spatial_dim_count + 1 + i;
    }

    TF_ASSIGN_OR_RETURN(
        batch_split_activations,
        MakeTransposeHlo(batch_split_activations, transpose_dims));
  }
  return batch_split_activations;
}

absl::StatusOr<HloInstruction*>
ConvolutionVisitor::ChangeSpatialSizeOnSpaceToBatchedShape(
    HloInstruction* activations, int64_t batch_dimension,
    int64_t old_batch_size, absl::Span<const int64_t> spatial_dimensions,
    int64_t new_spatial_dim_size, bool increase_spatial_size) {
  CHECK_EQ(batch_dimension + 1, spatial_dimensions[0]);
  std::vector<int64_t> new_dimensions(activations->shape().dimensions().begin(),
                                      activations->shape().dimensions().end());

  const int64_t spatial_dim_count = spatial_dimensions.size();
  const int64_t spatial_dim_size =
      activations->shape().dimensions(spatial_dimensions[0]);
  const int64_t reshaped_space_size = spatial_dim_size * ctrl_.number_of_splits;

  // Reshape the output of the new conv into the old convolutions shape.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * batch_split_activations,
      SplitAndTransposeMergedBatch(activations, batch_dimension, old_batch_size,
                                   spatial_dimensions));

  // Now merge the individual (split) batch and space dimensions.
  std::vector<int64_t> batch_space_collapse_reshape_dims(
      batch_split_activations->shape().dimensions().begin(),
      batch_split_activations->shape().dimensions().end());

  batch_space_collapse_reshape_dims.erase(
      batch_space_collapse_reshape_dims.begin() + spatial_dimensions[0],
      batch_space_collapse_reshape_dims.begin() + spatial_dimensions[0] +
          spatial_dim_count);

  for (auto spatial_dimension : spatial_dimensions) {
    batch_space_collapse_reshape_dims[spatial_dimension] = reshaped_space_size;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * batch_space_collapsed_reshape,
                      MakeReshapeHlo(batch_space_collapse_reshape_dims,
                                     batch_split_activations));

  VLOG(3) << "First reshape done";

  const int64_t rank = activations->shape().rank();

  // If spatial size is increased, we add padding. If it has shrunk, we slice
  // out the padding that was added before.
  if (increase_spatial_size) {
    PaddingConfig padding_config = MakeNoPaddingConfig(
        batch_space_collapsed_reshape->shape().dimensions_size());
    for (auto spatial_dimension : spatial_dimensions) {
      padding_config.mutable_dimensions(spatial_dimension)
          ->set_edge_padding_high(new_spatial_dim_size *
                                      ctrl_.number_of_splits -
                                  reshaped_space_size);
      padding_config.mutable_dimensions(spatial_dimension)
          ->set_edge_padding_low(0);
    }
    HloInstruction* padding = activations->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(
            batch_space_collapsed_reshape->shape().element_type())));

    TF_ASSIGN_OR_RETURN(
        batch_space_collapsed_reshape,
        MakePadHlo(batch_space_collapsed_reshape, padding, padding_config,
                   &batch_space_collapsed_reshape->metadata(),
                   &batch_space_collapsed_reshape->frontend_attributes()));
  } else {
    std::vector<int64_t> start_indices(rank, 0),
        end_indices(batch_space_collapsed_reshape->shape().dimensions().begin(),
                    batch_space_collapsed_reshape->shape().dimensions().end()),
        strides(rank, 1);
    for (auto spatial_dimension : spatial_dimensions) {
      end_indices[spatial_dimension] =
          new_spatial_dim_size * ctrl_.number_of_splits;
    }
    // This is the slice from halo padding.
    TF_ASSIGN_OR_RETURN(
        batch_space_collapsed_reshape,
        MakeSliceHlo(batch_space_collapsed_reshape, start_indices, end_indices,
                     strides, &batch_space_collapsed_reshape->metadata(),
                     &batch_space_collapsed_reshape->frontend_attributes()));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * activations_new,
      PerformSplitSpace(batch_space_collapsed_reshape, spatial_dimensions,
                        batch_dimension, new_spatial_dim_size,
                        ctrl_.number_of_splits));

  VLOG(3) << "Size decreased activations " << activations_new->ToString();

  return activations_new;
}

absl::StatusOr<bool> ConvolutionVisitor::Run() {
  for (auto conv : conv_visitor_list_) {
    // If we expect to see an unpropagatable op, space-to-batch may not be
    // beneficial.
    if (ctrl_.disable_starting_on_small_chains &&
        DoesConvolutionFeedUnpropagatableOp(conv)) {
      VLOG(1) << "Giving up on conv " << conv->ToString()
              << " because it feeds an unpropagatable op";
      convs_to_visit_.erase(conv);
    }
    if (convs_to_visit_.count(conv) > 0) {
      TF_CHECK_OK(PerformSpaceToBatchOnConvolution(conv));
      changed_ = true;
    }
  }
  conv_visitor_list_.clear();
  convs_to_visit_.clear();
  // Iterate through all instructions that we could not propagate through, and
  // turn their operands from batch-to-space as needed.
  for (auto instr : non_propagatable_instrs_) {
    if (instr->opcode() == HloOpcode::kConvolution) {
      VLOG(1) << "Instr " << instr->ToString();
    }
    // Try to propagate on backprop filters
    if (instr->opcode() == HloOpcode::kConvolution &&
        !IsConvSuitableForSpaceToBatch(instr)) {
      HloInstruction* producer = nullptr;
      if (old_to_new_instrs_.contains(instr->mutable_operand(0))) {
        producer = instr->mutable_operand(0);
      } else if (old_to_new_instrs_.contains(instr->mutable_operand(1))) {
        producer = instr->mutable_operand(1);
      }
      if (producer) {
        if (CanPropagate(instr, producer)) {
          bool needs_further_propagation;
          TF_ASSIGN_OR_RETURN(needs_further_propagation,
                              Propagate(instr, producer));
          TF_CHECK_OK(computation_->ReplaceInstruction(
              instr, old_to_new_instrs_[instr]));
          continue;
        }
      }
    }
    VLOG(1) << "Could not eventually propagate through " << instr->ToString();
    absl::flat_hash_map<int64_t, HloInstruction*> operand_map;
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      if (old_to_new_instrs_.count(instr->mutable_operand(i))) {
        TF_ASSIGN_OR_RETURN(operand_map[i],
                            BatchToSpace(instr->mutable_operand(i)));
      }
    }
    for (auto entry : operand_map) {
      TF_CHECK_OK(instr->ReplaceOperandWith(entry.first, entry.second));
    }
  }
  non_propagatable_instrs_.clear();
  return changed_;
}

bool IsTrivialElementwise(HloInstruction* hlo) {
  if (hlo->opcode() == HloOpcode::kFusion || hlo->opcode() == HloOpcode::kRng ||
      hlo->opcode() == HloOpcode::kCopy ||
      hlo->opcode() == HloOpcode::kConstant ||
      hlo->opcode() == HloOpcode::kIota || hlo->opcode() == HloOpcode::kMap) {
    return false;
  }
  return hlo->IsElementwise();
}

bool ConvolutionVisitor::CanPropagate(HloInstruction* consumer,
                                      HloInstruction* producer) {
  if (IsTrivialElementwise(consumer)) {
    VLOG(2) << "Doing propagation check on elementwise op: "
            << consumer->ToString();

    HloInstruction* pivot_operand = nullptr;
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      auto old_producer = consumer->mutable_operand(i);
      std::vector<HloInstruction*> to_transform;
      const bool broadcast_or_constant =
          (old_producer->opcode() == HloOpcode::kConstant) ||
          (old_producer->opcode() == HloOpcode::kBroadcast &&
           IsBroadcastPropagatable(old_producer, producer)) ||
          (consumer->IsElementwiseBinary() &&
           old_producer->opcode() == HloOpcode::kBroadcast &&
           IsBroadcastTree(old_producer, producer, to_transform));

      if (!old_to_new_instrs_.contains(old_producer) &&
          !broadcast_or_constant) {
        VLOG(1) << "Cannot propagate on elementwise op " << consumer->ToString()
                << " because operand " << old_producer->ToString()
                << " isn't ready ";
        return false;
      } else {
        if (broadcast_or_constant) {
          VLOG(2) << "Skipping on " << old_producer->ToString();
          continue;
        }

        CHECK(old_to_new_instrs_.contains(old_producer));

        CHECK(instr_to_dim_map_.contains(old_producer));
        if (pivot_operand == nullptr) {
          pivot_operand = old_producer;
          VLOG(2) << "Elementwise op: pivot " << old_producer->ToString();
        } else {
          if (instr_to_dim_map_[pivot_operand]
                               [DimMapper(SpaceToBatchDimMap::kBatch)] !=
                  instr_to_dim_map_[old_producer]
                                   [DimMapper(SpaceToBatchDimMap::kBatch)] ||
              instr_to_dim_map_[pivot_operand]
                               [DimMapper(SpaceToBatchDimMap::kSpace0)] !=
                  instr_to_dim_map_[old_producer]
                                   [DimMapper(SpaceToBatchDimMap::kSpace0)]) {
            VLOG(2) << "Elementwise op: checking for shape equivalence "
                    << consumer->ToString()
                    << " failed due to changed batch space ordering ";
            return false;
          }
          auto pivot_new_instr = old_to_new_instrs_[pivot_operand];
          auto pivot_permute_dims = instr_to_dim_permute_map_[pivot_new_instr];
          auto new_instr = old_to_new_instrs_[old_producer];
          auto permute_dims = instr_to_dim_permute_map_[new_instr];
          for (int j = 0; j < pivot_permute_dims.size(); ++j) {
            // Ensure the dimension mapping is the same.
            if (pivot_permute_dims[j] != permute_dims[j]) {
              VLOG(2) << "Elementwise op: checking for shape equivalence "
                      << consumer->ToString()
                      << " failed due to permuted dimensions ";
              return false;
            }

            // Make sure all other dimensions are of the same size.
            if (pivot_new_instr->shape().dimensions(j) !=
                new_instr->shape().dimensions(j)) {
              if (!((consumer->IsElementwiseBinary() ||
                     consumer->opcode() == HloOpcode::kSelect) &&
                    j == instr_to_dim_map_[pivot_operand][DimMapper(
                             SpaceToBatchDimMap::kSpace0)])) {
                VLOG(2) << "Elementwise op: checking for shape equivalence "
                        << consumer->ToString()
                        << " failed due to changed shape sizes ";
                return false;
              }
            }
          }
        }
      }
    }
  }

  if (consumer->opcode() == HloOpcode::kConcatenate) {
    // Make sure all operands have been space-to-batched.
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      if (!instr_to_dim_map_.contains(consumer->mutable_operand(i))) {
        return false;
      }
    }
    auto pivot_operand = consumer->mutable_operand(0);
    auto pivot_new_instr = old_to_new_instrs_[pivot_operand];
    auto pivot_permute_dims = instr_to_dim_permute_map_[pivot_new_instr];
    for (int64_t i = 1; i < consumer->operand_count(); ++i) {
      auto new_instr = old_to_new_instrs_[consumer->mutable_operand(i)];
      auto permute_dims = instr_to_dim_permute_map_[new_instr];

      for (int j = 0; j < pivot_permute_dims.size(); ++j) {
        // Ensure the dimension mapping is the same.
        if (pivot_permute_dims[j] != permute_dims[j]) {
          VLOG(2) << "Concat op: checking for shape equivalence "
                  << consumer->ToString()
                  << " failed due to permuted dimensions ";
          return false;
        }
        // Make sure all other dimensions are of the same size.
        if (pivot_new_instr->shape().dimensions(j) !=
            new_instr->shape().dimensions(j)) {
          VLOG(2) << "Concat op: checking for shape equivalence "
                  << consumer->ToString()
                  << " failed due to changed shape sizes ";
          return false;
        }
      }
    }
    return true;
  }

  if (consumer->opcode() == HloOpcode::kConvolution) {
    if (!ConsumeFuel("space-to-batch-converter", [&] {
          return "Skipping space-to-batch propagation because fuel over\n";
        })) {
      return false;
    }
    // Lambda that checks basic sanity of dimension propagation on convolutions.
    // This includes: the split dimension from the previous convolution should
    // remain the same. No feature/batch dimension should be turned into a
    // spatial dimension.
    auto are_conv_dims_compatible =
        [&](const ConvolutionDimensionNumbers dim_numbers,
            std::vector<int64_t>& dim_map, bool check_lhs) {
          if (check_lhs) {
            if (dim_numbers.input_spatial_dimensions(
                    GetFirstChosenSpatialDim(consumer)) !=
                dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)]) {
              return false;
            }
            for (int i = 0; i < dim_numbers.input_spatial_dimensions().size();
                 ++i) {
              if (dim_numbers.input_spatial_dimensions(i) ==
                      dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] ||
                  dim_numbers.input_spatial_dimensions(i) ==
                      dim_map[DimMapper(SpaceToBatchDimMap::kFeature)]) {
                return false;
              }
            }
          } else {
            if (dim_numbers.kernel_spatial_dimensions(
                    GetFirstChosenSpatialDim(consumer)) !=
                dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)]) {
              return false;
            }
            for (int i = 0; i < dim_numbers.kernel_spatial_dimensions().size();
                 ++i) {
              if (dim_numbers.kernel_spatial_dimensions(i) ==
                      dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] ||
                  dim_numbers.kernel_spatial_dimensions(i) ==
                      dim_map[DimMapper(SpaceToBatchDimMap::kFeature)]) {
                return false;
              }
            }
          }
          return true;
        };

    VLOG(1) << "Checking if conv is supported for propagation "
            << consumer->ToString();
    bool found_good_non_window_dilated_conv = true;
    if (IsConvSuitableForSpaceToBatch(consumer)) {
      // Activations must have been space-to-batched to enable propagation.
      if (!old_to_new_instrs_.contains(consumer->mutable_operand(0))) {
        found_good_non_window_dilated_conv = false;
      }
      ConvolutionDimensionNumbers dim_numbers =
          consumer->convolution_dimension_numbers();

      ConvDetails c = GetConvolutionDetails(consumer, dim_numbers);

      auto retval = GetSpatialDimsToSplit(consumer->mutable_operand(0));
      std::vector<int64_t> new_spatial_dims = retval.second;

      auto new_activations = old_to_new_instrs_[consumer->mutable_operand(0)];
      // If low padding is large, there's no benefit in propagating. This
      // also makes halo creation unnecessarily difficult (b/246862180).
      if (new_activations->shape().dimensions(retval.second[0]) <
          c.inherent_low_padding) {
        return false;
      }

      auto dim_map_val_op_0 = instr_to_dim_map_[consumer->mutable_operand(0)];

      if (!are_conv_dims_compatible(consumer->convolution_dimension_numbers(),
                                    dim_map_val_op_0, /*check_lhs*/ true)) {
        found_good_non_window_dilated_conv = false;
      }
      // Make sure that the batch dimension is the same across the producer
      // and consumer.
      if (consumer->convolution_dimension_numbers().input_batch_dimension() !=
          dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)]) {
        found_good_non_window_dilated_conv = false;
      }

      if (found_good_non_window_dilated_conv) {
        return true;
      }
    }

    if (!ctrl_.enable_propagations_on_window_dilations) {
      return false;
    }

    if (!IsThisBackPropFilterConv(consumer)) {
      return false;
    }
    // Check for space-to-depth readiness here. Note this is not done in
    // SupportedOpForPropagation because the readiness is dependent upon
    // space-to-batchedness of the operands.

    // If there are no specified spatial dims, we return.
    if (GetFirstChosenSpatialDim(consumer) < 0) {
      return false;
    }

    // We currently only support stride of 1.
    if (consumer->window()
            .dimensions(GetFirstChosenSpatialDim(consumer))
            .stride() != 1) {
      return false;
    }

    // Same reason why we give up on batch group counts applies to features in
    // backprop.
    if (consumer->feature_group_count() != 1) {
      return false;
    }

    VLOG(2) << "Checking for backprop filter conv propagatability";
    CHECK_EQ(consumer->operand_count(), 2);

    auto activations = consumer->mutable_operand(0);
    auto kernel = consumer->mutable_operand(1);

    auto win_dims =
        consumer->window().dimensions(GetFirstChosenSpatialDim(consumer));
    const int64_t rhs_dilation = win_dims.window_dilation();
    const int64_t lhs_dilation = win_dims.base_dilation();

    // LHS dilations are supported by PropagateOnConv, and not by
    // PropagateOnBackpropFilterConv.
    if (lhs_dilation != 1) {
      return false;
    }
    // If the rhs_dilation is absent, we want both LHS and RHS to be space-to-
    // batched for propagating on backprop convolutions.

    if (rhs_dilation == 1 &&
        !ctrl_.enable_propagations_on_trivial_window_dilations) {
      if (!old_to_new_instrs_.contains(kernel) ||
          !old_to_new_instrs_.contains(activations)) {
        return false;
      }
    }

    if (!old_to_new_instrs_.contains(kernel)) {
      const int64_t rhs_batch =
          kernel->shape().dimensions(consumer->convolution_dimension_numbers()
                                         .kernel_input_feature_dimension());
      auto dim_map_val_op_0 = instr_to_dim_map_[activations];
      const int64_t old_batch_dim =
          dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)];
      const int64_t old_space_dim =
          dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kSpace0)];
      auto first_operand = old_to_new_instrs_[activations];
      auto permute_dims_first_operand =
          instr_to_dim_permute_map_[first_operand];
      const int64_t new_batch_dim =
          DimLookUp(permute_dims_first_operand, old_batch_dim);
      const int64_t new_space_dim =
          DimLookUp(permute_dims_first_operand, old_space_dim);
      const int64_t lhs_batch =
          first_operand->shape().dimensions(new_batch_dim);

      if (first_operand->shape().dimensions(new_space_dim) % rhs_dilation !=
          0) {
        return false;
      }
      // Because we want to convert activations into a space-to-batched version
      // only for backprop filter convolutions, we want to make sure that the
      // batch dimensions (feature dimensions, technically) are same sized.
      // Since LHS is already space-to-batched, we need to account for it too.
      if (rhs_batch * ctrl_.number_of_splits != lhs_batch) {
        return false;
      }

      if (!are_conv_dims_compatible(consumer->convolution_dimension_numbers(),
                                    dim_map_val_op_0, /*check_lhs*/ true)) {
        return false;
      }

      // If kernel have not been propagated through, we can do
      // space-to-batch on them provided kernel has been propagated.
      VLOG(2)
          << "Backprop filter conv ready for propagation: activations ready, "
             " kernel will be space-to-batched";
      return true;
    }

    if (!old_to_new_instrs_.contains(activations)) {
      const int64_t lhs_batch = activations->shape().dimensions(
          consumer->convolution_dimension_numbers().input_feature_dimension());
      auto dim_map_val_op_1 = instr_to_dim_map_[consumer->mutable_operand(1)];
      const int64_t old_batch_dim =
          dim_map_val_op_1[DimMapper(SpaceToBatchDimMap::kBatch)];
      auto second_operand = old_to_new_instrs_[kernel];
      auto permute_dims_second_operand =
          instr_to_dim_permute_map_[second_operand];
      const int64_t new_batch_dim =
          DimLookUp(permute_dims_second_operand, old_batch_dim);
      const int64_t rhs_batch =
          second_operand->shape().dimensions(new_batch_dim);

      // Because we want to convert activations into a space-to-batched version
      // only for backprop filter convolutions, we want to make sure that the
      // batch dimensions (feature dimensions, technically) are same sized.
      // Since RHS is already space-to-batched, we need to account for it too.
      if (rhs_batch != ctrl_.number_of_splits * lhs_batch) {
        return false;
      }

      if (!are_conv_dims_compatible(consumer->convolution_dimension_numbers(),
                                    dim_map_val_op_1, /*check_lhs*/ false)) {
        return false;
      }

      // If activations have not been propagated through, we can do
      // space-to-batch on them provided kernel has been propagated.
      VLOG(2) << "Backprop filter conv ready for propagation: kernel ready, "
                 " activations will be space-to-batched";
      return true;
    }

    auto first_operand = old_to_new_instrs_[activations];
    auto dim_map_val_op_0 = instr_to_dim_map_[activations];

    auto second_operand = old_to_new_instrs_[kernel];
    auto dim_map_val_op_1 = instr_to_dim_map_[kernel];

    auto permute_dims_first_operand = instr_to_dim_permute_map_[first_operand];
    auto permute_dims_second_operand =
        instr_to_dim_permute_map_[second_operand];

    const int64_t new_batch_dim_operand_0 =
        DimLookUp(permute_dims_first_operand,
                  dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)]);

    const int64_t new_space_dim_operand_0 =
        DimLookUp(permute_dims_first_operand,
                  dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kSpace0)]);

    const int64_t new_batch_dim_operand_1 =
        DimLookUp(permute_dims_second_operand,
                  dim_map_val_op_1[DimMapper(SpaceToBatchDimMap::kBatch)]);
    const int64_t new_space_dim_operand_1 =
        DimLookUp(permute_dims_second_operand,
                  dim_map_val_op_1[DimMapper(SpaceToBatchDimMap::kSpace0)]);

    if (first_operand->shape().dimensions(new_batch_dim_operand_0) !=
        second_operand->shape().dimensions(new_batch_dim_operand_1)) {
      VLOG(2) << "Backprop filter conv not ready for propagation because batch "
                 "dimensions don't line up";
      return false;
    }

    if (first_operand->shape().dimensions(new_space_dim_operand_0) >
        rhs_dilation *
            second_operand->shape().dimensions(new_space_dim_operand_1)) {
      VLOG(2) << "Backprop filter conv not ready for propagation because of "
                 "dilation factor mismatch";
      return false;
    }

    if (!are_conv_dims_compatible(consumer->convolution_dimension_numbers(),
                                  dim_map_val_op_0, /*check_lhs*/ true)) {
      return false;
    }

    if (!are_conv_dims_compatible(consumer->convolution_dimension_numbers(),
                                  dim_map_val_op_1, /*check_lhs*/ false)) {
      return false;
    }

    VLOG(2) << "Backprop filter conv ready for propagation";

    return true;
  }

  if (consumer->opcode() == HloOpcode::kReduceWindow ||
      consumer->opcode() == HloOpcode::kReduce) {
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      auto old_producer = consumer->mutable_operand(i);
      if (i == 0 && !old_to_new_instrs_.contains(old_producer)) {
        return false;
      }
    }

    // Make sure the post space-to-batch dim size is larger than window size.
    if (consumer->opcode() == HloOpcode::kReduceWindow) {
      return IsSpaceToBatchedSpaceSizeSuitable(consumer);
    }
  }

  if (consumer->opcode() == HloOpcode::kSelectAndScatter) {
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      auto old_producer = consumer->mutable_operand(i);
      if (i < 2 && !old_to_new_instrs_.contains(old_producer)) {
        return false;
      }
    }

    auto first_operand = old_to_new_instrs_[consumer->mutable_operand(0)];
    auto dim_map_val_op_0 = instr_to_dim_map_[consumer->mutable_operand(0)];
    auto second_operand = old_to_new_instrs_[consumer->mutable_operand(1)];

    auto permute_dims_first_operand = instr_to_dim_permute_map_[first_operand];
    auto permute_dims_second_operand =
        instr_to_dim_permute_map_[second_operand];

    // The permuting must match.
    if (permute_dims_first_operand != permute_dims_second_operand) {
      VLOG(2) << "Can't propagate through select and scatter due to "
                 "permutation mismatch";
      return false;
    }

    const int64_t old_batch_dim =
        dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        dim_map_val_op_0[DimMapper(SpaceToBatchDimMap::kSpace0)];

    const int64_t new_batch_dim =
        DimLookUp(permute_dims_first_operand, old_batch_dim);
    const int64_t new_space_dim =
        DimLookUp(permute_dims_first_operand, old_space_dim);

    if (first_operand->shape().dimensions(new_batch_dim) !=
        second_operand->shape().dimensions(new_batch_dim)) {
      VLOG(2)
          << "Can't propagate through select and scatter due to dim mismatch";
      return false;
    }

    const int64_t stride =
        consumer->window().dimensions(old_space_dim).stride();
    const int64_t pad_high =
        consumer->window().dimensions(old_space_dim).padding_high();
    const int64_t pad_low =
        consumer->window().dimensions(old_space_dim).padding_low();

    if ((first_operand->shape().dimensions(new_space_dim) + pad_high +
         pad_low) /
            stride !=
        second_operand->shape().dimensions(new_space_dim)) {
      VLOG(2) << "Can't propagate through select and scatter due to stride "
                 "mismatch";
      return false;
    }

    return IsSpaceToBatchedSpaceSizeSuitable(consumer);
  }
  return true;
}

void ConvolutionVisitor::PropagateOnBroadcast(HloInstruction* consumer,
                                              HloInstruction* producer) {
  auto new_producer = old_to_new_instrs_[producer];
  auto permute_dims = instr_to_dim_permute_map_[new_producer];
  auto dim_map_val = instr_to_dim_map_[producer];

  const int64_t old_batch_dim =
      dim_map_val[DimMapper(SpaceToBatchDimMap::kBatch)];
  const int64_t old_space_dim =
      dim_map_val[DimMapper(SpaceToBatchDimMap::kSpace0)];

  auto orig_broadcast_dims = consumer->dimensions();

  bool batch_is_broadcasted =
      absl::c_linear_search(orig_broadcast_dims, old_batch_dim);
  const int64_t new_batch_dim = DimLookUp(permute_dims, old_batch_dim);
  const int64_t new_space_dim = DimLookUp(permute_dims, old_space_dim);

  bool map_found = broadcast_map_.contains(consumer);
  if (map_found) {
    // Check if we previously had created the same broadcast.
    for (auto previous_broadcast : broadcast_map_[consumer]) {
      if (ShapeUtil::CompatibleIgnoringElementType(previous_broadcast->shape(),
                                                   new_producer->shape())) {
        return;
      }
    }
  }

  std::vector<int64_t> final_shape_dims(
      new_producer->shape().dimensions().begin(),
      new_producer->shape().dimensions().end());
  if (batch_is_broadcasted) {
    final_shape_dims[new_batch_dim] =
        producer->shape().dimensions(old_batch_dim);
    final_shape_dims[new_space_dim] *= ctrl_.number_of_splits;
  }

  std::vector<int64_t> broadcast_dims;
  const auto& dimensions = consumer->dimensions();
  broadcast_dims.reserve(dimensions.size());
  for (auto j : dimensions) {
    broadcast_dims.push_back(DimLookUp(permute_dims, j));
  }
  auto new_broadcast = MakeBroadcastHlo(
      consumer->mutable_operand(0), broadcast_dims, final_shape_dims,
      &consumer->metadata(), &consumer->frontend_attributes());
  VLOG(1) << "Created broadcast " << new_broadcast->ToString();

  if (batch_is_broadcasted) {
    new_broadcast =
        MakeReshapeHlo(new_producer->shape().dimensions(), new_broadcast)
            .value();
    VLOG(2) << "Created reshape of broadcast " << new_broadcast->ToString();
  }

  if (!map_found) {
    absl::flat_hash_set<HloInstruction*> set_of_broadcasts;
    broadcast_map_[consumer] = set_of_broadcasts;
  }
  broadcast_map_[consumer].insert(new_broadcast);
}

void ConvolutionVisitor::RewriteBroadcastTree(
    HloInstruction* producer,
    std::vector<HloInstruction*>& instructions_to_transform) {
  CHECK(old_to_new_instrs_.contains(producer));
  for (auto instr : instructions_to_transform) {
    if (instr->opcode() == HloOpcode::kBroadcast) {
      PropagateOnBroadcast(instr, producer);
    } else if (IsTrivialElementwise(instr)) {
      Propagate(instr, /*producer=*/instr->mutable_operand(0)).value();
    } else {
      LOG(FATAL) << "Unsupported opcode in RewriteBroadcastTree";
    }
  }
}

bool ConvolutionVisitor::IsBroadcastTree(
    HloInstruction* op, HloInstruction* consumer,
    std::vector<HloInstruction*>& instructions_to_transform) {
  if (op->opcode() == HloOpcode::kBroadcast) {
    // We want to ensure that the broadcast did not happen on the space and
    // batch dimensions.
    if (IsBroadcastPropagatable(op, consumer)) {
      instructions_to_transform.push_back(op);
      return true;
    } else {
      return false;
    }
  }
  if (Match(op, m::ConstantScalar())) {
    return true;
  }
  if (!IsTrivialElementwise(op)) {
    return false;
  }
  for (int64_t i = 0; i < op->operand_count(); ++i) {
    if (!IsBroadcastTree(op->mutable_operand(i), consumer,
                         instructions_to_transform)) {
      return false;
    }
  }
  instructions_to_transform.push_back(op);
  return true;
}

bool ConvolutionVisitor::IsBroadcastPropagatable(HloInstruction* broadcast,
                                                 HloInstruction* old_other_op) {
  CHECK_EQ(broadcast->opcode(), HloOpcode::kBroadcast);
  CHECK(instr_to_dim_map_.contains(old_other_op));

  auto result = instr_to_dim_map_[old_other_op];
  const int64_t space_dim = result[DimMapper(SpaceToBatchDimMap::kSpace0)];
  auto broadcast_dims = broadcast->dimensions();
  return !absl::c_linear_search(broadcast_dims, space_dim);
}

bool ConvolutionVisitor::IsOpcodeNonPropagatable(HloInstruction* consumer) {
  // We can add more non-propagatable opcodes as needed.
  switch (consumer->opcode()) {
    case HloOpcode::kCustomCall:
      return true;
    default:
      return false;
  }
}

bool ConvolutionVisitor::SupportedDotForPropagation(HloInstruction* consumer,
                                                    HloInstruction* producer) {
  if (consumer->opcode() != HloOpcode::kDot) {
    return false;
  }
  auto operand = consumer->mutable_operand(0);
  if (operand != producer || !instr_to_dim_map_.contains(operand)) {
    return false;
  }
  const auto& dnums = consumer->dot_dimension_numbers();
  const auto& contracting_dims = dnums.lhs_contracting_dimensions();
  const auto& batch_dims = dnums.lhs_batch_dimensions();
  auto result = instr_to_dim_map_[operand];
  const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
  const int64_t old_space_dim = result[DimMapper(SpaceToBatchDimMap::kSpace0)];
  const int64_t old_feature_dim =
      result[DimMapper(SpaceToBatchDimMap::kFeature)];
  // No feature dimension in output
  if (consumer->operand(1)->shape().rank() ==
      batch_dims.size() + contracting_dims.size()) {
    return false;
  }
  // If the convolution space or batch dimension are contracting or batch on
  // the dot, do not propagate.
  bool found = false;
  for (auto dim : batch_dims) {
    if (dim == old_batch_dim || dim == old_space_dim) {
      return false;
    }
    if (dim == old_feature_dim) {
      found = true;
    }
  }
  if (!found) {
    return false;
  }
  for (auto dim : contracting_dims) {
    if (dim == old_batch_dim || dim == old_space_dim) {
      return false;
    }
  }
  return true;
}

bool ConvolutionVisitor::SupportedOpForPropagation(HloInstruction* consumer,
                                                   HloInstruction* producer) {
  if (IsOpcodeNonPropagatable(consumer)) {
    return false;
  }

  if (IsTrivialElementwise(consumer)) {
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      if (consumer->operand(i)->opcode() == HloOpcode::kBroadcast) {
        if (!IsBroadcastPropagatable(consumer->mutable_operand(i), producer)) {
          VLOG(2) << "Could not propagate through broadcast";
          return false;
        }
      }
    }
    return true;
  }

  if (consumer->opcode() == HloOpcode::kConvolution) {
    return true;
  }

  if (consumer->opcode() == HloOpcode::kConcatenate) {
    HloInstruction* pivot_operand = nullptr;
    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      if (instr_to_dim_map_.contains(consumer->mutable_operand(i))) {
        pivot_operand = consumer->mutable_operand(i);
        break;
      }
    }
    if (pivot_operand == nullptr) {
      VLOG(1) << "Concat: Dim map not found on any operand";
      return false;
    }
    // Disallow concating on the batch and space dims
    auto result = instr_to_dim_map_[pivot_operand];
    const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        result[DimMapper(SpaceToBatchDimMap::kSpace0)];
    if (consumer->concatenate_dimension() == old_batch_dim ||
        consumer->concatenate_dimension() == old_space_dim) {
      return false;
    }
    return true;
  }

  if (consumer->opcode() == HloOpcode::kReverse) {
    auto operand_0 = consumer->mutable_operand(0);
    if (!instr_to_dim_map_.contains(operand_0)) {
      return false;
    }
    // Disallow reversing on the batch and space dims
    auto result = instr_to_dim_map_[operand_0];
    const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        result[DimMapper(SpaceToBatchDimMap::kSpace0)];

    for (auto dim : consumer->dimensions()) {
      if (dim == old_batch_dim || dim == old_space_dim) {
        return false;
      }
    }
    return true;
  }

  if (consumer->opcode() == HloOpcode::kTranspose) {
    return true;
  }

  if (consumer->opcode() == HloOpcode::kPad) {
    auto operand_0 = consumer->mutable_operand(0);
    if (!instr_to_dim_map_.contains(operand_0)) {
      return false;
    }
    // Disallow reversing on the batch and space dims
    auto result = instr_to_dim_map_[operand_0];
    const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        result[DimMapper(SpaceToBatchDimMap::kSpace0)];

    auto does_dim_have_padding = [](PaddingConfig padding_config, int64_t dim) {
      return padding_config.dimensions(dim).edge_padding_low() != 0 ||
             padding_config.dimensions(dim).edge_padding_high() != 0 ||
             padding_config.dimensions(dim).interior_padding() != 0;
    };
    // Batch and space dims should not have padding.
    if (does_dim_have_padding(consumer->padding_config(), old_batch_dim) ||
        does_dim_have_padding(consumer->padding_config(), old_space_dim)) {
      return false;
    }
    return true;
  }

  if (consumer->opcode() == HloOpcode::kSlice) {
    auto operand = consumer->mutable_operand(0);
    if (!instr_to_dim_map_.contains(operand)) {
      return false;
    }
    auto result = instr_to_dim_map_[operand];
    const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        result[DimMapper(SpaceToBatchDimMap::kSpace0)];
    // Disallow slice on the batch and space dims
    if (consumer->shape().dimensions(old_batch_dim) !=
        operand->shape().dimensions(old_batch_dim)) {
      return false;
    }
    if (consumer->shape().dimensions(old_space_dim) !=
        operand->shape().dimensions(old_space_dim)) {
      return false;
    }
    return true;
  }

  if (SupportedDotForPropagation(consumer, producer)) {
    return true;
  }

  if (consumer->opcode() == HloOpcode::kReduce) {
    // Support only the trivial case where both batch and split spatial dim are
    // being reduced

    auto reduce_dims = consumer->dimensions();
    auto result = instr_to_dim_map_[consumer->mutable_operand(0)];
    const int64_t batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t space_dim = result[DimMapper(SpaceToBatchDimMap::kSpace0)];
    VLOG(1) << "Checking if reduce is supported batch_dim " << batch_dim
            << " space_dim " << space_dim << " reduce " << consumer->ToString();
    return absl::c_linear_search(reduce_dims, batch_dim) &&
           absl::c_linear_search(reduce_dims, space_dim);
  }

  if (consumer->opcode() == HloOpcode::kReduceWindow &&
      consumer->shape().IsTuple()) {
    // TODO (b/73062247) variadic reduce window is not yet supported.
    return false;
  }
  if (consumer->opcode() == HloOpcode::kReduceWindow ||
      consumer->opcode() == HloOpcode::kSelectAndScatter) {
    auto first_operand = consumer->mutable_operand(0);
    auto window = consumer->window();
    if (instr_to_dim_map_.count(first_operand) <= 0) {
      VLOG(1) << "Dim map not found on windowed operand. Window dim count "
              << window.dimensions().size();
      return false;
    }
    // Disallow windowing on the batch dim
    auto result = instr_to_dim_map_[first_operand];
    const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        result[DimMapper(SpaceToBatchDimMap::kSpace0)];
    if (window.dimensions(old_batch_dim).size() != 1) {
      return false;
    }

    // Only allow no-low-padding cases.
    if (window.dimensions(old_space_dim).padding_low() != 0) {
      return false;
    }

    // No base/window dilations allowed on space and batch dimensions.
    if (window.dimensions(old_space_dim).base_dilation() != 1 ||
        window.dimensions(old_space_dim).window_dilation() != 1) {
      return false;
    }
    // No base/window dilations allowed on space and batch dimensions.
    if (window.dimensions(old_batch_dim).base_dilation() != 1 ||
        window.dimensions(old_batch_dim).window_dilation() != 1) {
      return false;
    }

    // Only allow small high pads.
    if (window.dimensions(old_space_dim).padding_high() >
        window.dimensions(old_space_dim).size()) {
      return false;
    }

    // Operand 0 must have been propagated through
    if (old_to_new_instrs_.count(first_operand) <= 0) {
      return false;
    }

    auto new_operand = old_to_new_instrs_[first_operand];
    auto permute_dims = instr_to_dim_permute_map_[new_operand];

    // Select-and-scatter specific checks.
    if (consumer->opcode() == HloOpcode::kSelectAndScatter) {
      const int64_t new_space_dim = DimLookUp(permute_dims, old_space_dim);
      // Make sure that the stride lines up.
      if (new_operand->shape().dimensions(new_space_dim) %
              window.dimensions(old_space_dim).stride() !=
          0) {
        return false;
      }

      // Only support floating point datatypes.
      if (!ShapeUtil::ElementIsFloating(consumer->shape())) {
        return false;
      }
      // We currently only support adds in the scatter.
      auto scatter_comp = consumer->scatter();
      if (!Match(scatter_comp->root_instruction(),
                 m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
        return false;
      }
      // Select should just be a single comparison with GE as the direction.
      auto select_comp = consumer->select();
      if (!Match(select_comp->root_instruction(),
                 m::Compare(m::Parameter(0), m::Parameter(1))
                     .WithComparisonDirection(ComparisonDirection::kGe)) &&
          !Match(select_comp->root_instruction(),
                 m::Compare(m::Parameter(1), m::Parameter(0))
                     .WithComparisonDirection(ComparisonDirection::kGe))) {
        return false;
      }
      // We do not support low padding on select-and-scatter.
      if (consumer->window().dimensions(old_space_dim).padding_low() != 0) {
        return false;
      }
    }

    return true;
  }

  return false;
}

absl::StatusOr<bool> ConvolutionVisitor::Propagate(HloInstruction* consumer,
                                                   HloInstruction* producer) {
  auto computation = consumer->parent();
  if (IsTrivialElementwise(consumer)) {
    auto dim_map_val = instr_to_dim_map_[producer];
    auto new_consumer = computation->AddInstruction(consumer->Clone());

    bool is_pivot_producer_modified = false;
    // For elementwise binary ops, both of whose operands have been space-to-
    // batched, if their new spatial sizes don't match, choose the bigger one
    // as the producer.
    if (consumer->IsElementwiseBinary() ||
        consumer->opcode() == HloOpcode::kSelect) {
      int64_t pivot_operand_number = -1;
      HloInstruction* pivot_operand = nullptr;
      for (int i = 0; i < consumer->operand_count(); ++i) {
        if (consumer->operand(i)->opcode() == HloOpcode::kBroadcast) {
          continue;
        }
        auto operand = consumer->mutable_operand(i);
        if (old_to_new_instrs_.contains(operand)) {
          if (pivot_operand_number == -1 ||
              old_to_new_instrs_[pivot_operand]->shape().dimensions() <
                  old_to_new_instrs_[operand]->shape().dimensions()) {
            is_pivot_producer_modified = true;
            pivot_operand_number = i;
            pivot_operand = consumer->mutable_operand(pivot_operand_number);
          }
        }
      }
      if (pivot_operand_number != -1) {
        producer = pivot_operand;
      }
    }

    for (int64_t i = 0; i < consumer->operand_count(); ++i) {
      std::vector<HloInstruction*> instructions_to_transform;

      if (consumer->operand(i)->opcode() == HloOpcode::kBroadcast) {
        auto broadcast = consumer->mutable_operand(i);
        PropagateOnBroadcast(broadcast, producer);
        HloInstruction* new_broadcast = nullptr;
        auto new_producer = old_to_new_instrs_[producer];
        for (auto previous_broadcast : broadcast_map_[broadcast]) {
          if (ShapeUtil::CompatibleIgnoringElementType(
                  previous_broadcast->shape(), new_producer->shape())) {
            new_broadcast = previous_broadcast;
            break;
          }
        }
        CHECK_NE(new_broadcast, nullptr);
        TF_CHECK_OK(
            new_consumer->ReplaceOperandWithDifferentShape(i, new_broadcast));
      } else if (old_to_new_instrs_.contains(consumer->mutable_operand(i))) {
        HloInstruction* operand_to_use = nullptr;
        auto result = instr_to_dim_map_[producer];
        const int64_t old_batch_dim =
            result[DimMapper(SpaceToBatchDimMap::kBatch)];
        const int64_t old_space_dim =
            result[DimMapper(SpaceToBatchDimMap::kSpace0)];
        const int64_t old_batch_size =
            producer->shape().dimensions(old_batch_dim);
        HloInstruction* new_instr =
            old_to_new_instrs_[consumer->mutable_operand(i)];
        HloInstruction* pivot_new_instr = old_to_new_instrs_[producer];

        auto permute_dims = instr_to_dim_permute_map_[new_instr];
        const int64_t batch_dim = DimLookUp(permute_dims, old_batch_dim);
        const int64_t space_dim = DimLookUp(permute_dims, old_space_dim);
        const int64_t batch_size = new_instr->shape().dimensions(batch_dim);

        if (new_instr->shape().dimensions(space_dim) !=
            pivot_new_instr->shape().dimensions(space_dim)) {
          // Because we do not propagate through transposes, the batch should
          // always be followed by the split space dimension.
          CHECK_EQ(batch_dim + 1, space_dim);

          // Reshape to 1D, pad to the producer's size, reshape back to 2D.
          std::vector<int64_t> new_dimensions(
              new_instr->shape().dimensions().begin(),
              new_instr->shape().dimensions().end());
          new_dimensions[space_dim] *= (batch_size / old_batch_size);
          new_dimensions[batch_dim] = old_batch_size;

          TF_ASSIGN_OR_RETURN(HloInstruction * reshape,
                              MakeReshapeHlo(new_dimensions, new_instr));

          const int64_t pivot_space_size =
              pivot_new_instr->shape().dimensions(space_dim) * batch_size /
              old_batch_size;

          CHECK(pivot_space_size > new_dimensions[space_dim] ||
                !is_pivot_producer_modified);

          PaddingConfig padding_config =
              MakeNoPaddingConfig(reshape->shape().dimensions_size());
          padding_config.mutable_dimensions(space_dim)->set_edge_padding_high(
              pivot_space_size - new_dimensions[space_dim]);
          padding_config.mutable_dimensions(space_dim)->set_edge_padding_low(0);
          HloInstruction* padding =
              consumer->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::Zero(reshape->shape().element_type())));
          TF_ASSIGN_OR_RETURN(
              HloInstruction * padded_operand,
              MakePadHlo(reshape, padding, padding_config, &reshape->metadata(),
                         &reshape->frontend_attributes()));

          TF_ASSIGN_OR_RETURN(
              operand_to_use,
              MakeReshapeHlo(pivot_new_instr->shape().dimensions(),
                             padded_operand));

        } else {
          operand_to_use = old_to_new_instrs_[consumer->mutable_operand(i)];
        }
        TF_CHECK_OK(
            new_consumer->ReplaceOperandWithDifferentShape(i, operand_to_use));
      } else if (consumer->IsElementwiseBinary() &&
                 consumer->mutable_operand(i)->opcode() ==
                     HloOpcode::kBroadcast &&
                 IsBroadcastTree(consumer->mutable_operand(i), producer,
                                 instructions_to_transform)) {
        RewriteBroadcastTree(producer, instructions_to_transform);
        TF_CHECK_OK(new_consumer->ReplaceOperandWithDifferentShape(
            i, old_to_new_instrs_[consumer->mutable_operand(i)]));
      } else if (consumer->operand(i)->opcode() == HloOpcode::kConstant) {
        TF_ASSIGN_OR_RETURN(
            auto new_constant,
            PropagateOnConstant(consumer->mutable_operand(i), producer));
        TF_CHECK_OK(
            new_consumer->ReplaceOperandWithDifferentShape(i, new_constant));
      }
    }
    auto old_type = new_consumer->mutable_shape()->element_type();
    *(new_consumer->mutable_shape()) = old_to_new_instrs_[producer]->shape();

    // The element type needs to be retained.
    new_consumer->mutable_shape()->set_element_type(old_type);

    old_to_new_instrs_[consumer] = new_consumer;
    instr_to_dim_map_[consumer] = std::vector<int64_t>(dim_map_val);
    CHECK(instr_to_dim_permute_map_.contains(old_to_new_instrs_[producer]));
    instr_to_dim_permute_map_[new_consumer] = std::vector<int64_t>(
        instr_to_dim_permute_map_[old_to_new_instrs_[producer]]);

    VLOG(2) << " new_consumer " << new_consumer->ToString()
            << " old_to_new_instrs_[producer] "
            << old_to_new_instrs_[producer]->ToString() << " permute dims "
            << instr_to_dim_permute_map_.count(new_consumer);

    return true;
  }

  if (consumer->opcode() == HloOpcode::kConvolution) {
    if (IsConvSuitableForSpaceToBatch(consumer)) {
      TF_CHECK_OK(PropagateOnConv(consumer));
      return true;
    } else {
      TF_CHECK_OK(PropagateOnBackpropFilterConv(consumer));
      return false;
    }
  }

  if (consumer->opcode() == HloOpcode::kConcatenate) {
    TF_CHECK_OK(PropagateOnConcat(consumer));
    return true;
  }

  if (consumer->opcode() == HloOpcode::kReverse) {
    TF_CHECK_OK(PropagateOnReverse(consumer));
    return true;
  }

  if (consumer->opcode() == HloOpcode::kDot) {
    auto dim_map_val = instr_to_dim_map_[producer];
    const int64_t old_batch_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kSpace0)];
    int64_t new_batch_dim = -1;
    int64_t new_space_dim = -1;
    int64_t outer = 0;
    for (int64_t i = 0; i < producer->shape().rank(); ++i) {
      if (absl::c_linear_search(
              consumer->dot_dimension_numbers().lhs_batch_dimensions(), i) ||
          absl::c_linear_search(
              consumer->dot_dimension_numbers().lhs_contracting_dimensions(),
              i)) {
        continue;
      }
      if (i == old_batch_dim) {
        new_batch_dim =
            outer +
            consumer->dot_dimension_numbers().lhs_batch_dimensions_size();
      }
      if (i == old_space_dim) {
        new_batch_dim =
            outer +
            consumer->dot_dimension_numbers().lhs_batch_dimensions_size();
      }
      ++outer;
    }
    std::vector<int64_t> dim_map(NumMappedDims());
    dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] = new_batch_dim;
    dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)] = new_space_dim;
    dim_map[DimMapper(SpaceToBatchDimMap::kFeature)] =
        consumer->shape().rank() - 1;
    instr_to_dim_map_[consumer] = dim_map;
    auto new_consumer = computation->AddInstruction(consumer->Clone());
    new_consumer->mutable_shape()->mutable_dimensions()[new_batch_dim] =
        producer->shape().dimensions(old_batch_dim);
    new_consumer->mutable_shape()->mutable_dimensions()[new_space_dim] =
        producer->shape().dimensions(old_space_dim);
    old_to_new_instrs_[consumer] = new_consumer;
    return true;
  }

  // TODO(b/189500737) : Consider a common way of propagation for
  // slice/pad/reduce-window.
  if (consumer->opcode() == HloOpcode::kPad) {
    TF_CHECK_OK(PropagateOnPad(consumer));
    return true;
  }

  if (consumer->opcode() == HloOpcode::kSlice) {
    TF_CHECK_OK(PropagateOnSlice(consumer));
    return true;
  }

  if (consumer->opcode() == HloOpcode::kReduce) {
    auto new_consumer = computation->AddInstruction(consumer->Clone());
    auto first_operand = old_to_new_instrs_[consumer->mutable_operand(0)];

    auto dim_map_val = instr_to_dim_map_[consumer->mutable_operand(0)];
    const int64_t old_batch_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kBatch)];

    auto permute_dims = instr_to_dim_permute_map_[first_operand];
    const int64_t new_batch_dim = DimLookUp(permute_dims, old_batch_dim);

    auto retval = GetSpatialDimsToSplit(consumer->mutable_operand(0));
    std::vector<int64_t> old_spatial_dims = retval.first;
    std::vector<int64_t> new_spatial_dims = retval.second;

    TF_ASSIGN_OR_RETURN(
        first_operand,
        SelectValidPortion(first_operand, consumer->mutable_operand(0),
                           consumer->mutable_operand(1), new_batch_dim,
                           new_spatial_dims, old_batch_dim, old_spatial_dims));

    std::vector<int64_t> changed_dims(new_consumer->dimensions().size());
    for (int64_t i = 0; i < new_consumer->dimensions().size(); ++i) {
      changed_dims[i] = DimLookUp(permute_dims, new_consumer->dimensions(i));
    }
    *(new_consumer->mutable_dimensions()) = changed_dims;
    // Replace operand 0.
    TF_CHECK_OK(
        new_consumer->ReplaceOperandWithDifferentShape(0, first_operand));
    // We do not set instr_to_dim_permute_map_ here because no further
    // propagation is needed here.
    old_to_new_instrs_[consumer] = new_consumer;
    instr_to_dim_map_[consumer] = std::vector<int64_t>(dim_map_val);

    // Since the resultant ordering of dimension is the same as before, no
    // further propagation is needed.
    return false;
  }

  if (consumer->opcode() == HloOpcode::kTranspose) {
    auto first_operand = old_to_new_instrs_[consumer->mutable_operand(0)];
    // Pass the first operand forward, with the map of the permuted dims.
    auto new_consumer = computation->AddInstruction(first_operand->Clone());
    old_to_new_instrs_[consumer] = new_consumer;
    auto dim_map_val = instr_to_dim_map_[consumer->mutable_operand(0)];
    const int64_t old_batch_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kSpace0)];
    const int64_t old_feature_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kFeature)];

    int64_t new_batch_dim, new_space_dim, new_feature_dim;
    std::vector<int64_t> new_dimensions(consumer->dimensions().size());
    for (int64_t ctr = 0; ctr < consumer->dimensions().size(); ++ctr) {
      int64_t dim = consumer->dimensions(ctr);
      if (dim == old_batch_dim) {
        new_batch_dim = ctr;
      }
      if (dim == old_space_dim) {
        new_space_dim = ctr;
      }
      if (dim == old_feature_dim) {
        new_feature_dim = ctr;
      }
    }

    std::vector<int64_t> dim_map(NumMappedDims());
    dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] = new_batch_dim;
    dim_map[DimMapper(SpaceToBatchDimMap::kFeature)] = new_feature_dim;
    dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)] = new_space_dim;
    instr_to_dim_map_[consumer] = dim_map;

    std::vector<int64_t> new_permute_dims(consumer->dimensions().size());
    auto permute_dims = instr_to_dim_permute_map_[first_operand];
    for (int64_t i = 0; i < consumer->dimensions().size(); ++i) {
      new_permute_dims[i] = DimLookUp(permute_dims, consumer->dimensions(i));
    }

    instr_to_dim_permute_map_[new_consumer] = new_permute_dims;

    return true;
  }

  if (consumer->opcode() == HloOpcode::kReduceWindow ||
      consumer->opcode() == HloOpcode::kSelectAndScatter) {
    bool is_select_and_scatter =
        consumer->opcode() == HloOpcode::kSelectAndScatter;
    auto first_operand = old_to_new_instrs_[consumer->mutable_operand(0)];

    auto init_val = is_select_and_scatter ? consumer->mutable_operand(2)
                                          : consumer->mutable_operand(1);
    auto dim_map_val = instr_to_dim_map_[consumer->mutable_operand(0)];

    auto retval = GetSpatialDimsToSplit(consumer->mutable_operand(0));
    std::vector<int64_t> old_spatial_dims = retval.first;
    std::vector<int64_t> new_spatial_dims = retval.second;

    const int64_t old_batch_dim =
        dim_map_val[DimMapper(SpaceToBatchDimMap::kBatch)];
    const int64_t old_space_dim = old_spatial_dims[0];
    auto permute_dims = instr_to_dim_permute_map_[first_operand];
    const int64_t new_batch_dim = DimLookUp(permute_dims, old_batch_dim);
    const int64_t new_space_dim = new_spatial_dims[0];

    // Calculate the required halo size
    auto new_shape = first_operand->shape();
    auto old_shape = consumer->mutable_operand(0)->shape();

    const int64_t new_space_size = new_shape.dimensions(new_space_dim);
    const int64_t stride =
        consumer->window().dimensions(old_space_dim).stride();

    auto pad_val =
        is_select_and_scatter
            ? consumer->AddInstruction(
                  HloInstruction::CreateConstant(LiteralUtil::MinValue(
                      consumer->operand(2)->shape().element_type())))
            : init_val;
    TF_ASSIGN_OR_RETURN(
        first_operand,
        SelectValidPortion(first_operand, consumer->mutable_operand(0), pad_val,
                           new_batch_dim, new_spatial_dims, old_batch_dim,
                           old_spatial_dims));

    const int64_t extra_space = new_space_size % stride;
    if (extra_space) {
      CHECK_EQ(consumer->opcode(), HloOpcode::kReduceWindow);
      const int64_t old_batch_size = old_shape.dimensions(old_batch_dim);
      const int64_t old_space_size = old_shape.dimensions(old_space_dim);
      // If the shrunk space is still larger/equal than the original space, we
      // reduce the space.
      if ((new_space_size - extra_space) * old_batch_size *
              ctrl_.number_of_splits >=
          old_batch_size * old_space_size) {
        TF_ASSIGN_OR_RETURN(
            first_operand, ChangeSpatialSizeOnSpaceToBatchedShape(
                               first_operand, new_batch_dim, old_batch_size,
                               new_spatial_dims, new_space_size - extra_space));
      } else {
        TF_ASSIGN_OR_RETURN(
            first_operand,
            ChangeSpatialSizeOnSpaceToBatchedShape(
                first_operand, new_batch_dim, old_batch_size, new_spatial_dims,
                new_space_size + stride - extra_space,
                /*increase_spatial_size*/ true));
      }
    }
    const int64_t window_size =
        consumer->window().dimensions(old_space_dim).size();
    const int64_t last_overlap_point = ((new_space_size - 1) / stride) * stride;
    VLOG(1) << "last_overlap_point " << last_overlap_point << " window_size "
            << window_size << " new_space_size " << new_space_size;

    const int64_t halo_size = last_overlap_point + window_size - new_space_size;
    if (halo_size > 0) {
      TF_ASSIGN_OR_RETURN(
          first_operand,
          HaloDuplicateWithSlice(first_operand, new_spatial_dims, new_batch_dim,
                                 /*low_padding=*/0, halo_size, init_val));
    }

    Window new_win;
    for (int64_t i = 0; i < consumer->window().dimensions().size(); ++i) {
      auto dim = ReverseDimLookUp(permute_dims, i);
      new_win.add_dimensions();
      new_win.mutable_dimensions(i)->set_stride(
          consumer->window().dimensions(dim).stride());
      new_win.mutable_dimensions(i)->set_size(
          consumer->window().dimensions(dim).size());
      if (i == old_space_dim) {
        new_win.mutable_dimensions(i)->set_padding_high(0);
        new_win.mutable_dimensions(i)->set_padding_low(0);
      } else {
        new_win.mutable_dimensions(i)->set_padding_high(
            consumer->window().dimensions(dim).padding_high());
        new_win.mutable_dimensions(i)->set_padding_low(
            consumer->window().dimensions(dim).padding_low());
      }
      new_win.mutable_dimensions(i)->set_window_dilation(
          consumer->window().dimensions(dim).window_dilation());
      new_win.mutable_dimensions(i)->set_base_dilation(
          consumer->window().dimensions(dim).base_dilation());
      new_win.mutable_dimensions(i)->set_window_reversal(
          consumer->window().dimensions(dim).window_reversal());
    }

    new_shape = first_operand->shape();

    HloInstruction* new_consumer = nullptr;
    if (is_select_and_scatter) {
      auto second_operand = old_to_new_instrs_[consumer->mutable_operand(1)];

      auto select_comp = consumer->select();

      auto scatter_comp = consumer->scatter();
      TF_ASSIGN_OR_RETURN(
          auto new_select_and_scatter_shape,
          ShapeInference::InferSelectAndScatterShape(
              new_shape, select_comp->ComputeProgramShape(), new_win,
              second_operand->shape(), init_val->shape(),
              scatter_comp->ComputeProgramShape()));
      new_consumer = computation_->AddInstruction(
          HloInstruction::CreateSelectAndScatter(
              new_select_and_scatter_shape, first_operand, select_comp, new_win,
              second_operand, init_val, scatter_comp),
          &consumer->metadata(), &consumer->frontend_attributes());
      // Replace operand 0.
      TF_CHECK_OK(
          new_consumer->ReplaceOperandWithDifferentShape(0, first_operand));
      // Replace operand 1.
      TF_CHECK_OK(
          new_consumer->ReplaceOperandWithDifferentShape(1, second_operand));
      VLOG(2) << "New select and scatter " << new_consumer->ToString();

      // If the window size was larger than the stride, there could be overlaps.
      // Such cases require updates from both overlaps to be applied.
      if (halo_size > 0) {
        const int64_t rank = new_consumer->shape().rank();

        const int64_t batch_size =
            new_consumer->shape().dimensions(new_batch_dim);

        std::vector<int64_t> start_indices(rank, 0),
            end_indices(new_consumer->shape().dimensions().begin(),
                        new_consumer->shape().dimensions().end()),
            strides(rank, 1);
        start_indices[new_space_dim] = new_space_size;
        end_indices[new_space_dim] = new_space_size + halo_size;
        end_indices[new_batch_dim] = batch_size - 1;

        // This is the slice from halo padding.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * bottom,
            MakeSliceHlo(new_consumer, start_indices, end_indices, strides,
                         &consumer->metadata(),
                         &consumer->frontend_attributes()));

        std::vector<int64_t> start_indices_top(rank, 0),
            end_indices_top(new_consumer->shape().dimensions().begin(),
                            new_consumer->shape().dimensions().end());
        end_indices_top[new_space_dim] = halo_size;
        // The first batch has correct data.
        start_indices_top[new_batch_dim] = 1;

        // This is the original area from where halo pad was extracted.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * top,
            MakeSliceHlo(new_consumer, start_indices_top, end_indices_top,
                         strides, &consumer->metadata(),
                         &consumer->frontend_attributes()));

        HloInstruction* default_fill = MakeBroadcastHlo(
            init_val, {}, top->shape().dimensions(), &init_val->metadata(),
            &init_val->frontend_attributes());

        // Compare to see if the bottom area was changed.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * bottom_compare,
            // TODO(hanrach): Verify that this is the correct metadata
            MakeCompareHlo(ComparisonDirection::kNe, bottom, default_fill,
                           &bottom->metadata(),
                           &bottom->frontend_attributes()));

        // Take out only the changed values.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * bottom_taken,
            MakeSelectHlo(bottom_compare, bottom, default_fill, nullptr,
                          &bottom_compare->metadata(),
                          &bottom_compare->frontend_attributes()));

        // Compare to see if the top area was changed.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * top_compare,
            MakeCompareHlo(ComparisonDirection::kNe, top, default_fill,
                           &top->metadata(), &top->frontend_attributes()));

        // Take out only the changed values.
        TF_ASSIGN_OR_RETURN(HloInstruction * top_taken,
                            MakeSelectHlo(top_compare, top, bottom_taken,
                                          nullptr, &top_compare->metadata(),
                                          &top_compare->frontend_attributes()));

        // This makes checks if the area was updated by both overlaps.
        TF_ASSIGN_OR_RETURN(HloInstruction * both_compare,
                            MakeBinaryHlo(HloOpcode::kAnd, top_compare,
                                          bottom_compare, &consumer->metadata(),
                                          &consumer->frontend_attributes()));

        // If it was, add them up.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * both_added,
            MakeBinaryHlo(HloOpcode::kAdd, top, bottom, &consumer->metadata(),
                          &consumer->frontend_attributes()));

        // Pad the final result to the original shape.
        TF_ASSIGN_OR_RETURN(
            HloInstruction * final_selection,
            MakeSelectHlo(both_compare, both_added, top_taken, nullptr,
                          &both_compare->metadata(),
                          &both_compare->frontend_attributes()));

        PaddingConfig padding_config =
            MakeNoPaddingConfig(final_selection->shape().dimensions_size());
        padding_config.mutable_dimensions(new_batch_dim)
            ->set_edge_padding_low(1);
        padding_config.mutable_dimensions(new_space_dim)
            ->set_edge_padding_high(new_space_size);
        HloInstruction* padding = computation_->AddInstruction(
            HloInstruction::CreateConstant(
                LiteralUtil::Zero(final_selection->shape().element_type())),
            &consumer->metadata(), &consumer->frontend_attributes());

        TF_ASSIGN_OR_RETURN(
            final_selection,
            MakePadHlo(final_selection, padding, padding_config,
                       &final_selection->metadata(),
                       &final_selection->frontend_attributes()));

        tsl::core::Bitmap b(batch_size * (new_space_size + halo_size));
        for (int k = 0; k < batch_size * (new_space_size + halo_size); ++k) {
          const int64_t space_index = k % (new_space_size + halo_size);
          const int64_t batch_index = (k / (new_space_size + halo_size));
          if (batch_index < 1 || space_index >= halo_size) {
            b.set(k);
          } else {
            b.clear(k);
          }
        }

        auto arg_literal = LiteralUtil::CreateR1(b);
        VLOG(4) << "Slice mask created: arg literal " << arg_literal.ToString();
        HloInstruction* slice_mask = computation_->AddInstruction(
            HloInstruction::CreateConstant(std::move(arg_literal)),
            &consumer->metadata(), &consumer->frontend_attributes());

        std::vector<int64_t> slice_mask_reshape_dims(2);
        slice_mask_reshape_dims[0] = batch_size;
        slice_mask_reshape_dims[1] = (new_space_size + halo_size);

        TF_ASSIGN_OR_RETURN(
            HloInstruction * slice_mask_reshaped,
            MakeReshapeHlo(slice_mask_reshape_dims, slice_mask));

        // Broadcast the mask in all dimensions.
        HloInstruction* shape_mask = MakeBroadcastHlo(
            slice_mask_reshaped, {new_batch_dim, new_space_dim},
            final_selection->shape().dimensions(), &slice_mask->metadata(),
            &slice_mask->frontend_attributes());

        TF_ASSIGN_OR_RETURN(
            new_consumer,
            MakeSelectHlo(shape_mask, new_consumer, final_selection, nullptr,
                          &shape_mask->metadata(),
                          &shape_mask->frontend_attributes()));
      }

      auto previous_shape =
          old_to_new_instrs_[consumer->mutable_operand(0)]->shape();
      std::vector<int64_t> start_indices(previous_shape.rank(), 0),
          end_indices(previous_shape.dimensions().begin(),
                      previous_shape.dimensions().end()),
          strides(previous_shape.rank(), 1);

      TF_ASSIGN_OR_RETURN(new_consumer,
                          MakeSliceHlo(new_consumer, start_indices, end_indices,
                                       strides, &consumer->metadata(),
                                       &consumer->frontend_attributes()));

    } else {
      auto reduce_comp = consumer->to_apply();
      TF_ASSIGN_OR_RETURN(auto new_reduce_window_shape,
                          ShapeInference::InferReduceWindowShape(
                              new_shape, init_val->shape(), new_win));
      new_consumer = computation_->AddInstruction(
          HloInstruction::CreateReduceWindow(new_reduce_window_shape,
                                             first_operand, init_val, new_win,
                                             reduce_comp),
          &consumer->metadata(), &consumer->frontend_attributes());
      // Replace operand 0.
      TF_CHECK_OK(
          new_consumer->ReplaceOperandWithDifferentShape(0, first_operand));
      VLOG(1) << "New reduce window " << new_consumer->ToString();
    }

    old_to_new_instrs_[consumer] = new_consumer;
    instr_to_dim_map_[consumer] = std::vector<int64_t>(dim_map_val);

    instr_to_dim_permute_map_[new_consumer] = std::vector<int64_t>(
        instr_to_dim_permute_map_[old_to_new_instrs_[consumer->mutable_operand(
            0)]]);

    return true;
  }

  LOG(FATAL) << "Trying to propagate through an unsupported instruction "
             << consumer->ToString();
  return true;
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::SelectValidPortion(
    HloInstruction* new_instr, HloInstruction* old_instr,
    HloInstruction* select_val, int64_t new_batch_dim,
    absl::Span<const int64_t> new_space_dims, int64_t old_batch_dim,
    absl::Span<const int64_t> old_space_dims) {
  auto new_shape = new_instr->shape();
  auto old_shape = old_instr->shape();
  VLOG(1) << "In SelectValidPortion new_batch_dim " << new_batch_dim
          << " new_space_dim " << new_space_dims[0] << " old_batch_dim "
          << old_batch_dim << " old_space_dim " << old_space_dims[0];
  const int64_t new_batch_size = new_shape.dimensions(new_batch_dim);
  const int64_t new_space_size = new_shape.dimensions(new_space_dims[0]);
  const int64_t old_batch_size = old_shape.dimensions(old_batch_dim);
  const int64_t old_space_size = old_shape.dimensions(old_space_dims[0]);
  CHECK_EQ(new_batch_size % old_batch_size, 0)
      << " New batch size " << new_batch_size << " old batch size "
      << old_batch_size;
  const int64_t num_splits = ctrl_.number_of_splits;
  const int64_t spatial_dim_count = new_space_dims.size();

  // The dimension ordering found bounds is Old Batch, BN, BN -1 .., B0, S0, S1
  // ..., SN
  std::vector<int64_t> bounds(2 + spatial_dim_count, new_space_size);
  bounds[0] = old_batch_size;
  bounds[1] = IPow<int64_t>(num_splits, spatial_dim_count);

  const int64_t total_new_space =
      IPow<int64_t>(new_space_size, spatial_dim_count);

  // Build a constant PRED to decide which elements in the split dimension
  // are from halo.
  tsl::core::Bitmap b(new_batch_size * total_new_space);
  for (int k = 0; k < new_batch_size * total_new_space; ++k) {
    auto radix = ToMixedRadix(k, bounds);

    bool out_of_bounds = false;
    int64_t batch_residue = 1;
    for (int i = 0; i < spatial_dim_count; ++i) {
      const int64_t space_index = radix[2 + i];
      const int64_t batch_index = (radix[1] / batch_residue) % num_splits;
      batch_residue *= num_splits;
      if (batch_index * new_space_size + space_index >= old_space_size) {
        out_of_bounds = true;
      }
    }

    if (!out_of_bounds) {
      b.set(k);
    } else {
      b.clear(k);
    }
  }

  auto arg_literal = LiteralUtil::CreateR1(b);
  VLOG(4) << "Slice mask created: arg literal " << arg_literal.ToString();
  HloInstruction* slice_mask = computation_->AddInstruction(
      HloInstruction::CreateConstant(std::move(arg_literal)),
      &old_instr->metadata(), &old_instr->frontend_attributes());

  std::vector<int64_t> slice_mask_reshape_dims(1 + spatial_dim_count,
                                               new_space_size);
  slice_mask_reshape_dims[0] = new_batch_size;

  TF_ASSIGN_OR_RETURN(HloInstruction * slice_mask_reshaped,
                      MakeReshapeHlo(slice_mask_reshape_dims, slice_mask));

  std::vector<int64_t> broadcast_dims(new_space_dims.begin(),
                                      new_space_dims.end());
  broadcast_dims.insert(broadcast_dims.begin(), new_batch_dim);
  // Broadcast the mask in all dimensions of the activations.
  HloInstruction* shape_mask = MakeBroadcastHlo(
      slice_mask_reshaped, broadcast_dims, new_instr->shape().dimensions(),
      &slice_mask_reshaped->metadata(),
      &slice_mask_reshaped->frontend_attributes());

  VLOG(1) << "Shape mask made " << shape_mask->ToString();

  HloInstruction* zeroes = MakeBroadcastHlo(
      select_val, {}, new_instr->shape().dimensions(), &select_val->metadata(),
      &select_val->frontend_attributes());

  TF_ASSIGN_OR_RETURN(new_instr,
                      MakeSelectHlo(shape_mask, new_instr, zeroes, nullptr,
                                    &shape_mask->metadata(),
                                    &shape_mask->frontend_attributes()));

  return new_instr;
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::BatchToSpace(
    HloInstruction* old_instr) {
  if (batch_to_space_map_.count(old_instr)) {
    CHECK_NE(batch_to_space_map_[old_instr], nullptr);
    return batch_to_space_map_[old_instr];
  }

  auto result = instr_to_dim_map_[old_instr];
  const int64_t old_batch_dim = result[DimMapper(SpaceToBatchDimMap::kBatch)];
  const int64_t old_space_dim = result[DimMapper(SpaceToBatchDimMap::kSpace0)];

  const int64_t old_batch_size = old_instr->shape().dimensions(old_batch_dim);
  CHECK(old_to_new_instrs_.contains(old_instr));
  auto new_instr = old_to_new_instrs_[old_instr];
  VLOG(2) << "old_batch_dim " << old_batch_dim << " old_space_dim "
          << old_space_dim << " old_instr " << old_instr->ToString()
          << "\n new_instr " << new_instr->ToString() << " permute dims "
          << instr_to_dim_permute_map_.count(new_instr) << " old_batch_size "
          << old_batch_size;
  CHECK(instr_to_dim_permute_map_.contains(new_instr));
  auto permute_dims = instr_to_dim_permute_map_[new_instr];
  const int64_t batch_dim = DimLookUp(permute_dims, old_batch_dim);
  const int64_t space_dim = DimLookUp(permute_dims, old_space_dim);

  const int64_t spatial_dim_size = new_instr->shape().dimensions(space_dim);

  std::vector<int64_t> split_spatial_dimensions(
      ctrl_.count_of_dimensions_to_convert);
  absl::c_iota(split_spatial_dimensions, space_dim);

  TF_ASSIGN_OR_RETURN(new_instr, SplitAndTransposeMergedBatch(
                                     new_instr, batch_dim, old_batch_size,
                                     split_spatial_dimensions));

  std::vector<int64_t> new_dimensions(new_instr->shape().dimensions().begin(),
                                      new_instr->shape().dimensions().end());

  new_dimensions.erase(new_dimensions.begin() + split_spatial_dimensions[0],
                       new_dimensions.begin() + split_spatial_dimensions[0] +
                           ctrl_.count_of_dimensions_to_convert);

  for (auto spatial_dimension : split_spatial_dimensions) {
    new_dimensions[spatial_dimension] =
        spatial_dim_size * ctrl_.number_of_splits;
  }

  // Reshape the output of the new conv into the old convolutions shape.
  TF_ASSIGN_OR_RETURN(HloInstruction * reshape,
                      MakeReshapeHlo(new_dimensions, new_instr));

  VLOG(1) << "Batch to space reshape " << reshape->ToString();
  const int64_t rank = old_instr->shape().rank();
  std::vector<int64_t> start_indices(rank, 0),
      end_indices(new_dimensions.begin(), new_dimensions.end()),
      strides(rank, 1);

  for (auto spatial_dimension : split_spatial_dimensions) {
    end_indices[spatial_dimension] =
        old_instr->shape().dimensions(old_space_dim);
  }

  // This slicing is getting rid of the padding we added to evenly divide space.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * output_slice,
      MakeSliceHlo(reshape, start_indices, end_indices, strides,
                   &reshape->metadata(), &reshape->frontend_attributes()));
  VLOG(1) << "Batch to space slice " << output_slice->ToString();
  std::vector<int64_t> transpose_dims(permute_dims);
  TF_ASSIGN_OR_RETURN(HloInstruction * output_transpose,
                      MakeTransposeHlo(output_slice, transpose_dims));
  old_instr->SetupDerivedInstruction(output_transpose);

  batch_to_space_map_[old_instr] = output_transpose;
  return output_transpose;
}

absl::Status ConvolutionVisitor::PropagateOnUsers(HloInstruction* old_conv) {
  std::queue<std::pair<HloInstruction*, HloInstruction*>> propagation_worklist;

  if (old_conv->user_count() == 0) {
    TF_ASSIGN_OR_RETURN(HloInstruction * batch_to_space,
                        BatchToSpace(old_conv));
    VLOG(1) << "Replacing the root instruction to "
            << batch_to_space->ToString();
    TF_CHECK_OK(computation_->ReplaceInstruction(old_conv, batch_to_space));
    VLOG(1) << "Replacement successful";
    return absl::OkStatus();
  }

  int64_t iteration_count = 0;
  propagation_worklist.push(
      std::make_pair(old_conv, old_conv->mutable_operand(0)));

  while (!propagation_worklist.empty()) {
    auto top = propagation_worklist.front();
    auto node = top.first;
    auto parent = top.second;
    VLOG(1) << "Traversing for propagation operating on " << node->ToString();
    propagation_worklist.pop();

    // Don't work on the same node again.
    if (old_to_new_instrs_.count(node) > 0 && iteration_count != 0) {
      continue;
    }

    bool needs_further_propagation = true;
    if (iteration_count != 0) {
      // Do the space-to-batch propagation on this node.
      TF_ASSIGN_OR_RETURN(needs_further_propagation, Propagate(node, parent));
    }
    iteration_count++;
    // If this is the root, no room for further propagation.
    if (node->parent()->root_instruction() == node) {
      // The below case does not need going back to space.
      if (!needs_further_propagation) {
        VLOG(1) << "Replacing the root instruction to "
                << old_to_new_instrs_[node]->ToString();
        TF_CHECK_OK(
            computation_->ReplaceInstruction(node, old_to_new_instrs_[node]));
        continue;
      }

      TF_ASSIGN_OR_RETURN(HloInstruction * batch_to_space, BatchToSpace(node));
      VLOG(1) << "Replacing the root instruction to "
              << batch_to_space->ToString();
      TF_CHECK_OK(computation_->ReplaceInstruction(node, batch_to_space));
    } else {
      if (!needs_further_propagation) {
        TF_CHECK_OK(
            computation_->ReplaceInstruction(node, old_to_new_instrs_[node]));
        continue;
      }

      HloInstructionSet unsupported_users;
      // Insert all users into the queue, as long as the ops are supported and
      // the op is ready for propagation. If the op is unsupported, do
      // batch-to-space. If not ready, mark as non-propagatable.
      for (auto user : node->users()) {
        if (!SupportedOpForPropagation(user, node)) {
          VLOG(1) << "Unsupported op found " << user->ToString();
          unsupported_users.insert(user);
          continue;
        }
        // If the instruction is ready for propagation, add it to the queue.
        if (CanPropagate(user, node)) {
          non_propagatable_instrs_.erase(user);
          propagation_worklist.push(std::make_pair(user, node));
        } else {
          // Mark it as non-propagatable for now, for later revisiting.
          non_propagatable_instrs_.insert(user);
        }
      }

      if (!unsupported_users.empty()) {
        TF_ASSIGN_OR_RETURN(HloInstruction * batch_to_space,
                            BatchToSpace(node));
        for (auto user : unsupported_users) {
          for (int64_t i = 0; i < user->operand_count(); ++i) {
            if (user->operand(i) == node) {
              TF_CHECK_OK(user->ReplaceOperandWith(i, batch_to_space));
            }
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status ConvolutionVisitor::PropagateOnConv(HloInstruction* convolution) {
  auto activations_old = convolution->mutable_operand(0);

  CHECK(old_to_new_instrs_.contains(activations_old));
  auto activations_new = old_to_new_instrs_[activations_old];
  auto permute_dims = instr_to_dim_permute_map_[activations_new];

  auto original_conv_dims = convolution->convolution_dimension_numbers();

  auto old_new_dims = GetSpatialDimsToSplit(activations_old);
  std::vector<int64_t> old_spatial_dims = old_new_dims.first;
  std::vector<int64_t> new_spatial_dims = old_new_dims.second;

  auto permuted_conv_dims_numbers = original_conv_dims;

  int64_t activations_batch_dim =
      DimLookUp(permute_dims, original_conv_dims.input_batch_dimension());
  int64_t activations_feature_dim =
      DimLookUp(permute_dims, original_conv_dims.input_feature_dimension());
  permuted_conv_dims_numbers.set_input_batch_dimension(activations_batch_dim);
  permuted_conv_dims_numbers.set_input_feature_dimension(
      activations_feature_dim);

  for (int64_t i = 0; i < original_conv_dims.input_spatial_dimensions_size();
       ++i) {
    permuted_conv_dims_numbers.set_input_spatial_dimensions(
        i, DimLookUp(permute_dims,
                     original_conv_dims.input_spatial_dimensions(i)));
  }

  const int64_t old_batch_dim = original_conv_dims.input_batch_dimension();
  const int64_t old_batch_size =
      activations_old->shape().dimensions(old_batch_dim);

  ConvDetails c =
      GetConvolutionDetails(convolution, permuted_conv_dims_numbers);

  VLOG(1) << "Propagating on conv activations_batch_dim "
          << activations_batch_dim << " spatial_dimension_to_split "
          << c.spatial_dimensions_to_split[0] << " old_batch_size "
          << old_batch_size;

  TF_ASSIGN_OR_RETURN(
      auto retval,
      BringSpaceNextToBatch(activations_new, permuted_conv_dims_numbers,
                            activations_batch_dim, &new_spatial_dims));
  activations_new = retval.instr;
  std::vector<int64_t> trans_dims = retval.transpose_dims;
  CHECK(!trans_dims.empty());
  auto select_val = computation_->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::Zero(activations_new->shape().element_type())),
      &convolution->metadata(), &convolution->frontend_attributes());

  TF_ASSIGN_OR_RETURN(
      activations_new,
      SelectValidPortion(activations_new, activations_old, select_val,
                         activations_batch_dim, new_spatial_dims, old_batch_dim,
                         old_spatial_dims));
  // Create the new convolution dim numbers.
  auto new_dim_numbers = permuted_conv_dims_numbers;

  const int64_t num_splits = ctrl_.number_of_splits;
  const int64_t output_offsets = convolution->shape().dimensions(
      permuted_conv_dims_numbers.output_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution)));
  const int64_t output_offsets_per_split =
      CeilOfRatio(output_offsets, num_splits);

  int64_t spatial_split_size =
      CeilOfRatio(output_offsets_per_split, c.base_dilation_factor) * c.stride;

  VLOG(1) << "spatial size " << c.spatial_size << " halo size " << c.halo_size
          << " spatial_split_size " << spatial_split_size;

  // Keep increasing the split size so that overall size isn't smaller than the
  // original spatial dimension. Unlike for the first space-to-batch'ed
  // convolution, while propagating, we can use the last halo_size as available
  // spatial size.
  // If the spatial size is less than the (halo size - low padding) required,
  // we need to increase the spatial size.
  while (spatial_split_size * num_splits + c.halo_size - c.spatial_size < 0 ||
         spatial_split_size < c.halo_size - c.inherent_low_padding) {
    spatial_split_size += c.stride;
  }

  VLOG(1) << "Modified spatial_split_size " << spatial_split_size;
  const int64_t new_space_size =
      activations_new->shape().dimensions(new_spatial_dims[0]);

  int64_t slice_size = spatial_split_size + c.halo_size;
  // In the below case, we cannot use the activations directly for Halo
  // Duplication. We must reshape them.
  if (spatial_split_size > new_space_size) {
    TF_ASSIGN_OR_RETURN(
        activations_new,
        ChangeSpatialSizeOnSpaceToBatchedShape(
            activations_new, activations_batch_dim, old_batch_size,
            new_spatial_dims, spatial_split_size,
            /*increase_spatial_size*/ true));

  } else {
    // If the ideal spatial_split_size was smaller than the incoming spatial
    // dimension size, we don't need reshaping. Instead, we determine the
    // additional space available, and adjust the required slice size (and
    // thereby the halo size).
    if (spatial_split_size < new_space_size) {
      VLOG(3)
          << "Decreasing the spatial size while propagating spatial_split_size "
          << spatial_split_size << " new_space_size " << new_space_size;
      // If there's a stride mismatch, we change the new_space_size be
      // smaller (equal to spatial_split_size).
      if (new_space_size % c.stride != 0 || c.base_dilation_factor != 1) {
        TF_ASSIGN_OR_RETURN(
            activations_new,
            ChangeSpatialSizeOnSpaceToBatchedShape(
                activations_new, activations_batch_dim, old_batch_size,
                new_spatial_dims, spatial_split_size));
      } else {
        const int64_t additional_space_present = spatial_split_size % c.stride;
        spatial_split_size = new_space_size;
        slice_size =
            spatial_split_size + std::max(c.kernel_spatial_dim_size - c.stride -
                                              additional_space_present,
                                          static_cast<int64_t>(0));
      }
    }
  }

  // For space-to-batch supported base-dilated convolutions, the low padding is
  // passed on to the new convolutions. Halo does not have to account for it.
  TF_ASSIGN_OR_RETURN(
      activations_new,
      HaloDuplicateWithSlice(
          activations_new, new_spatial_dims, activations_batch_dim,
          /*low_padding=*/c.base_dilation_factor != 1 &&
                  c.inherent_low_padding != 0
              ? (c.inherent_low_padding == c.base_dilation_factor ? 1 : 0)
              : c.inherent_low_padding,
          slice_size - spatial_split_size));

  // We will generate output such that batch is followed by the split spatial
  // dimension.
  const int64_t rank = (convolution->shape().rank());
  std::vector<int64_t> transpose_dims(rank);
  int dim_count = 0;
  std::map<int64_t, int64_t> dim_translator;

  for (int j = 0;
       j < permuted_conv_dims_numbers.output_spatial_dimensions_size(); ++j) {
    if (j == GetFirstChosenSpatialDim(convolution)) {
      dim_translator[permuted_conv_dims_numbers.output_batch_dimension()] =
          dim_count;
      new_dim_numbers.set_output_batch_dimension(dim_count++);
    }
    dim_translator[permuted_conv_dims_numbers.output_spatial_dimensions(j)] =
        dim_count;
    new_dim_numbers.set_output_spatial_dimensions(j, dim_count);
    dim_count++;
  }

  dim_translator[permuted_conv_dims_numbers.output_feature_dimension()] =
      dim_count;
  new_dim_numbers.set_output_feature_dimension(dim_count);

  int p = 0;
  for (const auto& entry : dim_translator) {
    transpose_dims[p] = entry.second;
    p++;
  }

  auto new_window = convolution->window();
  const int64_t first_dim = GetFirstChosenSpatialDim(convolution);
  for (int i = 0; i < ctrl_.count_of_dimensions_to_convert; ++i) {
    new_window.mutable_dimensions(first_dim + i)
        ->set_padding_high(c.high_padding_for_conv);
    new_window.mutable_dimensions(first_dim + i)
        ->set_padding_low(c.low_padding_for_conv);
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_conv,
      MakeConvolveHlo(
          activations_new, /*rhs=*/convolution->mutable_operand(1),
          convolution->feature_group_count(), convolution->batch_group_count(),
          new_window, new_dim_numbers, convolution->precision_config(),
          /*preferred_element_type=*/convolution->shape().element_type()));
  convolution->SetupDerivedInstruction(new_conv);

  old_to_new_instrs_[convolution] = new_conv;
  VLOG(1) << "Space-to-batched convolution " << new_conv->ToString();

  std::vector<int64_t> dim_map(NumMappedDims());
  dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] =
      original_conv_dims.output_batch_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kFeature)] =
      original_conv_dims.output_feature_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)] =
      original_conv_dims.output_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution));
  instr_to_dim_map_[convolution] = dim_map;

  instr_to_dim_permute_map_[new_conv] = std::vector<int64_t>(transpose_dims);

  convs_to_visit_.erase(convolution);
  return absl::OkStatus();
}

absl::Status ConvolutionVisitor::PropagateOnConcat(HloInstruction* concat) {
  auto first_operand = old_to_new_instrs_[concat->mutable_operand(0)];
  auto permute_dims = instr_to_dim_permute_map_[first_operand];
  const int64_t new_concat_dim =
      DimLookUp(permute_dims, concat->concatenate_dimension());
  std::vector<HloInstruction*> new_operands(concat->operand_count());
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    new_operands[i] = old_to_new_instrs_[concat->mutable_operand(i)];
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_concat,
      MakeConcatHlo(new_operands, new_concat_dim, &concat->metadata(),
                    &concat->frontend_attributes()));
  old_to_new_instrs_[concat] = new_concat;
  // Set mappings from operand 0.
  instr_to_dim_map_[concat] =
      std::vector<int64_t>(instr_to_dim_map_[concat->mutable_operand(0)]);
  instr_to_dim_permute_map_[new_concat] =
      std::vector<int64_t>(instr_to_dim_permute_map_[first_operand]);

  return absl::OkStatus();
}

absl::Status ConvolutionVisitor::PropagateOnReverse(HloInstruction* reverse) {
  auto first_operand = old_to_new_instrs_[reverse->mutable_operand(0)];
  auto permute_dims = instr_to_dim_permute_map_[first_operand];

  std::vector<int64_t> new_reverse_dimensions(reverse->dimensions().size());
  int dim_count = 0;
  for (auto dim : reverse->dimensions()) {
    new_reverse_dimensions[dim_count++] = DimLookUp(permute_dims, dim);
  }
  TF_ASSIGN_OR_RETURN(HloInstruction * new_reverse,
                      MakeReverseHlo(first_operand, new_reverse_dimensions));
  old_to_new_instrs_[reverse] = new_reverse;
  // Set mappings from operand 0.
  instr_to_dim_map_[reverse] =
      std::vector<int64_t>(instr_to_dim_map_[reverse->mutable_operand(0)]);
  instr_to_dim_permute_map_[new_reverse] =
      std::vector<int64_t>(instr_to_dim_permute_map_[first_operand]);

  return absl::OkStatus();
}

absl::Status ConvolutionVisitor::PropagateOnPad(HloInstruction* pad) {
  auto first_operand = old_to_new_instrs_[pad->mutable_operand(0)];
  auto permute_dims = instr_to_dim_permute_map_[first_operand];

  PaddingConfig padding_config;
  for (int i = 0; i < pad->shape().rank(); ++i) {
    auto dimension = padding_config.add_dimensions();
    const int64_t old_dim = ReverseDimLookUp(permute_dims, i);
    auto old_padding = pad->padding_config().dimensions(old_dim);
    dimension->set_edge_padding_low(old_padding.edge_padding_low());
    dimension->set_edge_padding_high(old_padding.edge_padding_high());
    dimension->set_interior_padding(old_padding.interior_padding());
  }

  HloInstruction* padding = pad->mutable_operand(1);

  TF_ASSIGN_OR_RETURN(auto new_pad,
                      MakePadHlo(first_operand, padding, padding_config,
                                 &first_operand->metadata(),
                                 &first_operand->frontend_attributes()));

  old_to_new_instrs_[pad] = new_pad;
  // Set mappings from operand 0.
  instr_to_dim_map_[pad] =
      std::vector<int64_t>(instr_to_dim_map_[pad->mutable_operand(0)]);
  instr_to_dim_permute_map_[new_pad] =
      std::vector<int64_t>(instr_to_dim_permute_map_[first_operand]);

  return absl::OkStatus();
}

absl::Status ConvolutionVisitor::PropagateOnSlice(HloInstruction* slice) {
  auto operand = old_to_new_instrs_[slice->mutable_operand(0)];
  auto permute_dims = instr_to_dim_permute_map_[operand];

  DimensionVector starts(slice->shape().rank());
  DimensionVector limits(slice->shape().rank());
  DimensionVector strides(slice->shape().rank());
  for (int i = 0; i < slice->shape().rank(); ++i) {
    const int64_t old_dim = ReverseDimLookUp(permute_dims, i);
    if (slice->shape().dimensions(old_dim) ==
        slice->operand(0)->shape().dimensions(old_dim)) {
      starts[i] = 0;
      strides[i] = 1;
      limits[i] = operand->shape().dimensions(i);
      continue;
    }
    starts[i] = slice->slice_starts(old_dim);
    strides[i] = slice->slice_strides(old_dim);
    limits[i] = slice->slice_limits(old_dim);
  }

  TF_ASSIGN_OR_RETURN(
      auto new_slice,
      MakeSliceHlo(operand, starts, limits, strides, &operand->metadata(),
                   &operand->frontend_attributes()));

  old_to_new_instrs_[slice] = new_slice;
  // Set mappings from operand 0.
  instr_to_dim_map_[slice] =
      std::vector<int64_t>(instr_to_dim_map_[slice->mutable_operand(0)]);
  instr_to_dim_permute_map_[new_slice] =
      std::vector<int64_t>(instr_to_dim_permute_map_[operand]);

  return absl::OkStatus();
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::TransposeAndMergeBatch(
    HloInstruction* activations,
    absl::Span<const int64_t> final_split_spatial_dim_positioning,
    int64_t activations_batch_dim, int64_t old_batch_size) {
  const int64_t spatial_dim_count = final_split_spatial_dim_positioning.size();

  if (final_split_spatial_dim_positioning.size() > 1) {
    int64_t start_batch_dim_position = activations_batch_dim + 1;
    int64_t start_space_dim_position =
        start_batch_dim_position + spatial_dim_count;

    std::vector<int64_t> trans_dims(activations->shape().dimensions_size());
    absl::c_iota(trans_dims, 0);

    for (int i = 0; i < spatial_dim_count; ++i) {
      trans_dims[start_batch_dim_position + i] =
          start_batch_dim_position + (spatial_dim_count - 1 - i) * 2;
      trans_dims[start_space_dim_position + i] =
          start_batch_dim_position + i * 2 + 1;
    }

    TF_ASSIGN_OR_RETURN(activations, MakeTransposeHlo(activations, trans_dims));
  }

  std::vector<int64_t> batch_collapse_reshape_dims(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());

  const int64_t collapsed_batch_size =
      old_batch_size * IPow<int64_t>(ctrl_.number_of_splits, spatial_dim_count);

  batch_collapse_reshape_dims.erase(
      batch_collapse_reshape_dims.begin() + activations_batch_dim,
      batch_collapse_reshape_dims.begin() + activations_batch_dim +
          spatial_dim_count);
  batch_collapse_reshape_dims[activations_batch_dim] = collapsed_batch_size;

  TF_ASSIGN_OR_RETURN(HloInstruction * batch_collapsed_reshape,
                      MakeReshapeHlo(batch_collapse_reshape_dims, activations));
  return batch_collapsed_reshape;
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::PerformSplitSpace(
    HloInstruction* activations,
    absl::Span<const int64_t> spatial_dimensions_to_split,
    int64_t activations_batch_dim, int64_t spatial_split_size,
    int64_t num_splits) {
  const int64_t old_batch_size =
      activations->shape().dimensions(activations_batch_dim);

  // Now we reorganize the activations. E.g. if the shape [B, SPACE] was [1, 16]
  // and 4 splits were needed, we first create [4, 4]. Next, to deal with halo
  // in the spatial dimension, we generate a gather. E.g. if halo size was 2,
  // we'd create a shape of [24] using the gather, and reshape it into [6, 4]
  // (4 being the batch).

  // The benefit of the above mentioned scheme is that it allows for batch
  // growth. Here are some examples of the size increases it causes for a 3x3
  // kernel.
  // with batch=1, [1,16] -> [4,4] ->   [4,6] ->   [1,24] growth of 8.
  // with batch=2, [2,16] -> [8,4] ->   [8,6] ->   [1,48] growth of 16.
  // with batch=3, [3,16] -> [12,4] -> [12,6] -> [1,72] growth of 24.

  std::vector<int64_t> reshape_dimensions(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());

  for (auto spatial_dimension_to_split : spatial_dimensions_to_split) {
    reshape_dimensions[spatial_dimension_to_split] = spatial_split_size;
  }

  int counter = 0;
  for (auto spatial_dimension_to_split : spatial_dimensions_to_split) {
    reshape_dimensions.insert(
        reshape_dimensions.begin() + (spatial_dimension_to_split + counter),
        num_splits);
    counter++;
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * batch_increased_reshape,
                      MakeReshapeHlo(reshape_dimensions, activations));

  return TransposeAndMergeBatch(
      batch_increased_reshape,
      /*final_split_spatial_dim_positioning=*/spatial_dimensions_to_split,
      activations_batch_dim, old_batch_size);
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::PadAndSplitSpace(
    HloInstruction* activations,
    absl::Span<const int64_t> spatial_dimensions_to_split,
    int64_t activations_batch_dim, int64_t high_padding, int64_t low_padding,
    int64_t spatial_split_size, int64_t num_splits) {
  const int64_t old_batch_size =
      activations->shape().dimensions(activations_batch_dim);

  // Because we are splitting the spatial dimension, if convolution needed
  // padding in the spatial dimension, we materialize it.
  if (high_padding || low_padding) {
    PaddingConfig padding_config =
        MakeNoPaddingConfig(activations->shape().dimensions_size());
    for (auto spatial_dimension_to_split : spatial_dimensions_to_split) {
      padding_config.mutable_dimensions(spatial_dimension_to_split)
          ->set_edge_padding_high(high_padding);
      padding_config.mutable_dimensions(spatial_dimension_to_split)
          ->set_edge_padding_low(low_padding);
    }
    HloInstruction* padding = computation_->AddInstruction(
        HloInstruction::CreateConstant(
            LiteralUtil::Zero(activations->shape().element_type())),
        &activations->metadata(), &activations->frontend_attributes());
    TF_ASSIGN_OR_RETURN(activations,
                        MakePadHlo(activations, padding, padding_config,
                                   &activations->metadata(),
                                   &activations->frontend_attributes()));
  }
  VLOG(1) << "Initial padded activations shape "
          << activations->shape().ToString() << " old_batch_size "
          << old_batch_size << " activations_batch_dim "
          << activations_batch_dim;
  return PerformSplitSpace(activations, spatial_dimensions_to_split,
                           activations_batch_dim, spatial_split_size,
                           num_splits);
}

absl::StatusOr<std::pair<HloInstruction*, std::vector<int64_t>>>
ConvolutionVisitor::SplitSpace(
    HloInstruction* activations, ConvolutionDimensionNumbers& dim_numbers,
    int64_t& activations_batch_dim, int64_t high_padding, int64_t low_padding,
    int64_t spatial_split_size, int64_t num_splits,
    std::vector<int64_t>* spatial_dimensions_to_split, bool is_backprop,
    bool is_rhs) {
  TF_ASSIGN_OR_RETURN(
      auto retval,
      BringSpaceNextToBatch(activations, dim_numbers, activations_batch_dim,
                            spatial_dimensions_to_split, is_backprop, is_rhs));

  activations = retval.instr;
  std::vector<int64_t> transpose_dims = retval.transpose_dims;
  TF_ASSIGN_OR_RETURN(
      auto new_activations,
      PadAndSplitSpace(activations, *spatial_dimensions_to_split,
                       activations_batch_dim, high_padding, low_padding,
                       spatial_split_size, num_splits));
  return std::make_pair(new_activations, transpose_dims);
}

absl::StatusOr<HloInstruction*> ConvolutionVisitor::PropagateOnConstant(
    HloInstruction* consumer, HloInstruction* producer) {
  CHECK(old_to_new_instrs_.contains(producer));
  HloInstruction* new_producer = old_to_new_instrs_[producer];
  auto prod_transpose_dims = instr_to_dim_permute_map_[new_producer];
  std::vector<int64_t> reversed_transpose_dims(prod_transpose_dims.size());
  for (int64_t i = 0; i < prod_transpose_dims.size(); ++i) {
    reversed_transpose_dims[i] = ReverseDimLookUp(prod_transpose_dims, i);
  }
  // Bring space next to batch.
  TF_ASSIGN_OR_RETURN(consumer,
                      MakeTransposeHlo(consumer, reversed_transpose_dims));

  auto retval = GetSpatialDimsToSplit(producer);
  std::vector<int64_t> old_spatial_dims = retval.first;
  std::vector<int64_t> new_spatial_dims = retval.second;

  auto dim_map = instr_to_dim_map_[producer];
  const int64_t old_batch_dim = dim_map[DimMapper(SpaceToBatchDimMap::kBatch)];
  const int64_t old_space_dim = old_spatial_dims[0];
  const int64_t new_batch_dim = DimLookUp(prod_transpose_dims, old_batch_dim);
  const int64_t new_space_dim = new_spatial_dims[0];

  const int64_t old_batch_size = producer->shape().dimensions(old_batch_dim);
  const int64_t new_batch_size = old_batch_size * ctrl_.number_of_splits;
  const int64_t high_padding =
      (new_batch_size * new_producer->shape().dimensions(new_space_dim) -
       old_batch_size * producer->shape().dimensions(old_space_dim)) /
      old_batch_size;

  auto new_consumer = PadAndSplitSpace(
      consumer, new_spatial_dims, new_batch_dim, high_padding,
      /*low_padding=*/0, new_producer->shape().dimensions(new_space_dim),
      ctrl_.number_of_splits);

  return new_consumer;
}

absl::Status ConvolutionVisitor::PropagateOnBackpropFilterConv(
    HloInstruction* convolution) {
  auto activations_old = convolution->mutable_operand(0);

  const int64_t rhs_dilation =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .window_dilation();

  auto original_conv_dims = convolution->convolution_dimension_numbers();

  std::vector<int64_t> old_split_spatial_dims(
      ctrl_.dimension_from_end_to_convert),
      old_split_kernel_spatial_dims(ctrl_.dimension_from_end_to_convert);
  for (int i = 0; i < ctrl_.dimension_from_end_to_convert; ++i) {
    old_split_spatial_dims[i] = original_conv_dims.input_spatial_dimensions(
        GetFirstChosenSpatialDim(convolution) + i);
    old_split_kernel_spatial_dims[i] =
        original_conv_dims.kernel_spatial_dimensions(
            GetFirstChosenSpatialDim(convolution) + i);
  }

  auto kernel_old = convolution->mutable_operand(1);
  const int64_t old_kernel_split_dim_size =
      kernel_old->shape().dimensions(old_split_kernel_spatial_dims[0]);

  int64_t old_split_dim_size =
      activations_old->shape().dimensions(old_split_spatial_dims[0]);

  int64_t old_batch_dim = original_conv_dims.input_feature_dimension();
  int64_t kernel_old_batch_dim =
      original_conv_dims.kernel_input_feature_dimension();
  const int64_t old_batch_size =
      activations_old->shape().dimensions(old_batch_dim);

  CHECK(old_to_new_instrs_.contains(kernel_old) ||
        old_to_new_instrs_.contains(activations_old));

  HloInstruction* activations_new = nullptr;
  HloInstruction* kernel_new = nullptr;
  bool activations_locally_space_to_batched = false;
  bool kernel_locally_space_to_batched = false;
  std::vector<int64_t> permute_dims_kernel, permute_dims;

  if (old_to_new_instrs_.contains(activations_old)) {
    activations_new = old_to_new_instrs_[activations_old];
    permute_dims = instr_to_dim_permute_map_[activations_new];
  }

  if (old_to_new_instrs_.contains(kernel_old)) {
    kernel_new = old_to_new_instrs_[kernel_old];
    permute_dims_kernel = instr_to_dim_permute_map_[kernel_new];
  }

  // If activations were not space-to-batched, we space-to-batch them below.
  if (!old_to_new_instrs_.contains(activations_old)) {
    kernel_new = old_to_new_instrs_[kernel_old];
    permute_dims_kernel = instr_to_dim_permute_map_[kernel_new];

    VLOG(1) << "Space-to-batching activations to enable space-to-depth";

    const int64_t new_kernel_space_dim =
        DimLookUp(permute_dims_kernel, old_split_kernel_spatial_dims[0]);

    const int64_t new_kernel_split_dim_size =
        kernel_new->shape().dimensions(new_kernel_space_dim);
    const int64_t needed_spatial_size =
        rhs_dilation * new_kernel_split_dim_size;
    const int64_t pad_size =
        needed_spatial_size * ctrl_.number_of_splits - old_split_dim_size;
    ConvolutionDimensionNumbers tmp_dim_numbers;
    tmp_dim_numbers = original_conv_dims;
    TF_ASSIGN_OR_RETURN(
        auto retval, SplitSpace(activations_old, tmp_dim_numbers, old_batch_dim,
                                /*high_padding=*/pad_size, /*low_padding=*/0,
                                needed_spatial_size, ctrl_.number_of_splits,
                                &old_split_spatial_dims,
                                /*is_backprop=*/true));

    activations_new = retval.first;

    std::vector<int64_t> reversed_transpose_dims(retval.second.size());
    for (int64_t i = 0; i < retval.second.size(); ++i) {
      reversed_transpose_dims[i] = ReverseDimLookUp(retval.second, i);
    }
    permute_dims = reversed_transpose_dims;

    VLOG(3) << "New Activations " << retval.first->ToString();

    activations_locally_space_to_batched = true;
  } else if (!old_to_new_instrs_.contains(kernel_old)) {
    activations_new = old_to_new_instrs_[activations_old];
    permute_dims = instr_to_dim_permute_map_[activations_new];

    VLOG(1) << "Space-to-batching kernel to enable space-to-depth";

    const int64_t new_space_dim =
        DimLookUp(permute_dims, old_split_spatial_dims[0]);
    const int64_t new_split_dim_size =
        activations_new->shape().dimensions(new_space_dim);
    const int64_t needed_spatial_size =
        CeilOfRatio(new_split_dim_size, rhs_dilation);
    int64_t old_kernel_split_dim_size =
        kernel_old->shape().dimensions(old_split_kernel_spatial_dims[0]);
    const int64_t pad_size = needed_spatial_size * ctrl_.number_of_splits -
                             old_kernel_split_dim_size;

    ConvolutionDimensionNumbers tmp_dim_numbers;
    tmp_dim_numbers = original_conv_dims;
    TF_ASSIGN_OR_RETURN(
        auto retval,
        SplitSpace(kernel_old, tmp_dim_numbers, kernel_old_batch_dim,
                   /*high_padding=*/pad_size, /*low_padding=*/0,
                   needed_spatial_size, ctrl_.number_of_splits,
                   &old_split_kernel_spatial_dims,
                   /*is_backprop=*/true, /*is_rhs=*/true));

    kernel_new = retval.first;

    std::vector<int64_t> reversed_transpose_dims(retval.second.size());
    for (int64_t i = 0; i < retval.second.size(); ++i) {
      reversed_transpose_dims[i] = ReverseDimLookUp(retval.second, i);
    }
    permute_dims_kernel = reversed_transpose_dims;

    VLOG(3) << "New kernel " << retval.first->ToString();

    kernel_locally_space_to_batched = true;
  }

  CHECK_NE(activations_new, nullptr);
  CHECK_NE(kernel_new, nullptr);

  // TODO(b/189500737): For multi-dimensional space-to-batch, we'd need to add
  // an auxiliary dim per converted dimension.
  const int64_t new_spatial_dimension =
      activations_new->shape().dimensions_size();

  auto permuted_conv_dims_numbers = original_conv_dims;

  // Note the inversion here : batch and feature are inverted in backprop
  // filters.
  int64_t activations_batch_dim =
      DimLookUp(permute_dims, original_conv_dims.input_feature_dimension());
  int64_t activations_feature_dim =
      DimLookUp(permute_dims, original_conv_dims.input_batch_dimension());

  const int64_t previous_spatial_dim_count =
      original_conv_dims.input_spatial_dimensions_size();
  for (int64_t i = 0; i < previous_spatial_dim_count; ++i) {
    permuted_conv_dims_numbers.set_input_spatial_dimensions(
        i, DimLookUp(permute_dims,
                     original_conv_dims.input_spatial_dimensions(i)));
    permuted_conv_dims_numbers.set_kernel_spatial_dimensions(
        i, DimLookUp(permute_dims_kernel,
                     original_conv_dims.kernel_spatial_dimensions(i)));
  }

  permuted_conv_dims_numbers.add_input_spatial_dimensions(
      new_spatial_dimension);
  permuted_conv_dims_numbers.add_kernel_spatial_dimensions(
      new_spatial_dimension);
  permuted_conv_dims_numbers.add_output_spatial_dimensions(
      new_spatial_dimension);

  // For the output, make the last dimension size 1.
  const int64_t previous_chosen_spatial_dim_in_output =
      permuted_conv_dims_numbers.output_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution));
  permuted_conv_dims_numbers.set_output_spatial_dimensions(
      GetFirstChosenSpatialDim(convolution), new_spatial_dimension);
  permuted_conv_dims_numbers.set_output_spatial_dimensions(
      previous_spatial_dim_count, previous_chosen_spatial_dim_in_output);

  const int64_t kernel_input_feature_dim = DimLookUp(
      permute_dims_kernel, original_conv_dims.kernel_input_feature_dimension());

  const int64_t kernel_output_feature_dim =
      DimLookUp(permute_dims_kernel,
                original_conv_dims.kernel_output_feature_dimension());

  permuted_conv_dims_numbers.set_kernel_input_feature_dimension(
      kernel_input_feature_dim);
  permuted_conv_dims_numbers.set_kernel_output_feature_dimension(
      kernel_output_feature_dim);

  std::vector<int64_t> spatial_dimensions_to_split(
      ctrl_.count_of_dimensions_to_convert);
  const int64_t first_dim_to_split = GetFirstChosenSpatialDim(convolution);
  for (int64_t i = 0; i < ctrl_.count_of_dimensions_to_convert; ++i) {
    spatial_dimensions_to_split[i] =
        permuted_conv_dims_numbers.input_spatial_dimensions(first_dim_to_split +
                                                            i);
  }

  const int64_t kernel_spatial_dimension_to_split =
      permuted_conv_dims_numbers.kernel_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution));

  int64_t new_split_dim_size =
      activations_new->shape().dimensions(spatial_dimensions_to_split[0]);

  const int64_t kernel_new_split_dim_size =
      kernel_new->shape().dimensions(kernel_spatial_dimension_to_split);

  permuted_conv_dims_numbers.set_input_batch_dimension(activations_feature_dim);
  permuted_conv_dims_numbers.set_input_feature_dimension(activations_batch_dim);

  VLOG(1) << "Propagating on conv activations_batch_dim "
          << activations_batch_dim << " spatial_dimension_to_split "
          << spatial_dimensions_to_split[0] << " old_batch_size "
          << old_batch_size << " new_split_dim_size " << new_split_dim_size;

  TF_ASSIGN_OR_RETURN(
      auto retval,
      BringSpaceNextToBatch(activations_new, permuted_conv_dims_numbers,
                            activations_batch_dim, &spatial_dimensions_to_split,
                            /*is_backprop=*/true));

  int64_t spatial_dimension_to_split = spatial_dimensions_to_split[0];

  std::vector<int64_t> transpose_dims = retval.transpose_dims;
  CHECK(!transpose_dims.empty());
  activations_new = retval.instr;

  VLOG(1) << "Activations_new post BringSpaceNextToBatch "
          << activations_new->ToString();
  VLOG(1) << "activations_batch_dim " << activations_batch_dim
          << " activations_feature_dim " << activations_feature_dim;
  const int64_t expected_split_dim_size =
      rhs_dilation * kernel_new_split_dim_size;
  if (new_split_dim_size != expected_split_dim_size) {
    CHECK_LT(new_split_dim_size, expected_split_dim_size);
    new_split_dim_size = expected_split_dim_size;
    TF_ASSIGN_OR_RETURN(
        activations_new,
        ChangeSpatialSizeOnSpaceToBatchedShape(
            activations_new, activations_batch_dim, old_batch_size,
            spatial_dimensions_to_split, new_split_dim_size, true));
  }

  spatial_dimension_to_split = spatial_dimensions_to_split[0];
  auto select_val = computation_->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::Zero(activations_new->shape().element_type())),
      &activations_new->metadata(), &activations_new->frontend_attributes());

  if (!activations_locally_space_to_batched) {
    // Select activations correctly by masking additional space.
    TF_ASSIGN_OR_RETURN(
        activations_new,
        SelectValidPortion(activations_new, activations_old, select_val,
                           activations_batch_dim, spatial_dimensions_to_split,
                           old_batch_dim, old_split_spatial_dims));
  }
  if (!kernel_locally_space_to_batched) {
    VLOG(3) << "Selecting the valid kernel area";
    // Select kernel correctly by masking additional space.

    std::vector<int64_t> new_kernel_split_spatial_dims(
        ctrl_.dimension_from_end_to_convert);

    // TODO(b/189500737) : Extend this once
    // IncreaseSpatialSizeOnSpaceToBatchedShape returns all dimensions.
    new_kernel_split_spatial_dims[0] = kernel_spatial_dimension_to_split;

    TF_ASSIGN_OR_RETURN(
        kernel_new,
        SelectValidPortion(kernel_new, kernel_old, select_val,
                           /*new_batch_dim=*/kernel_input_feature_dim,
                           new_kernel_split_spatial_dims,
                           /*old_batch_dim=*/
                           original_conv_dims.kernel_input_feature_dimension(),
                           old_split_kernel_spatial_dims));
  }

  // Create the new convolution dim numbers.
  auto new_dim_numbers = permuted_conv_dims_numbers;

  VLOG(2) << "New dim numbers " << new_dim_numbers.DebugString();

  const int64_t inherent_low_padding =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .padding_low();

  const int64_t inherent_high_padding =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .padding_high();

  std::vector<HloInstruction*> activations_chunks;

  // Insert slices for low padding.
  for (int64_t i = 0; i < inherent_low_padding; ++i) {
    HloInstruction* activations_to_use = nullptr;
    if (i == 0) {
      activations_to_use = activations_new;
    } else {
      activations_to_use = activations_chunks.back();
    }
    TF_ASSIGN_OR_RETURN(
        HloInstruction * activations_slice,
        HaloDuplicateWithSlice(activations_to_use, spatial_dimensions_to_split,
                               activations_batch_dim, /*low_padding=*/1,
                               /*halo_size=*/0));
    activations_chunks.push_back(activations_slice);
  }
  // Reverse the low padding slices because we created them in the opposite
  // order above.
  absl::c_reverse(activations_chunks);

  const int64_t expanded_kernel =
      old_kernel_split_dim_size * rhs_dilation - (rhs_dilation - 1);
  const int64_t overlap_count =
      old_split_dim_size - expanded_kernel + 1 +
      (inherent_low_padding < 0 ? inherent_low_padding : 0) +
      (inherent_high_padding < 0 ? inherent_high_padding : 0);
  VLOG(1) << "overlap_count " << overlap_count << " inherent_low_padding "
          << inherent_low_padding << " inherent_high_padding "
          << inherent_high_padding;

  const int64_t total_overlap_count =
      overlap_count + (inherent_low_padding > 0 ? inherent_low_padding : 0) +
      (inherent_high_padding > 0 ? inherent_high_padding : 0);

  // Insert original activations.
  for (int64_t i = 0; i < overlap_count; ++i) {
    HloInstruction* activations_to_use = nullptr;
    HloInstruction* activations_slice = nullptr;
    if (i == 0) {
      activations_to_use = activations_new;
      if (inherent_low_padding < 0) {
        TF_ASSIGN_OR_RETURN(
            activations_slice,
            HaloDuplicateWithSlice(
                activations_to_use, spatial_dimensions_to_split,
                activations_batch_dim,
                /*low_padding=*/inherent_low_padding, /*halo_size=*/0));
      } else {
        activations_slice = activations_to_use;
      }
    } else {
      activations_to_use = activations_chunks.back();

      TF_ASSIGN_OR_RETURN(activations_slice,
                          HaloDuplicateWithSlice(
                              activations_to_use, spatial_dimensions_to_split,
                              activations_batch_dim, /*low_padding=*/-1,
                              /*halo_size=*/0));
    }

    activations_chunks.push_back(activations_slice);
  }

  int64_t high_padding_to_materialize = 0;

  if (inherent_high_padding > 0) {
    high_padding_to_materialize =
        std::max(total_overlap_count -
                     (std::max(overlap_count, static_cast<int64_t>(0)) +
                      std::max(inherent_low_padding, static_cast<int64_t>(0))),
                 static_cast<int64_t>(0));
  }

  // Insert slices for high padding.
  for (int64_t i = 0; i < high_padding_to_materialize; ++i) {
    HloInstruction* activations_to_use = nullptr;
    activations_to_use = activations_chunks.back();

    TF_ASSIGN_OR_RETURN(
        HloInstruction * activations_slice,
        HaloDuplicateWithSlice(activations_to_use, spatial_dimensions_to_split,
                               activations_batch_dim,
                               /*low_padding=*/-1, /*halo_size=*/0));
    activations_chunks.push_back(activations_slice);
  }

  for (int64_t i = 0; i < activations_chunks.size(); ++i) {
    std::vector<int64_t> input_sizes(
        activations_chunks[i]->shape().dimensions().begin(),
        activations_chunks[i]->shape().dimensions().end());
    // Insert 1-sized dimension at the end
    input_sizes.push_back(1);
    TF_ASSIGN_OR_RETURN(activations_chunks[i],
                        MakeReshapeHlo(input_sizes, activations_chunks[i]));
    VLOG(1) << "new_spatial_dimension " << new_spatial_dimension << " slice "
            << activations_chunks[i]->ToString();
  }

  TF_ASSIGN_OR_RETURN(
      activations_new,
      MakeConcatHlo(absl::MakeSpan(activations_chunks), new_spatial_dimension,
                    &activations_old->metadata(),
                    &activations_old->frontend_attributes()));

  // Reshape the kernel with additional spatial dim.
  std::vector<int64_t> kernel_sizes(kernel_new->shape().dimensions().begin(),
                                    kernel_new->shape().dimensions().end());
  // Insert 1-sized dimension at the end
  kernel_sizes.push_back(1);
  TF_ASSIGN_OR_RETURN(kernel_new, MakeReshapeHlo(kernel_sizes, kernel_new));

  auto new_window = convolution->window();
  new_window.mutable_dimensions(GetFirstChosenSpatialDim(convolution))
      ->set_padding_high(-(rhs_dilation - 1));
  new_window.mutable_dimensions(GetFirstChosenSpatialDim(convolution))
      ->set_padding_low(0);
  new_window.mutable_dimensions(GetFirstChosenSpatialDim(convolution))
      ->set_size(CeilOfRatio(new_split_dim_size, rhs_dilation));

  // Set the window for the additional spatial dim. This is a vanilla window.
  auto window_dim = new_window.add_dimensions();
  window_dim->set_base_dilation(1);
  window_dim->set_size(1);
  int64_t stride = 1;
  // This condition means there's only a single overlap possible (as the shapes
  // were grown due to padding). In this case, we increase the stride.
  if (inherent_low_padding > total_overlap_count) {
    stride = activations_chunks.size();
  }
  window_dim->set_stride(stride);
  window_dim->set_padding_low(0);
  window_dim->set_padding_high(0);
  window_dim->set_window_reversal(false);
  window_dim->set_window_dilation(1);

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_conv,
      MakeConvolveHlo(
          activations_new, kernel_new, convolution->feature_group_count(),
          convolution->batch_group_count(), new_window, new_dim_numbers,
          convolution->precision_config(),
          /*preferred_element_type=*/convolution->shape().element_type()));
  convolution->SetupDerivedInstruction(new_conv);

  VLOG(2) << "New backprop filter convolution " << new_conv->ToString();

  std::vector<int64_t> output_sizes(new_conv->shape().dimensions().begin(),
                                    new_conv->shape().dimensions().end());

  output_sizes.erase(output_sizes.begin() +
                     new_dim_numbers.output_spatial_dimensions(
                         GetFirstChosenSpatialDim(convolution)));

  TF_ASSIGN_OR_RETURN(new_conv, MakeReshapeHlo(output_sizes, new_conv));

  old_to_new_instrs_[convolution] = new_conv;
  VLOG(1) << "Space-to-featured convolution " << new_conv->ToString();

  std::vector<int64_t> dim_map(NumMappedDims());
  dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] =
      original_conv_dims.output_batch_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kFeature)] =
      original_conv_dims.output_feature_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)] =
      original_conv_dims.output_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution));
  instr_to_dim_map_[convolution] = dim_map;

  std::vector<int64_t> trans_dims(convolution->shape().dimensions_size());
  absl::c_iota(trans_dims, 0);
  instr_to_dim_permute_map_[new_conv] = trans_dims;

  return absl::OkStatus();
}

HloInstruction*
ConvolutionVisitor::DoesConvolutionFeedReduceWindowOrSelectAndScatter(
    HloInstruction* instr, int64_t depth = kReduceWindowSearchDepth) {
  if (depth == 0) {
    return nullptr;
  }

  for (auto user : instr->users()) {
    if (user->opcode() == HloOpcode::kReduceWindow ||
        user->opcode() == HloOpcode::kSelectAndScatter) {
      return user;
    }
    // Stop the search if these ops are encountered.
    if (user->opcode() == HloOpcode::kConvolution ||
        user->opcode() == HloOpcode::kPad ||
        user->opcode() == HloOpcode::kTranspose ||
        user->opcode() == HloOpcode::kDot) {
      continue;
    }
    auto ret =
        DoesConvolutionFeedReduceWindowOrSelectAndScatter(user, depth - 1);
    if (ret != nullptr) {
      return ret;
    }
  }
  return nullptr;
}

bool ConvolutionVisitor::DoesConvolutionFeedUnpropagatableOp(
    HloInstruction* instr, int64_t depth) {
  auto key = std::make_pair(instr, depth);
  if (unpropagatability_cache_.contains(key)) {
    return unpropagatability_cache_[key];
  }

  if (depth == 0 || instr->user_count() == 0) {
    unpropagatability_cache_[key] = false;
    return false;
  }

  for (auto user : instr->users()) {
    if (IsOpcodeNonPropagatable(user)) {
      unpropagatability_cache_[key] = true;
      return true;
    }

    int64_t depth_to_use = depth;
    // When we see a convolution, we reduce the depth to look further for.
    if (user->opcode() == HloOpcode::kConvolution) {
      depth_to_use--;
    }

    if (DoesConvolutionFeedUnpropagatableOp(user, depth_to_use)) {
      unpropagatability_cache_[key] = true;
      return true;
    }
  }

  unpropagatability_cache_[key] = false;
  return false;
}

bool ConvolutionVisitor::IsSpaceToBatchedSpaceSizeSuitable(
    HloInstruction* instr) {
  CHECK(instr->opcode() == HloOpcode::kSelectAndScatter ||
        instr->opcode() == HloOpcode::kReduceWindow);
  auto old_producer = instr->mutable_operand(0);

  auto dim_map_val_op = instr_to_dim_map_[old_producer];
  const int64_t old_space_dim =
      dim_map_val_op[DimMapper(SpaceToBatchDimMap::kSpace0)];
  auto first_operand = old_to_new_instrs_[old_producer];
  auto permute_dims_first_operand = instr_to_dim_permute_map_[first_operand];
  const int64_t new_space_dim =
      DimLookUp(permute_dims_first_operand, old_space_dim);

  const int64_t window_size = instr->window().dimensions(old_space_dim).size();

  if (first_operand->shape().dimensions(new_space_dim) < window_size) {
    return false;
  }

  return true;
}

ConvolutionVisitor::ConvDetails ConvolutionVisitor::GetConvolutionDetails(
    HloInstruction* convolution, ConvolutionDimensionNumbers& dim_numbers) {
  auto activations = convolution->mutable_operand(0);

  auto kernel = convolution->mutable_operand(1);
  const auto& kernel_shape = kernel->shape();
  const int64_t kernel_spatial_dim = dim_numbers.kernel_spatial_dimensions(
      GetFirstChosenSpatialDim(convolution));
  int64_t kernel_spatial_dim_size = kernel_shape.dimensions(kernel_spatial_dim);

  if (IsForwardWindowDilatedConv(convolution, dim_numbers)) {
    const int64_t window_dilation_factor =
        convolution->window()
            .dimensions(GetFirstChosenSpatialDim(convolution))
            .window_dilation();
    kernel_spatial_dim_size =
        (kernel_spatial_dim_size - 1) * (window_dilation_factor - 1) +
        kernel_spatial_dim_size;
  }

  std::vector<int64_t> spatial_dimensions_to_split =
      GetChosenSpatialDims(convolution);
  const int64_t spatial_dimension_to_split = spatial_dimensions_to_split[0];

  const int64_t input_dim_size =
      activations->shape().dimensions(spatial_dimension_to_split);

  const int64_t inherent_low_padding =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .padding_low();
  const int64_t inherent_high_padding =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .padding_high();

  const int64_t stride = convolution->window()
                             .dimensions(GetFirstChosenSpatialDim(convolution))
                             .stride();

  const int64_t base_dilation_factor =
      convolution->window()
          .dimensions(GetFirstChosenSpatialDim(convolution))
          .base_dilation();

  bool is_base_dilated = base_dilation_factor > 1;

  const int64_t spatial_size = input_dim_size +
                               (is_base_dilated ? 0 : inherent_low_padding) +
                               inherent_high_padding;

  const int64_t last_overlap = base_dilation_factor == inherent_low_padding
                                   ? kernel_spatial_dim_size
                                   : kernel_spatial_dim_size - 1;
  const int64_t halo_size = is_base_dilated
                                ? last_overlap / base_dilation_factor
                                : kernel_spatial_dim_size - 1;

  const int64_t high_padding_for_base_dilation =
      inherent_low_padding == 0 ? base_dilation_factor - 1
                                : last_overlap % base_dilation_factor;

  const int64_t high_padding_for_conv =
      is_base_dilated ? high_padding_for_base_dilation : 0;

  const int64_t low_padding_for_conv =
      is_base_dilated && (base_dilation_factor != inherent_low_padding)
          ? inherent_low_padding
          : 0;

  return ConvDetails{spatial_dimensions_to_split,
                     inherent_low_padding,
                     inherent_high_padding,
                     stride,
                     spatial_size,
                     base_dilation_factor,
                     halo_size,
                     high_padding_for_conv,
                     low_padding_for_conv,
                     kernel_spatial_dim_size,
                     input_dim_size};
}

absl::Status ConvolutionVisitor::PerformSpaceToBatchOnConvolution(
    HloInstruction* convolution) {
  if (!ConsumeFuel("space-to-batch-converter", [&] {
        return "Skipping space-to-batch propagation because fuel over\n";
      })) {
    return absl::OkStatus();
  }
  VLOG(1) << "Handling conv " << convolution->ToString();

  ConvolutionDimensionNumbers dim_numbers =
      convolution->convolution_dimension_numbers();

  ConvDetails c = GetConvolutionDetails(convolution, dim_numbers);

  int64_t activations_batch_dim = dim_numbers.input_batch_dimension();

  auto activations = convolution->mutable_operand(0);
  VLOG(1) << "spatial size " << c.spatial_size;

  // A very primitive cost model to thwart propagations on tiny shapes.
  if (c.spatial_size < 2 * ctrl_.number_of_splits) {
    return absl::OkStatus();
  }

  auto original_conv = convolution;

  const int64_t output_spatial_dim = dim_numbers.output_spatial_dimensions(
      GetFirstChosenSpatialDim(convolution));
  const int64_t output_offsets =
      convolution->shape().dimensions(output_spatial_dim);
  const int64_t output_offsets_per_split =
      CeilOfRatio(output_offsets, ctrl_.number_of_splits);

  int64_t spatial_split_size =
      CeilOfRatio(output_offsets_per_split, c.base_dilation_factor) * c.stride;
  // Keep increasing the split size so that overall size isn't smaller than the
  // original spatial dimension.
  while (spatial_split_size * ctrl_.number_of_splits - c.spatial_size < 0) {
    spatial_split_size += c.stride;
  }

  auto reduce_window_or_select_and_scatter =
      DoesConvolutionFeedReduceWindowOrSelectAndScatter(convolution);

  if (reduce_window_or_select_and_scatter != nullptr &&
      reduce_window_or_select_and_scatter->shape().IsArray() &&
      reduce_window_or_select_and_scatter->shape().rank() ==
          convolution->shape().rank()) {
    VLOG(2)
        << "DoesConvolutionFeedReduceWindowOrSelectAndScatter returned true";
    // Take into account the stride of the reduce window while choosing the
    // spatial_split_size. This will guarantee propagation through reduce
    // windows.
    const int64_t win_stride =
        std::max(reduce_window_or_select_and_scatter->window()
                     .dimensions(output_spatial_dim)
                     .stride(),
                 static_cast<int64_t>(1));
    CHECK_NE(win_stride, 0)
        << "Bad op " << reduce_window_or_select_and_scatter->ToString();
    CHECK_NE(c.stride, 0) << "Bad op " << convolution->ToString();
    while ((spatial_split_size / c.stride) % win_stride != 0) {
      spatial_split_size += c.stride;
    }
  }

  const int64_t slice_size = spatial_split_size + c.halo_size;

  const int64_t low_pad_to_handle_base_dilation =
      (c.base_dilation_factor > 1 &&
       c.base_dilation_factor == c.inherent_low_padding)
          ? 1
          : 0;

  // Pad spatial dim.
  int64_t pad_size =
      spatial_split_size * ctrl_.number_of_splits - c.spatial_size;

  bool handle_low_pad_in_first_reshape = false;
  if (pad_size > low_pad_to_handle_base_dilation) {
    pad_size -= low_pad_to_handle_base_dilation;
    handle_low_pad_in_first_reshape = true;
  }

  VLOG(1) << "spatial_split_size " << spatial_split_size << " stride "
          << c.stride << " slice_size " << slice_size;
  VLOG(1) << "spatial_dimension_to_split " << c.spatial_dimensions_to_split[0]
          << " num_splits " << ctrl_.number_of_splits
          << " kernel_spatial_dim_size " << c.kernel_spatial_dim_size;
  std::vector<int64_t> spatial_dimensions_to_split =
      c.spatial_dimensions_to_split;
  TF_ASSIGN_OR_RETURN(
      auto retval,
      SplitSpace(
          activations, dim_numbers, activations_batch_dim,
          /*high_padding=*/c.inherent_high_padding + pad_size,
          /*low_padding=*/c.base_dilation_factor == 1 ? c.inherent_low_padding
          : handle_low_pad_in_first_reshape ? low_pad_to_handle_base_dilation
                                            : 0,
          spatial_split_size, ctrl_.number_of_splits,
          &spatial_dimensions_to_split));
  HloInstruction* batch_increased_reshape = retval.first;
  convolution->SetupDerivedInstruction(batch_increased_reshape);

  VLOG(1) << "First reshape done " << batch_increased_reshape->ToString();

  TF_ASSIGN_OR_RETURN(
      activations,
      HaloDuplicateWithSlice(
          batch_increased_reshape, spatial_dimensions_to_split,
          activations_batch_dim,
          /*low_padding=*/
          handle_low_pad_in_first_reshape ? 0 : low_pad_to_handle_base_dilation,
          c.halo_size));

  VLOG(1) << "Batch merge done " << activations->ToString();

  // Now, we rewrite the convolution with a larger batch.

  // Create the new convolution dim numbers.
  auto new_dim_numbers = dim_numbers;

  // We will generate output such that batch is followed by the split spatial
  // dimension.
  const int64_t rank = convolution->shape().rank();
  std::vector<int64_t> transpose_dims(rank);
  int dim_count = 0;
  std::map<int64_t, int64_t> dim_translator;

  for (int j = 0; j < dim_numbers.output_spatial_dimensions_size(); ++j) {
    if (j == GetFirstChosenSpatialDim(convolution)) {
      dim_translator[dim_numbers.output_batch_dimension()] = dim_count;
      new_dim_numbers.set_output_batch_dimension(dim_count++);
    }
    dim_translator[dim_numbers.output_spatial_dimensions(j)] = dim_count;
    new_dim_numbers.set_output_spatial_dimensions(j, dim_count);
    dim_count++;
  }

  dim_translator[dim_numbers.output_feature_dimension()] = dim_count;
  new_dim_numbers.set_output_feature_dimension(dim_count);

  int p = 0;
  for (const auto& entry : dim_translator) {
    transpose_dims[p] = entry.second;
    p++;
  }
  VLOG(1) << "New dim numbers " << new_dim_numbers.DebugString()
          << " batch dim " << new_dim_numbers.input_batch_dimension();
  auto new_window = convolution->window();
  const int64_t first_dim = GetFirstChosenSpatialDim(convolution);
  for (int i = 0; i < ctrl_.count_of_dimensions_to_convert; ++i) {
    new_window.mutable_dimensions(first_dim + i)
        ->set_padding_high(c.high_padding_for_conv);
    new_window.mutable_dimensions(first_dim + i)
        ->set_padding_low(c.low_padding_for_conv);
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_conv,
      MakeConvolveHlo(
          activations, /*rhs=*/convolution->mutable_operand(1),
          convolution->feature_group_count(), convolution->batch_group_count(),
          new_window, new_dim_numbers, convolution->precision_config(),
          /*preferred_element_type=*/convolution->shape().element_type(),
          &convolution->metadata(), &convolution->frontend_attributes()));
  convolution->SetupDerivedInstruction(new_conv);

  // If the activations were to be batch-to-spaced again, simply use the
  // original value.
  batch_to_space_map_[convolution->mutable_operand(0)] =
      convolution->mutable_operand(0);

  VLOG(1) << "Space-to-batched convolution " << new_conv->ToString();

  std::vector<int64_t> new_output_split_spatial_dims(
      ctrl_.count_of_dimensions_to_convert),
      old_output_split_spatial_dims(ctrl_.count_of_dimensions_to_convert);
  for (int i = 0; i < ctrl_.count_of_dimensions_to_convert; ++i) {
    old_output_split_spatial_dims[i] =
        dim_numbers.output_spatial_dimensions(first_dim + i);
    new_output_split_spatial_dims[i] =
        new_dim_numbers.output_spatial_dimensions(first_dim + i);
  }

  const int64_t output_batch_dim = new_dim_numbers.output_batch_dimension();

  auto select_val = computation_->AddInstruction(
      HloInstruction::CreateConstant(
          LiteralUtil::Zero(new_conv->shape().element_type())),
      &convolution->metadata(), &convolution->frontend_attributes());

  TF_ASSIGN_OR_RETURN(
      new_conv,
      SelectValidPortion(new_conv, original_conv, select_val, output_batch_dim,
                         new_output_split_spatial_dims,
                         dim_numbers.output_batch_dimension(),
                         old_output_split_spatial_dims));
  old_to_new_instrs_[original_conv] = new_conv;

  std::vector<int64_t> dim_map(NumMappedDims());
  dim_map[DimMapper(SpaceToBatchDimMap::kBatch)] =
      dim_numbers.output_batch_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kFeature)] =
      dim_numbers.output_feature_dimension();
  dim_map[DimMapper(SpaceToBatchDimMap::kSpace0)] =
      dim_numbers.output_spatial_dimensions(
          GetFirstChosenSpatialDim(convolution));
  instr_to_dim_map_[original_conv] = dim_map;

  instr_to_dim_permute_map_[new_conv] = std::vector<int64_t>(transpose_dims);
  if (non_propagatable_instrs_.count(convolution) > 0) {
    non_propagatable_instrs_.erase(convolution);
  }
  TF_CHECK_OK(PropagateOnUsers(original_conv));

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> SpaceToBatchConverter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(
      2, "SpaceToBatchConverter::Run(), before:\n" + module->ToString());
  bool changed = false;

  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    ConvolutionVisitor visitor(ctrl_, comp);
    if (visitor.Run().value()) {
      changed = true;
    }
    VLOG(1) << "Done operating on computation";
  }
  XLA_VLOG_LINES(2,
                 "SpaceToBatchConverter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
