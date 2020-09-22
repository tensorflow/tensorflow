/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/space_to_batch_converter.h"

#include <map>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace {

// ConvolutionVisitor traverses the HLO computation and rewrites Convolution
// operations with small batch counts into convolutions with larger batch
// counts by moving space to batch.
class ConvolutionVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleConvolution(HloInstruction* convolution) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(HloComputation* computation)
      : computation_(computation) {}

  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  // Whether rewrite has occurred.
  bool changed_ = false;
};

bool ConvolutionVisitor::Run(HloComputation* computation) {
  ConvolutionVisitor visitor(computation);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

Status ConvolutionVisitor::HandleConvolution(HloInstruction* convolution) {
  VLOG(1) << "Handling conv " << convolution->ToString();
  changed_ = false;

  ConvolutionDimensionNumbers dim_numbers =
      convolution->convolution_dimension_numbers();

  // If there are no spatial dims, we return.
  if (dim_numbers.input_spatial_dimensions_size() < 1) {
    return Status::OK();
  }

  // This is the spatial dimension we choose to spilt.
  constexpr int64 kChosenSpatialDim = 0;
  constexpr int64 kLowLimitForSplitCount = 4;
  constexpr int64 kHighLimitForSplitCount = 24;

  if (convolution->window().dimensions(kChosenSpatialDim).window_dilation() !=
      1) {
    return Status::OK();
  }

  if (convolution->window().dimensions(kChosenSpatialDim).base_dilation() !=
      1) {
    return Status::OK();
  }

  if (convolution->window().dimensions(kChosenSpatialDim).window_reversal()) {
    return Status::OK();
  }

  int64 activations_batch_dim = dim_numbers.input_batch_dimension();

  const int64 old_batch_size =
      convolution->operand(0)->shape().dimensions(activations_batch_dim);

  // TODO(b/168316428): Only doing this for batch 1 currently. Extend later.
  if (old_batch_size != 1) {
    return Status::OK();
  }

  auto kernel = convolution->mutable_operand(1);
  const auto& kernel_shape = kernel->shape();
  const int64 kernel_spatial_dim_size = kernel_shape.dimensions(
      dim_numbers.kernel_spatial_dimensions(kChosenSpatialDim));

  auto activations = convolution->mutable_operand(0);

  int64 spatial_dimension_to_split =
      dim_numbers.input_spatial_dimensions(kChosenSpatialDim);

  const int64 input_dim_size = activations->shape().dimensions(
      dim_numbers.input_spatial_dimensions(kChosenSpatialDim));

  const int64 inherent_low_padding =
      convolution->window().dimensions(kChosenSpatialDim).padding_low();
  const int64 inherent_high_padding =
      convolution->window().dimensions(kChosenSpatialDim).padding_high();
  const bool inherent_padding_needed =
      inherent_low_padding != 0 || inherent_high_padding != 0;

  const int64 stride =
      convolution->window().dimensions(kChosenSpatialDim).stride();

  const int64 spatial_size =
      input_dim_size + inherent_low_padding + inherent_high_padding;
  VLOG(1) << "spatial size " << spatial_size;

  int64 min_pad_size = INT64_MAX;
  int64 num_splits;
  // Explore several splitting points; choose one that requires least padding.
  // This padding is done so that we can evenly reshape.
  for (int64 j = kHighLimitForSplitCount; j >= kLowLimitForSplitCount; j--) {
    if (input_dim_size / j < kernel_spatial_dim_size) {
      continue;
    }

    if (spatial_size < j) {
      continue;
    }

    const int64 output_offsets = convolution->shape().dimensions(
        dim_numbers.output_spatial_dimensions(kChosenSpatialDim));
    const int64 output_offsets_per_split = CeilOfRatio(output_offsets, j);

    const int64 spatial_split_size = output_offsets_per_split * stride;

    // Pad spatial dim
    const int64 pad_size = spatial_split_size * j - spatial_size;
    if (pad_size >= 0 && pad_size < min_pad_size) {
      min_pad_size = pad_size;
      num_splits = j;
    }
  }

  // No suitable split found.
  if (min_pad_size == INT64_MAX) {
    return Status::OK();
  }

  // By now, we are certain that the space-to-batch transormation is going to
  // take place.

  // Create the new convolution dim numbers.
  auto new_dim_numbers = dim_numbers;

  // We'd need transposition of activations here such that batch and space dim
  // that is being split are adjacent (in that order).
  if (spatial_dimension_to_split != activations_batch_dim + 1) {
    int64 pushed_counter = 0;
    std::vector<int64> transpose_dims;
    int64 new_batch_dim, new_spatial_dim;
    for (int i = 0; i < activations->shape().rank(); ++i) {
      if (i == activations_batch_dim) {
        continue;
      }
      if (i == spatial_dimension_to_split) {
        new_dim_numbers.set_input_batch_dimension(pushed_counter);
        transpose_dims.push_back(activations_batch_dim);
        new_batch_dim = pushed_counter;
        pushed_counter++;
        new_spatial_dim = pushed_counter;
      }

      if (i == dim_numbers.input_feature_dimension()) {
        new_dim_numbers.set_input_feature_dimension(pushed_counter);
      } else {
        for (int j = 0; j < dim_numbers.input_spatial_dimensions_size(); ++j) {
          if (i == dim_numbers.input_spatial_dimensions(j)) {
            new_dim_numbers.set_input_spatial_dimensions(j, pushed_counter);
            break;
          }
        }
      }
      transpose_dims.push_back(i);
      pushed_counter++;
    }

    activations_batch_dim = new_batch_dim;
    spatial_dimension_to_split = new_spatial_dim;
    TF_ASSIGN_OR_RETURN(activations,
                        MakeTransposeHlo(activations, transpose_dims));
  }

  const int64 output_offsets = convolution->shape().dimensions(
      dim_numbers.output_spatial_dimensions(kChosenSpatialDim));
  const int64 output_offsets_per_split =
      CeilOfRatio(output_offsets, num_splits);

  const int64 spatial_split_size = output_offsets_per_split * stride;
  const int64 slice_size =
      (output_offsets_per_split - 1) * stride + kernel_spatial_dim_size;

  VLOG(1) << "spatial_split_size " << spatial_split_size << " stride "
          << stride;

  // Pad spatial dim.
  const int64 pad_size = spatial_split_size * num_splits - spatial_size;

  VLOG(1) << "spatial_dimension_to_split " << spatial_dimension_to_split
          << " num_splits " << num_splits << " kernel_spatial_dim_size "
          << kernel_spatial_dim_size;

  // Because we are splitting the spatial dimension, if convolution needed
  // padding in the spatial dimension, we materialize it.
  if (pad_size != 0 || inherent_padding_needed) {
    PaddingConfig padding_config =
        MakeNoPaddingConfig(activations->shape().dimensions_size());
    padding_config.mutable_dimensions(spatial_dimension_to_split)
        ->set_edge_padding_high(inherent_high_padding + pad_size);
    padding_config.mutable_dimensions(spatial_dimension_to_split)
        ->set_edge_padding_low(inherent_low_padding);
    HloInstruction* padding =
        computation_->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::Zero(activations->shape().element_type())));
    TF_ASSIGN_OR_RETURN(activations,
                        MakePadHlo(activations, padding, padding_config));
  }
  VLOG(1) << "Initial padded activations shape "
          << activations->shape().ToString();

  // Now we reorganize the activations. E.g. if the shape [B, SPACE] was [1, 16]
  // and 4 splits were needed, we first create [4, 4]. Next, to deal with halo
  // in the spatial dimension, we first pad that dimension. E.g. if halo size
  // was 2, we'd create a shape of [4, 6]. We then flatten the shape such that
  // A = [1, 24]. Now, we rotate the flattened 24 dimension left by 2 (with
  // -2 low padding and +2 high padding) to create shape B. Then, we select
  // between A and B such that halo regions are placed into A at the right
  // locations.
  std::vector<int64> reshape_dimensions(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());
  reshape_dimensions[spatial_dimension_to_split] = spatial_split_size;
  reshape_dimensions[activations_batch_dim] = num_splits;

  TF_ASSIGN_OR_RETURN(HloInstruction * batch_increased_reshape,
                      MakeReshapeHlo(reshape_dimensions, activations));
  convolution->SetupDerivedInstruction(batch_increased_reshape);

  VLOG(1) << "First reshape done " << batch_increased_reshape->ToString();

  PaddingConfig padding_config =
      MakeNoPaddingConfig(batch_increased_reshape->shape().dimensions_size());
  padding_config.mutable_dimensions(spatial_dimension_to_split)
      ->set_edge_padding_high(slice_size - spatial_split_size);
  HloInstruction* padding =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(batch_increased_reshape->shape().element_type())));
  TF_ASSIGN_OR_RETURN(
      HloInstruction * pad_applied,
      MakePadHlo(batch_increased_reshape, padding, padding_config));

  VLOG(1) << "Padding done " << pad_applied->ToString();

  auto straightened_activations_dims = reshape_dimensions;
  straightened_activations_dims[spatial_dimension_to_split] =
      num_splits * slice_size;
  straightened_activations_dims[activations_batch_dim] = old_batch_size;

  VLOG(1) << "slice_size " << slice_size;
  TF_ASSIGN_OR_RETURN(
      HloInstruction * straightened_activations,
      MakeReshapeHlo(straightened_activations_dims, pad_applied));

  VLOG(1) << "Straightening done";

  PaddingConfig rotation_padding_config =
      MakeNoPaddingConfig(straightened_activations->shape().dimensions_size());
  rotation_padding_config.mutable_dimensions(spatial_dimension_to_split)
      ->set_edge_padding_high(slice_size - spatial_split_size);
  rotation_padding_config.mutable_dimensions(spatial_dimension_to_split)
      ->set_edge_padding_low(spatial_split_size - slice_size);
  HloInstruction* rotation_padding =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(straightened_activations->shape().element_type())));
  TF_ASSIGN_OR_RETURN(HloInstruction * rotated_activations,
                      MakePadHlo(straightened_activations, rotation_padding,
                                 rotation_padding_config));
  convolution->SetupDerivedInstruction(rotated_activations);

  // Build a constant PRED to decide which elements in the split dimension
  // are from halo.
  tensorflow::core::Bitmap b(num_splits * slice_size);
  for (int k = 0; k < num_splits * slice_size; ++k) {
    if (k % slice_size < spatial_split_size) {
      b.set(k);
    } else {
      b.clear(k);
    }
  }

  auto arg_literal = LiteralUtil::CreateR1(b);
  HloInstruction* slice_mask = computation_->AddInstruction(
      HloInstruction::CreateConstant(std::move(arg_literal)));

  // Broadcast the mask in all dimensions of the activations.
  HloInstruction* shape_mask =
      MakeBroadcastHlo(slice_mask, {spatial_dimension_to_split},
                       straightened_activations->shape().dimensions());

  VLOG(1) << "Shape mask made " << shape_mask->ToString();

  TF_ASSIGN_OR_RETURN(HloInstruction * select,
                      MakeSelectHlo(shape_mask, straightened_activations,
                                    rotated_activations, convolution));
  VLOG(1) << "Select generated";

  // Increase batch size for one last time.
  TF_ASSIGN_OR_RETURN(
      activations, MakeReshapeHlo(pad_applied->shape().dimensions(), select));

  // Now, we rewrite the convolution with a larger batch.
  const auto& activations_shape = activations->shape();
  const int64 rank = activations_shape.dimensions_size();

  // We will generate output such that batch is followed by the split spatial
  // dimension.
  std::vector<int64> transpose_dims(convolution->shape().rank());
  int dim_count = 0;
  std::map<int64, int64> dim_map;

  for (int j = 0; j < dim_numbers.output_spatial_dimensions_size(); ++j) {
    if (j == kChosenSpatialDim) {
      dim_map[dim_numbers.output_batch_dimension()] = dim_count;
      new_dim_numbers.set_output_batch_dimension(dim_count++);
    }
    dim_map[dim_numbers.output_spatial_dimensions(j)] = dim_count;
    new_dim_numbers.set_output_spatial_dimensions(j, dim_count);
    dim_count++;
  }

  dim_map[dim_numbers.output_feature_dimension()] = dim_count;
  new_dim_numbers.set_output_feature_dimension(dim_count);

  int p = 0;
  for (auto [k, v] : dim_map) {
    transpose_dims[p] = v;
    p++;
  }

  auto new_window = convolution->window();
  new_window.mutable_dimensions(kChosenSpatialDim)->set_padding_high(0);
  new_window.mutable_dimensions(kChosenSpatialDim)->set_padding_low(0);
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_conv,
      MakeConvolveHlo(activations, /*rhs=*/convolution->mutable_operand(1),
                      convolution->feature_group_count(),
                      convolution->batch_group_count(), new_window,
                      new_dim_numbers, convolution->precision_config()));
  convolution->SetupDerivedInstruction(new_conv);

  VLOG(1) << "new_conv " << new_conv->ToString();

  Shape new_shape = new_conv->shape();
  const int64 new_batch_size =
      new_shape.dimensions(new_dim_numbers.output_batch_dimension());
  const int64 new_spatial_dim_size = new_shape.dimensions(
      new_dim_numbers.output_spatial_dimensions(kChosenSpatialDim));
  new_shape.set_dimensions(
      new_dim_numbers.output_spatial_dimensions(kChosenSpatialDim),
      new_batch_size * new_spatial_dim_size);
  new_shape.set_dimensions(new_dim_numbers.output_batch_dimension(),
                           old_batch_size);

  // Reshape the output of the new conv into the old convolutions shape.
  TF_ASSIGN_OR_RETURN(HloInstruction * reshape,
                      MakeReshapeHlo(new_shape, new_conv));
  convolution->SetupDerivedInstruction(reshape);

  std::vector<int64> start_indices(rank, 0),
      end_indices(new_shape.dimensions().begin(), new_shape.dimensions().end()),
      strides(rank, 1);
  end_indices[new_dim_numbers.output_spatial_dimensions(kChosenSpatialDim)] =
      convolution->shape().dimensions(
          dim_numbers.output_spatial_dimensions(kChosenSpatialDim));

  // This slicing is getting rid of the padding we added to evenly divide space.
  TF_ASSIGN_OR_RETURN(
      HloInstruction * output_slice,
      MakeSliceHlo(reshape, start_indices, end_indices, strides));
  convolution->SetupDerivedInstruction(output_slice);

  TF_ASSIGN_OR_RETURN(HloInstruction * output_transpose,
                      MakeTransposeHlo(output_slice, transpose_dims));
  convolution->SetupDerivedInstruction(output_transpose);

  VLOG(1) << "output_transpose " << output_transpose->ToString();

  changed_ = true;
  return computation_->ReplaceInstruction(convolution, output_transpose);
}

}  // namespace

StatusOr<bool> ConvolutionSpaceToBatchConverter::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "ConvolutionSpaceToBatchConverter::Run(), before:\n" +
                        module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (ConvolutionVisitor::Run(comp)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "ConvolutionSpaceToBatchConverter::Run(), after:\n" +
                        module->ToString());
  return changed;
}

}  // namespace xla
