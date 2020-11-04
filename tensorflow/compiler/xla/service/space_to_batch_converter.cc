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
  static bool Run(int64 limit_on_batch_size, HloComputation* computation);

  // Returns whether any convolution ops were rewritten.
  const bool changed() const { return changed_; }

  ~ConvolutionVisitor() override = default;

 private:
  explicit ConvolutionVisitor(int64 limit_on_batch_size,
                              HloComputation* computation)
      : computation_(computation), limit_on_batch_size_(limit_on_batch_size) {}

  // Current HloComputation instance the ConvolutionVisitor is traversing.
  HloComputation* computation_;

  // Whether rewrite has occurred.
  bool changed_ = false;

  // Limit on batch size to apply this technique on.
  int64 limit_on_batch_size_;
};

bool ConvolutionVisitor::Run(int64 limit_on_batch_size,
                             HloComputation* computation) {
  ConvolutionVisitor visitor(limit_on_batch_size, computation);
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
  // We choose the new batch size to be a constant so that space-to-batch
  // propagation through several convolutional layers is consistent.
  constexpr int64 kNewBatchSize = 8;

  // Batch in batch_group_count has different semantics (it isn't true batch).
  // Consider supporting this case in future if needed.
  if (convolution->batch_group_count() != 1) {
    return Status::OK();
  }

  if (convolution->window().dimensions(kChosenSpatialDim).window_dilation() !=
      1) {
    return Status::OK();
  }

  // TODO(b/168316428): Support base dilations.
  if (convolution->window().dimensions(kChosenSpatialDim).base_dilation() !=
      1) {
    return Status::OK();
  }

  int64 activations_batch_dim = dim_numbers.input_batch_dimension();

  const int64 old_batch_size =
      convolution->operand(0)->shape().dimensions(activations_batch_dim);

  if (old_batch_size > limit_on_batch_size_) {
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

  const int64 num_splits = kNewBatchSize / old_batch_size;

  // We currently only cater to evenly divisible cases.
  if (kNewBatchSize % old_batch_size != 0) {
    return Status::OK();
  }

  // Splitting will be incorrect in these cases.
  if (spatial_size < num_splits ||
      input_dim_size / num_splits < kernel_spatial_dim_size) {
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

  int64 spatial_split_size = output_offsets_per_split * stride;
  // Keep increasing the split size so that overall size isn't smaller than the
  // original spatial dimension.
  while (spatial_split_size * num_splits - spatial_size < 0) {
    spatial_split_size += stride;
  }

  const int64 slice_size =
      spatial_split_size + kernel_spatial_dim_size - stride;

  // Pad spatial dim.
  const int64 pad_size = spatial_split_size * num_splits - spatial_size;

  VLOG(1) << "spatial_split_size " << spatial_split_size << " stride "
          << stride;
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
  // in the spatial dimension, we generate a gather. E.g. if halo size was 2,
  // we'd create a shape of [24] using the gather, and reshape it into [6, 4]
  // (4 being the batch).

  // The benefit of the above mentioned scheme is that it allows for batch
  // growth. Here are some examples of the size increases it causes for a 3x3
  // kernel.
  // with batch=1, [1,16] -> [4,4] ->   [4,6] ->   [1,24] growth of 8.
  // with batch=2, [2,16] -> [8,4] ->   [8,6] ->   [1,48] growth of 16.
  // with batch=3, [3,16] -> [12,4] -> [12,6] -> [1,72] growth of 24.

  std::vector<int64> reshape_dimensions(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());

  reshape_dimensions[spatial_dimension_to_split] = spatial_split_size;
  reshape_dimensions[activations_batch_dim] = num_splits * old_batch_size;

  TF_ASSIGN_OR_RETURN(HloInstruction * batch_increased_reshape,
                      MakeReshapeHlo(reshape_dimensions, activations));
  convolution->SetupDerivedInstruction(batch_increased_reshape);

  VLOG(1) << "First reshape done " << batch_increased_reshape->ToString();

  // Create a gather HLO. We extract slices for given spatial and batch
  // dimensions.
  std::vector<int64> slice_sizes(activations->shape().dimensions().begin(),
                                 activations->shape().dimensions().end());
  slice_sizes[spatial_dimension_to_split] = 1;
  slice_sizes[activations_batch_dim] = 1;

  const int64 rank = activations->shape().dimensions_size();
  std::vector<int64> offset_dims;
  std::vector<int64> collapsed_dims(2);
  int64 collapsed_dim_counter = 0;
  bool seen_collapsed = false;
  for (int j = 0; j < rank; ++j) {
    if (j == activations_batch_dim || j == spatial_dimension_to_split) {
      collapsed_dims[collapsed_dim_counter++] = j;
      seen_collapsed = true;
    } else {
      if (seen_collapsed) {
        offset_dims.push_back(j - 1);
      } else {
        offset_dims.push_back(j);
      }
    }
  }
  std::vector<int64> start_index(2);
  start_index[0] = activations_batch_dim;
  start_index[1] = spatial_dimension_to_split;

  xla::GatherDimensionNumbers gather_dim_numbers =
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/collapsed_dims,
          /*start_index_map=*/start_index,
          /*index_vector_dim=*/1);

  // Create a static index for the gather.
  auto arg_array = absl::make_unique<Array2D<int32>>(
      slice_size * old_batch_size * num_splits, 2);
  auto generate_cell = [&](int64 i, int64 j, int32* value) {
    const int64 row_number = i / (num_splits * old_batch_size);
    if (row_number >= spatial_split_size) {
      if (j == 0) {
        *value = i % (num_splits * old_batch_size) + 1;
        if (num_splits * old_batch_size <=
            i % (num_splits * old_batch_size) + 1) {
          *value = 0;
        }
      } else {
        *value = row_number - spatial_split_size;
      }
    } else {
      if (j == 0) {
        *value = i % (num_splits * old_batch_size);
      } else {
        *value = row_number;
      }
    }
  };

  arg_array->Each(generate_cell);

  auto arg_literal = LiteralUtil::CreateR2FromArray2D<int32>(*arg_array);
  VLOG(1) << " arg_literal " << arg_literal.ToString();
  HloInstruction* index = computation_->AddInstruction(
      HloInstruction::CreateConstant(std::move(arg_literal)));

  VLOG(1) << "slice_size " << slice_size;
  std::vector<int64> gather_output_shape_dims(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());

  gather_output_shape_dims[activations_batch_dim] =
      slice_size * old_batch_size * num_splits;
  gather_output_shape_dims.erase(gather_output_shape_dims.begin() +
                                 spatial_dimension_to_split);

  auto gather_shape = ShapeUtil::MakeShape(activations->shape().element_type(),
                                           gather_output_shape_dims);

  HloInstruction* gather = computation_->AddInstruction(
      HloInstruction::CreateGather(gather_shape, batch_increased_reshape, index,
                                   gather_dim_numbers, slice_sizes, false));

  std::vector<int64> gather_reshape_dimensions(
      activations->shape().dimensions().begin(),
      activations->shape().dimensions().end());

  gather_reshape_dimensions[activations_batch_dim] = slice_size;
  gather_reshape_dimensions[spatial_dimension_to_split] =
      old_batch_size * num_splits;

  // Reshape the gather so that batch is split out.
  TF_ASSIGN_OR_RETURN(activations,
                      MakeReshapeHlo(gather_reshape_dimensions, gather));

  VLOG(1) << "Batch merge done " << activations->ToString();

  // Now, we rewrite the convolution with a larger batch.

  // Set the batch and spatial dimensions for the new convolution.
  new_dim_numbers.set_input_batch_dimension(spatial_dimension_to_split);
  new_dim_numbers.set_input_spatial_dimensions(kChosenSpatialDim,
                                               activations_batch_dim);

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
  for (const auto& entry : dim_map) {
    transpose_dims[p] = entry.second;
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

  const int64 output_split_spatial_dim =
      new_dim_numbers.output_spatial_dimensions(kChosenSpatialDim);
  const int64 output_batch_dim = new_dim_numbers.output_batch_dimension();

  Shape new_shape = new_conv->shape();
  const int64 new_batch_size = new_shape.dimensions(output_batch_dim);
  const int64 new_spatial_dim_size =
      new_shape.dimensions(output_split_spatial_dim);

  CHECK_EQ(new_batch_size % old_batch_size, 0);

  const int64 output_split_batch_size = new_batch_size / old_batch_size;

  std::vector<int64> new_dimensions(new_conv->shape().dimensions().begin(),
                                    new_conv->shape().dimensions().end());
  new_dimensions[output_split_spatial_dim] =
      output_split_batch_size * new_spatial_dim_size;
  new_dimensions[new_dim_numbers.output_batch_dimension()] = old_batch_size;

  // Reshape the output of the new conv into the old convolutions shape.
  TF_ASSIGN_OR_RETURN(HloInstruction * reshape,
                      MakeReshapeHlo(new_dimensions, new_conv));
  convolution->SetupDerivedInstruction(reshape);

  std::vector<int64> start_indices(rank, 0),
      end_indices(new_dimensions.begin(), new_dimensions.end()),
      strides(rank, 1);
  end_indices[output_split_spatial_dim] = convolution->shape().dimensions(
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
    if (ConvolutionVisitor::Run(limit_on_batch_size_, comp)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "ConvolutionSpaceToBatchConverter::Run(), after:\n" +
                        module->ToString());
  return changed;
}

}  // namespace xla
