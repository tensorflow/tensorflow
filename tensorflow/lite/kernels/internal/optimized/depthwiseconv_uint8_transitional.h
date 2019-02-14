/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_

// This file provides kernel implementations that are not used in shipped
// inference code, but rather (a) show how model C++ code is designed and then
// transformed into asm code, and (b) aid with maintenance and later development
// of variations. Many projects (even including, say, the classic NAG libraries)
// develop highly optimized code, but do not maintain intermediate versions.
// Often the result is incomprehensible final-version code.

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/optimized/depthwiseconv_uint8_3x3_filter.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace optimized_ops {
namespace depthwise_conv {

// Permute filter data, and adjust bias data to account for symmetric input
// offset. Details are provided in the implementation of the
// kUseCModel3x3DotProduct version.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation>
struct ProcessPerDepth {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

template <>
struct ProcessPerDepth<DepthwiseConvImplementation::kUseCModel3x3DotProduct> {};

// Copy a macro block of data from the input buffer into the workspace,
// permuting data within each micro block.
//
// (a) Copy a macro block of data, padding as required along the width and
//     height.
// (b) Transpose the data within each micro block.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          DepthwiseConvDepthMultiplication depth_multiplication,
          int32 max_padding>
struct PackMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// TODO(b/118877434) Placeholder, to be implemented in subsequent CL.
template <int32 max_padding>
struct PackMacroBlock<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      max_padding> {
  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK(false);
    return;
  }
};

// TODO(b/118877434) Placeholder, to be implemented in subsequent CL.
template <int32 max_padding>
struct PackMacroBlock<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                      DepthwiseConvDepthMultiplication::kUnitInputDepth,
                      max_padding> {
  static inline void Run(int32 height_block_number, int32 width_block_number,
                         const uint8* input_block_data,
                         int8* scratch_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK(false);
    return;
  }
};

// Apply filter to macro block of input data and store results. Details are
// provided in the implementation of the kUseCModel3x3DotProduct version.
//
// Parameters for repeats and residual sizes are in terms of outputs.
//
// See the comments preceding DepthwiseConvDotProduct3x3() for further notes.
template <DepthwiseConvImplementation implementation,
          DepthwiseConvDepthMultiplication depth_multiplication, int32 stride>
struct KernelMacroBlock {
  // Routine is contained in a static Run() method. No default template version
  // is supplied, so that all implementations are deliberate choices of template
  // specialization.
  //
  // Note that the signature of the Run() method will be designed for the asm
  // implementation rather than conforming to style.
};

// TODO(b/118877434) Placeholder, to be implemented in subsequent CL.
template <int32 stride>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kNoMultiplication,
                        stride> {
  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK(false);
    return;
  }
};

// TODO(b/118877434) Placeholder, to be implemented in subsequent CL.
template <int32 stride>
struct KernelMacroBlock<DepthwiseConvImplementation::kUseCModel3x3DotProduct,
                        DepthwiseConvDepthMultiplication::kUnitInputDepth,
                        stride> {
  static inline void Run(const int8* scratch_block_data,
                         const int8* filter_workspace, const int32* bias_data,
                         uint8* output_block_data,
                         const DepthwiseConvDotProdParams* function_params) {
    TFLITE_DCHECK(false);
    return;
  }
};

// Top-level implementation function for 3x3 depthwise convolution using
// NEON dot-product instructions.
//
// MACRO & MICRO BLOCKS
//
// The task is divided into macro blocks. Data is copied first into a macro
// block in a workspace. This has two purposes: (a) bringing data into
// cache, and (b) permuting data so that it can be used much more easily in
// a dot-product filter.
//
// When there is no depth multiplication:
//
// The permutations required for dot-products are local, within 4 data points
// down the depth and 4 across the width. We want to pull in input data at least
// 8-bytes at a time, down the depth, and so we divide the macro blocks into
// 1x4x8 (height, width, depth) and further divide the micro blocks into
// sub-blocks with shape (1x4x4).
//
// Each macro-block is constructed from micro-blocks that are internally
// rearranged during loading into the macro-block workspace.
//
// In other words, the micro-block shape is
//     {1, 1, 4, 8}
// Each macro block is typically shape
//     {1, height_block_size, 4 * workspace_width_micro_repeats, 64}
// and workspace_width_micro_repeats is chosen so it fits into the
// workspace.
//
// However, if depth < 64, we decrease the macro block depth, enabling us to
// increase the macro-block width.
//
// When there is depth multiplication:
//
// We require input-depth = 1 and exploit that instead.  Note that output data
// is still full-depth, *as is the filter and bias data after certain
// adjustments*, and so the filter stage in this case still proceeds in
// terms of sub-blocks.
//
// The Magic of these numbers:
//     4 is the number of input elements used in each dot-product.
//     8 is the number of inputs we load at a time into a register.
//     64 is min amount of data to be loaded in a stretch (when possible).
//
// FILTER DATA PREPARATION
//
// Filter data needs to be permuted in a fashion like that of input data, and
// this is done in a preprocessing stage. In addition, this stage extends the
// filter in the direction of width from 3 to 4. The extra filter taps are set
// to zero so that input data does not have to be zeroed before applying
// dot-products.
//
// OVERALL COUNTS: HANDLING TRAILING ITERATION
//
// Often it is necessary to handle the last iteration in a loop differently,
// generally because the final item is shorter. The logic to detect the
// special case can be a bit expensive. We use a scheme in which there are
// two counts, in a pattern like xxx_yyy_repeats and
// xxx_overall_yyy_repeats. The first gives the count of "normal"
// iterations. The loop iterates over the second count, and the induction
// variable is checked to see if it reaches xxx_yyy_repeats. If there is no
// special trailing iteration, xxx_yyy_repeats = xxx_overall_yyy_repeats,
// and the special code is not executed.
//
// Example:
// Suppose that we characterize a size s as
// f(s) -> (block-4-repetitions, remainder, overall_repetitions):
// f(11) -> (2, 3, 3)
// f(12) -> (3, 0, 3)
// f(13) -> (3, 1, 4)
//
// POINTING OUTSIDE OF INPUT ARRAY.
//
// When there is padding, the input data pointer passed to the fill routines
// points outside of the input array and into a kind-of virtual padded
// margin. It turns out that this simplifies the code and removes
// conditional statements. It is hard to explain why without comparing two
// versions of the code. In summary, this way the adjustment into the margin
// can be made unconditionally, and the correction back into the input array
// is done where there is a conditional already.
//
// OVERLAP
//
// Since this is *depthwise* conv, neither the batch nor the depth have overlap.
// The height and depth overlap by (filter_size - 1). Thus some data is used
// twice on the borders of macro blocks.
//
template <DepthwiseConvImplementation implementation>
inline void DepthwiseConvDotProduct3x3(
    const DepthwiseParams& params, const RuntimeShape& input_shape,
    const uint8* input_data, const RuntimeShape& filter_shape,
    const uint8* filter_data, const RuntimeShape& bias_shape,
    const int32* bias_data, const RuntimeShape& output_shape,
    uint8* output_data) {
  // Check kernel restrictions.
  constexpr int filter_size = 3;
  constexpr int kSymmetricZeroPoint = 128;
  constexpr int kMaxStride = 2;
  constexpr int kMaxPadding = 1;
  TFLITE_DCHECK_EQ(params.weights_offset, -kSymmetricZeroPoint);
  TFLITE_DCHECK_LE(params.stride_width, kMaxStride);
  TFLITE_DCHECK_EQ(params.stride_height, params.stride_width);
  TFLITE_DCHECK_EQ(params.dilation_width_factor, 1);
  TFLITE_DCHECK_EQ(params.dilation_height_factor, 1);
  TFLITE_DCHECK_LE(params.padding_values.width, kMaxPadding);
  TFLITE_DCHECK_LE(params.padding_values.height, kMaxPadding);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  // Key kernel parameters (along with padding handled later).
  const int stride = params.stride_width;
  const int depth_multiplier = params.depth_multiplier;
  const bool has_depth_multiplication = depth_multiplier > 1;

  // Extract task dimensions.
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  TFLITE_DCHECK(!has_depth_multiplication || input_depth == 1);
  TFLITE_DCHECK(has_depth_multiplication || input_depth == output_depth);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  TFLITE_DCHECK_EQ(input_depth * depth_multiplier, output_depth);
  TFLITE_DCHECK_EQ(MatchingDim(filter_shape, 1, filter_shape, 2), filter_size);

  // Return now if nothing to do.
  if (output_width == 0 || output_height == 0) {
    return;
  }

  // Kernel parameter structure: set basic fields.
  //
  // In asm it is easier to pass a structure than more than, say, 8 parameters.
  DepthwiseConvDotProdParams function_params;
  function_params.input_depth = input_depth;
  function_params.output_depth = output_depth;
  function_params.input_offset = params.input_offset;
  function_params.output_offset = params.output_offset;
  function_params.output_multiplier = params.output_multiplier;
  function_params.output_shift = params.output_shift;
  function_params.quantized_activation_min = params.quantized_activation_min;
  function_params.quantized_activation_max = params.quantized_activation_max;
  function_params.stride = stride;

  // Handle inbound bias data.
  //
  // Note that this data is adjusted in a per-depth process before the main
  // filters. The adjustment accounts for a non-symmetric input offset.
  //
  // Kernel subroutines need to be able to operate consistently on an bias
  // array. Where there is no bias, we provide one filled with zeros.
  constexpr int kMinBiasLoad = 8;
  int32 zero_bias_data[kMinBiasLoad];
  if (bias_data) {
    function_params.bias_increment = 4;
  } else {
    memset(zero_bias_data, 0, sizeof(zero_bias_data));
    bias_data = &zero_bias_data[0];
    function_params.bias_increment = 0;
  }
  TFLITE_DCHECK_LE(2 * function_params.bias_increment, kMinBiasLoad);

  // Process padding.
  //
  // Whether "correct" or not, this matches ComputeConvSizes. When there is
  // stride > 1 there can be padding on the bottom or top, and therefore
  // we need to consider padding. This is true even if one or other of the
  // padding_values is 0.
  const int padded_width = (output_width - 1) * stride + filter_size;
  {
    const int padding_left = params.padding_values.width;
    // Right padding would be -1 if discarding input because of stride.
    const int padding_right =
        std::max(padded_width - input_width - padding_left, 0);
    const int padding_top = params.padding_values.height;
    const int padded_height = (output_height - 1) * stride + filter_size;
    const int padding_bottom =
        std::max(padded_height - input_height - padding_top, 0);

    function_params.padding_left = padding_left;
    function_params.padding_right = padding_right;
    function_params.padding_top = padding_top;
    function_params.padding_bottom = padding_bottom;

    TFLITE_DCHECK_LE(padding_left, padding_right);
    TFLITE_DCHECK_LE(padding_top, padding_bottom);
  }
  // When stride == 1 left or top padding may only be non-zero.
  // This is when padding is specified but not needed on a trailing dimension.
  // When stride == 2 right or bottom padding may only be non-zero.
  // This is a result of the details of the padding calculations.
  const bool padding_required =
      params.padding_type == tflite::PaddingType::kSame ||
      function_params.padding_right > 0 || function_params.padding_bottom > 0;

  // Choose parameter-specific kernel subroutines.
  //
  // The main part of the kernel has two stages. First, a temporary workspace is
  // filled with padded and permuted data. Second, the filter is applied to the
  // workspace data to generate output.
  //
  // The workspace fill stage handles padding so that the filter stage does not
  // need to account for it. The workspace fill stage does not need to
  // understand striding, and implicitly handles striding through the parameters
  // that it is given.
  using pack_macro_block_func_t = decltype(
      &PackMacroBlock<implementation,
                      DepthwiseConvDepthMultiplication::kNoMultiplication,
                      0>::Run);
  using kernel_macro_block_func_t = decltype(
      &KernelMacroBlock<implementation,
                        DepthwiseConvDepthMultiplication::kNoMultiplication,
                        1>::Run);
  pack_macro_block_func_t pack_macro_block_func;
  kernel_macro_block_func_t kernel_macro_block_func;
  {
    if (has_depth_multiplication) {
      if (padding_required) {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kUnitInputDepth,
                           /*max_padding=*/1>::Run;
      } else {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kUnitInputDepth,
                           /*max_padding=*/0>::Run;
      }
      if (stride == 1) {
        kernel_macro_block_func =
            KernelMacroBlock<implementation,
                             DepthwiseConvDepthMultiplication::kUnitInputDepth,
                             /*stride=*/1>::Run;
      } else {
        kernel_macro_block_func =
            KernelMacroBlock<implementation,
                             DepthwiseConvDepthMultiplication::kUnitInputDepth,
                             /*stride=*/2>::Run;
      }
    } else {
      if (padding_required) {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kNoMultiplication,
                           /*max_padding=*/1>::Run;
      } else {
        pack_macro_block_func =
            PackMacroBlock<implementation,
                           DepthwiseConvDepthMultiplication::kNoMultiplication,
                           /*max_padding=*/0>::Run;
      }
      if (stride == 1) {
        kernel_macro_block_func = KernelMacroBlock<
            implementation, DepthwiseConvDepthMultiplication::kNoMultiplication,
            /*stride=*/1>::Run;
      } else {
        kernel_macro_block_func = KernelMacroBlock<
            implementation, DepthwiseConvDepthMultiplication::kNoMultiplication,
            /*stride=*/2>::Run;
      }
    }
  }

  // Stride-only variables.
  //
  // stride == 1 ? 4 : 2:
  const int output_height_per_macro = 6 - 2 * stride;
  // output_height_per_macro * stride:
  constexpr int input_height_per_macro = 4;
  // Number of rows per micro block (= rows per macro block) is
  //   (output_height_per_macro - 1) * stride + 1 + (filter_size - 1)
  //   = stride == 1 ? 3 + filter_size : 2 + filter_size:
  const int height_block_size = 4 + filter_size - stride;
  const int input_height_overlap = filter_size - stride;
  // stride == 1 ? 4 : 2:
  function_params.four_over_stride = output_height_per_macro;

  TFLITE_DCHECK_EQ(stride * function_params.four_over_stride, 4);
  TFLITE_DCHECK_EQ(height_block_size,
                   input_height_per_macro + input_height_overlap);

  // Create workspaces.
  //
  // Filter workspace is for shuffle: only first depth/8 is used.
  // indexed as [depth/8][sub-block][height][depth][width].
  TFLITE_DCHECK_LE(output_depth, kDepthwiseConvAdjustedBiasLimit);
  TFLITE_DCHECK_EQ(kDepthwiseConvAdjustedBiasLimit % 8, 0);
  int8 macroblock_workspace[kDepthwiseConvScratchWorkspaceSize];
  int32 adjusted_bias_data[kDepthwiseConvAdjustedBiasLimit];
  int8 filter_workspace[kDepthwiseConvAdjustedBiasLimit >> 3][3][2][4][4];

  // Output depth characterization.
  //
  const int depth_macro_count = output_depth / 64;
  const int depth_overall_macro_count = (output_depth + 63) / 64;
  // Number of micro blocks down the depth in a final incomplete macro block.
  const int depth_trailing_micro_repeats = output_depth / 8 % 8;
  // The output_depth may not have a remainder: it must be a multiple of 8.
  TFLITE_DCHECK_EQ(output_depth,
                   64 * depth_macro_count + 8 * depth_trailing_micro_repeats);

  // Characterize the first macro block depth, the largest.
  //
  // We base treatment of the width on the trailing macro block if there are
  // no full blocks, in order to do more work together (that is, increase
  // workspace_width_micro_repeats when largest_macro_depth < 64).
  const int largest_macro_depth =
      has_depth_multiplication
          ? 1
          : (depth_macro_count > 0 ? 64 : 8 * depth_trailing_micro_repeats);

  // Characterize width, consumption of input and generation of output.
  //
  // In the case of depth multiplication, we ensure that some of the workspace
  // at the end remains unused. This enables the filter routines to load the
  // "next" data, of at least 16 bytes, even when at the end of the workspace.
  // It is relatively expensive to detect the end micro block. It is also very
  // difficult to test for (to trigger) erroneous reads (past end of array) in
  // the depth multplication case.
  int workspace_width_micro_repeats =
      (has_depth_multiplication ? kDepthwiseConvScratchWorkspaceSize - 16
                                : kDepthwiseConvScratchWorkspaceSize) /
      (4 * largest_macro_depth * height_block_size);
  // When there is no depth multiplication, the workspace depth is a multiple of
  // 8, which ensures that workspace rows are 16-byte aligned. (Actually 32,
  // because of the micro width of 4.) This is not necessarily the case under
  // depth multiplication, so we adjust now to impose this restriction.
  if (has_depth_multiplication) {
    workspace_width_micro_repeats = (workspace_width_micro_repeats / 4) * 4;
  }
  TFLITE_DCHECK_EQ((workspace_width_micro_repeats * largest_macro_depth) % 4,
                   0);
  // Discount 1 of the micro-block repeats in each macro block to account for
  // overlap.
  const int consumed_width_per_macro_block =
      4 * (workspace_width_micro_repeats - 1);
  const int output_width_per_macro_block =
      function_params.four_over_stride * (workspace_width_micro_repeats - 1);
  TFLITE_DCHECK_GT(workspace_width_micro_repeats, 1);
  TFLITE_DCHECK_EQ(output_width_per_macro_block * stride,
                   consumed_width_per_macro_block);

  // Width repetitions and residuals.
  //
  // Use of the workspace is characterized primarily in terms of *padded input*.
  // Striding only matters in a few places.
  //
  // Simplifications: We require that there always be at least one full
  // micro-block across the width. Since the maximum padding is 1, the trailing
  // padding cannot span two micro blocks.
  const int residual_micro_width = padded_width % 4;
  // We base the count of macro blocks on the amount of padded input data each
  // one consumes.
  int width_overall_macro_count = (padded_width - residual_micro_width +
                                   consumed_width_per_macro_block - 1) /
                                  consumed_width_per_macro_block;
  // Recall that we left a micro block at the end of each macro block for use as
  // overlap. There is a special case in which we can use one fewer macro
  // blocks, with the last one consuming extra input. (But not if the
  // calculation thinks that we can use zero blocks.)
  if (padded_width <=
      ((width_overall_macro_count - 1) * consumed_width_per_macro_block + 4)) {
    width_overall_macro_count -= 1;
  }
  width_overall_macro_count = std::max(width_overall_macro_count, 1);
  // We always have to treat the final macro block along width as trailing,
  // because even if it is full in terms of padded input, it will be incomplete
  // in terms of output.
  const int width_macro_count = width_overall_macro_count - 1;
  // Micro blocks are traversed in terms of input in fill routines.
  const int width_trailing_micro_repeats =
      (padded_width - consumed_width_per_macro_block * width_macro_count) / 4;
  const int width_overall_trailing_micro_repeats =
      (padded_width - consumed_width_per_macro_block * width_macro_count + 3) /
      4;
  // Micro blocks are traversed in terms of output in filtering routines.
  const int residual_output_micro_width =
      (output_width - 1) % function_params.four_over_stride + 1;
  const int output_width_trailing_micro_repeats =
      residual_micro_width > (filter_size - 1)
          ? width_trailing_micro_repeats
          : width_trailing_micro_repeats - 1;
  // Check results.
  TFLITE_DCHECK_GT(width_overall_trailing_micro_repeats, 0);
  TFLITE_DCHECK_EQ(padded_width,
                   residual_micro_width +
                       consumed_width_per_macro_block * width_macro_count +
                       4 * width_trailing_micro_repeats);
  TFLITE_DCHECK_LE(width_overall_macro_count, width_macro_count + 1);
  TFLITE_DCHECK_GE(width_overall_macro_count, width_macro_count);

  // Height repetitions and residuals.
  //
  const int height_macro_count = output_height / output_height_per_macro;
  const int residual_output_height = output_height % output_height_per_macro;
  const int height_overall_macro_count =
      (output_height + output_height_per_macro - 1) / output_height_per_macro;
  TFLITE_DCHECK_EQ(
      output_height,
      residual_output_height + output_height_per_macro * height_macro_count);
  TFLITE_DCHECK_LE(height_overall_macro_count, height_macro_count + 1);
  TFLITE_DCHECK_GE(height_overall_macro_count, height_macro_count);

  // Data strides.
  //
  const int input_height_stride = input_width * input_depth;
  const int output_height_stride = output_width * output_depth;
  const int input_batch_stride = input_height_stride * input_height;
  const int output_batch_stride = output_height_stride * output_height;
  const int input_depth_macro_stride = has_depth_multiplication ? 0 : 64;
  const int input_width_macro_stride =
      input_depth * consumed_width_per_macro_block;
  const int output_width_macro_stride =
      output_depth * output_width_per_macro_block;

  // Store parameters that do not vary across macro blocks.
  //
  function_params.workspace_width_micro_repeats = workspace_width_micro_repeats;
  function_params.height_macro_count = height_overall_macro_count;
  function_params.width_macro_count = width_overall_macro_count;
  function_params.input_height_stride = input_height_stride;
  function_params.output_height_stride = output_height_stride;
  function_params.residual_width = residual_micro_width;

  // Preprocess filter and bias data.
  //
  ProcessPerDepth<implementation>::Run(filter_data, bias_data,
                                       filter_workspace[0][0][0][0],
                                       adjusted_bias_data, &function_params);
  function_params.bias_increment = 4;  // Adjusted bias data always spans depth.

  // Main process.
  //
  // Most kernels are nested batch-height-width-depth. Here we proceed over
  // macro blocks batch-width-depth-height.
  //
  // Example of handling of trailing iteration: when there is trailing depth,
  // depth_overall_macro_count = depth_macro_count + 1, so we can adjust the
  // dimensions for trailing macro blocks by looking for
  // j_depth == depth_macro_count.
  for (int b = 0; b < batches; ++b) {
    for (int k_width = 0; k_width < width_overall_macro_count; ++k_width) {
      // Figure out the work to be done for this macro block. If it trails in
      // any dimension, the work in that dimension is adjusted.
      // The work to be done across widths has 3 cases:
      // (a) A full macro block,
      // (b) Partial terminal macro block, with input and output ending in
      //     same micro block, and
      // (c) Partial terminal macro block, with output corresponding to one
      //     fewer micro blocks, because filter extends across micro-block
      //     boundary.
      if (k_width != width_macro_count) {
        function_params.output_residual_width = 0;
        function_params.input_width_micro_repeats =
            workspace_width_micro_repeats;
        function_params.input_width_overall_micro_repeats =
            workspace_width_micro_repeats;
        function_params.output_width_micro_repeats =
            workspace_width_micro_repeats - 1;
      } else {
        function_params.output_residual_width = residual_output_micro_width;
        function_params.input_width_micro_repeats =
            width_trailing_micro_repeats;
        function_params.input_width_overall_micro_repeats =
            width_overall_trailing_micro_repeats;
        function_params.output_width_micro_repeats =
            output_width_trailing_micro_repeats;
      }
      function_params.output_width_overall_micro_repeats =
          function_params.output_residual_width == 0
              ? function_params.output_width_micro_repeats
              : function_params.output_width_micro_repeats + 1;

      for (int j_depth = 0; j_depth < depth_overall_macro_count; ++j_depth) {
        const uint8* input_data_block =
            input_data + b * input_batch_stride +
            j_depth * input_depth_macro_stride +
            k_width * input_width_macro_stride -
            function_params.padding_left * input_depth -
            function_params.padding_top * input_height_stride;
        uint8* output_data_block = output_data + b * output_batch_stride +
                                   j_depth * 64 +
                                   k_width * output_width_macro_stride;

        function_params.depth_micro_repeats =
            j_depth == depth_macro_count ? depth_trailing_micro_repeats : 8;
        // Under depth multiplication the workspace_height_stride does not have
        // to depend on input_width_overall_micro_repeats, but this improves the
        // compactness of workspace use.
        const int workspace_height_stride =
            has_depth_multiplication
                ? 16 * ((function_params.input_width_overall_micro_repeats +
                         3) >>
                        2)
                : 4 * function_params.input_width_overall_micro_repeats * 8 *
                      function_params.depth_micro_repeats;
        TFLITE_DCHECK_EQ(workspace_height_stride % 16, 0);
        function_params.workspace_height_stride = workspace_height_stride;

        // For the first macro block for output rows we fill in the first few
        // rows.  After this we will copy them (see below in loop.)
        function_params.inbound_block_height = input_height_overlap;
        pack_macro_block_func(-1, k_width, input_data_block,
                              macroblock_workspace, &function_params);
        input_data_block += input_height_stride * input_height_overlap;

        for (int i_height = 0; i_height < height_overall_macro_count;
             ++i_height) {
          if (i_height != height_macro_count) {
            function_params.inbound_block_height = input_height_per_macro;
            function_params.outbound_block_height = output_height_per_macro;
          } else {
            function_params.inbound_block_height =
                residual_output_height * stride;
            function_params.outbound_block_height = residual_output_height;
          }
          TFLITE_DCHECK_LT(i_height * output_height_per_macro, output_height);
          TFLITE_DCHECK_LT(i_height * input_height_per_macro, input_height);
          TFLITE_DCHECK_LT(k_width * output_width_per_macro_block,
                           output_width);
          TFLITE_DCHECK_LT(k_width * consumed_width_per_macro_block,
                           input_width);

          // Macro blocks overlap by input_height_overlap rows, so we copy
          // those instead of filling in afresh.  The first macro block across
          // output rows was filled in outside of the loop (above).
          if (i_height > 0) {
            memcpy(macroblock_workspace,
                   macroblock_workspace +
                       input_height_per_macro * workspace_height_stride,
                   input_height_overlap * workspace_height_stride);
          }

          pack_macro_block_func(
              i_height, k_width, input_data_block,
              macroblock_workspace +
                  input_height_overlap * workspace_height_stride,
              &function_params);

          kernel_macro_block_func(macroblock_workspace,
                                  filter_workspace[8 * j_depth][0][0][0],
                                  adjusted_bias_data + 64 * j_depth,
                                  output_data_block, &function_params);

          input_data_block += input_height_stride * input_height_per_macro;
          output_data_block += output_height_stride * output_height_per_macro;
        }
      }
    }
  }
}

}  // namespace depthwise_conv
}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_DEPTHWISECONV_UINT8_TRANSITIONAL_H_
