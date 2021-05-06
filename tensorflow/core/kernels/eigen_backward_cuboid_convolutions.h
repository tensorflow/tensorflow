/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

/** CuboidConvolutionBackwardInput
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Computes the backprop for the input of a 3D convolution.
 *
 * The output_backward parameter is expected to be a tensor with a rank of 4 or
 * more (channels, depth, height, width, and optionally others)
 * The kernel parameter is expected to be a 5D tensor (filters, channels,
 * kernel_depth, kernel_height, kernel_width)
 * output_backward and kernel have to be in the same layout.
 *
 * The dimensions of the result will be filters, depth, height, width (and
 * others if applicable).
 *
 * It is possible to swap the order of the depth, width and height dimensions
 * provided that the same order is used in the input, the kernel, and the
 * output.
 *
 * All dimension orders above are given for col-major, and should be reversed
 * for row-major.
 */

template <typename OutputBackward, typename Kernel>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<OutputBackward>::Layout == ColMajor,
    TensorReshapingOp<
        const DSizes<typename internal::traits<OutputBackward>::Index,
                     internal::traits<OutputBackward>::NumDimensions>,
        const TensorContractionOp<
            const array<
                IndexPair<typename internal::traits<OutputBackward>::Index>, 1>,
            const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorShufflingOp<
                    const array<
                        typename internal::traits<OutputBackward>::Index, 5>,
                    const TensorReverseOp<const Eigen::array<bool, 5>,
                                          const Kernel>>>>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const OutputBackward>>>>,
    TensorReshapingOp<
        const DSizes<typename internal::traits<OutputBackward>::Index,
                     internal::traits<OutputBackward>::NumDimensions>,
        const TensorContractionOp<
            const array<
                IndexPair<typename internal::traits<OutputBackward>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const OutputBackward>>,
            const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorShufflingOp<
                    const array<
                        typename internal::traits<OutputBackward>::Index, 5>,
                    const TensorReverseOp<const Eigen::array<bool, 5>,
                                          const Kernel>>>>>>>::type
CuboidConvolutionBackwardInput(
    const Kernel& kernel, const OutputBackward& output_backward,
    typename internal::traits<OutputBackward>::Index inputPlanes,
    typename internal::traits<OutputBackward>::Index inputRows,
    typename internal::traits<OutputBackward>::Index inputCols,
    const DenseIndex plane_stride = 1, const DenseIndex row_stride = 1,
    const DenseIndex col_stride = 1) {
  typedef typename internal::traits<OutputBackward>::Index TensorIndex;
  const TensorRef<const Tensor<typename internal::traits<Kernel>::Scalar,
                               internal::traits<Kernel>::NumDimensions,
                               internal::traits<Kernel>::Layout, TensorIndex>>
      kern(kernel);
  const TensorRef<
      const Tensor<typename internal::traits<OutputBackward>::Scalar,
                   internal::traits<OutputBackward>::NumDimensions,
                   internal::traits<OutputBackward>::Layout, TensorIndex>>
      out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Kernel>::Layout ==
                          internal::traits<OutputBackward>::Layout,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor =
      (internal::traits<OutputBackward>::Layout == ColMajor);

  static const int NumDims = internal::traits<OutputBackward>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[4];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[3];
  const TensorIndex kernelPlanes =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[4] : kern.dimensions()[0];

  const TensorIndex outputPlanes =
      isColMajor ? out.dimensions()[1] : out.dimensions()[NumDims - 2];
  const TensorIndex outputRows =
      isColMajor ? out.dimensions()[2] : out.dimensions()[NumDims - 3];
  const TensorIndex outputCols =
      isColMajor ? out.dimensions()[3] : out.dimensions()[NumDims - 4];

  // TODO(ezhulenev): Add support for inflated strides. Without inflated strides
  // effective kernel planes/rows/cols are always the same as the kernel itself
  // (see eigen_spatial_convolutions for details).
  const TensorIndex kernelPlanesEff = kernelPlanes;
  const TensorIndex kernelRowsEff = kernelRows;
  const TensorIndex kernelColsEff = kernelCols;

  // Computing the forward padding.
  const TensorIndex forward_pad_top_z = numext::maxi<Index>(
      0,
      ((outputPlanes - 1) * plane_stride + kernelPlanesEff - inputPlanes) / 2);
  const TensorIndex forward_pad_top = numext::maxi<Index>(
      0, ((outputRows - 1) * row_stride + kernelRowsEff - inputRows) / 2);
  const TensorIndex forward_pad_left = numext::maxi<Index>(
      0, ((outputCols - 1) * col_stride + kernelColsEff - inputCols) / 2);

  const TensorIndex padding_top_z = kernelPlanesEff - 1 - forward_pad_top_z;
  const TensorIndex padding_top = kernelRowsEff - 1 - forward_pad_top;
  const TensorIndex padding_left = kernelColsEff - 1 - forward_pad_left;

  const TensorIndex padding_bottom_z = inputPlanes -
                                       (outputPlanes - 1) * plane_stride - 2 -
                                       padding_top_z + kernelPlanesEff;
  const TensorIndex padding_bottom = inputRows - (outputRows - 1) * row_stride -
                                     2 - padding_top + kernelRowsEff;
  const TensorIndex padding_right = inputCols - (outputCols - 1) * col_stride -
                                    2 - padding_left + kernelColsEff;

  eigen_assert(padding_top_z >= 0);
  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom_z >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The kernel has dimensions :
  //   filters x channels x patch_planes x patch_rows x patch_cols.
  // We need to reverse the kernel along the spatial dimensions.
  Eigen::array<bool, 5> kernel_reverse;
  if (isColMajor) {
    kernel_reverse[0] = false;
    kernel_reverse[1] = false;
    kernel_reverse[2] = true;
    kernel_reverse[3] = true;
    kernel_reverse[4] = true;
  } else {
    kernel_reverse[0] = true;
    kernel_reverse[1] = true;
    kernel_reverse[2] = true;
    kernel_reverse[3] = false;
    kernel_reverse[4] = false;
  }

  // Reorder the dimensions to:
  //   filters x patch_planes x patch_rows x patch_cols x channels
  array<TensorIndex, 5> kernel_shuffle;
  if (isColMajor) {
    //  From: filters x channels x planes x rows x cols
    //  To:   filters x planes x rows x cols x channels
    kernel_shuffle[0] = 0;
    kernel_shuffle[1] = 2;
    kernel_shuffle[2] = 3;
    kernel_shuffle[3] = 4;
    kernel_shuffle[4] = 1;
  } else {
    //  From: cols x rows x planes x channels x filters
    //  To:   channels x cols x rows x planes x filters
    kernel_shuffle[0] = 3;
    kernel_shuffle[1] = 0;
    kernel_shuffle[2] = 1;
    kernel_shuffle[3] = 2;
    kernel_shuffle[4] = 4;
  }

  // Collapse the dims
  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters * kernelPlanes * kernelRows * kernelCols;
    kernel_dims[1] = kernelChannels;
  } else {
    kernel_dims[1] = kernelFilters * kernelPlanes * kernelRows * kernelCols;
    kernel_dims[0] = kernelChannels;
  }

  // The output_backward has dimensions out_depth X out_planes X out_rows X
  // out_cols X OTHERS
  // When we extract the image patches from output_backward, it will have
  // dimensions:
  //   out_depth X (patch_planes * patch_rows * patch_cols) X (input_planes *
  //   input_rows * input_cols * OTHERS)
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] =
        kernelFilters * kernelPlanes * kernelRows * kernelCols;
    pre_contract_dims[1] = inputPlanes * inputRows * inputCols;
    for (int i = 4; i < NumDims; ++i) {
      pre_contract_dims[1] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[1] =
        kernelFilters * kernelPlanes * kernelRows * kernelCols;
    pre_contract_dims[0] = inputPlanes * inputRows * inputCols;
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // We will contract along the collapsed dimension that contains the
  // kernelFilters, kernelPlanes, kernelRows and kernelCols.
  array<IndexPair<TensorIndex>, 1> contract_dims;
  if (isColMajor) {
    // col-major: kernel.contract(output.patches)
    contract_dims[0] = IndexPair<TensorIndex>(0, 0);
  } else {
    // row-major: output.patches.contract(kernel)
    contract_dims[0] = IndexPair<TensorIndex>(1, 1);
  }

  // Post contraction, the dimensions of the input_backprop is
  //  channels X input_planes X input_rows X input_cols X OTHERS
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelChannels;
    post_contract_dims[1] = inputPlanes;
    post_contract_dims[2] = inputRows;
    post_contract_dims[3] = inputCols;
    for (int i = 4; i < NumDims; ++i) {
      post_contract_dims[i] = out.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelChannels;
    post_contract_dims[NumDims - 2] = inputPlanes;
    post_contract_dims[NumDims - 3] = inputRows;
    post_contract_dims[NumDims - 4] = inputCols;
    for (int i = 0; i < NumDims - 4; ++i) {
      post_contract_dims[i] = out.dimension(i);
    }
  }

  return choose(
      Cond<internal::traits<OutputBackward>::Layout == ColMajor>(),
      kernel.reverse(kernel_reverse)
          .shuffle(kernel_shuffle)
          .reshape(kernel_dims)
          .eval()
          .contract(output_backward
                        .extract_volume_patches(
                            kernelPlanes, kernelRows, kernelCols, 1, 1, 1,
                            plane_stride, row_stride, col_stride, padding_top_z,
                            padding_bottom_z, padding_top, padding_bottom,
                            padding_left, padding_right)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      output_backward
          .extract_volume_patches(kernelPlanes, kernelRows, kernelCols, 1, 1, 1,
                                  plane_stride, row_stride, col_stride,
                                  padding_top_z, padding_bottom_z, padding_top,
                                  padding_bottom, padding_left, padding_right)
          .reshape(pre_contract_dims)
          .contract(kernel.reverse(kernel_reverse)
                        .shuffle(kernel_shuffle)
                        .reshape(kernel_dims)
                        .eval(),
                    contract_dims)
          .reshape(post_contract_dims));
}

/** CuboidConvolutionBackwardKernel
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Computes the backprop for the filter of a 3D convolution.
 *
 * The output_backward parameter is expected to be a tensor with a rank of 4 or
 * more (channels, depth, height, width, and optionally others)
 * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_depth, kernel_height, kernel_width)
 * output_backward and kernel have to be in the same layout.
 *
 * The dimensions of the result will be filters, depth, height, width (and
 * others if applicable).
 *
 * It is possible to swap the order of the depth, width and height dimensions
 * provided that the same order is used in the input, the kernel, and the
 * output.
 *
 * All dimension orders above are given for col-major, and should be reversed
 * for row-major.
 */
template <typename OutputBackward, typename Input>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<Input>::Layout == ColMajor,
    const TensorReverseOp<
        const Eigen::array<typename internal::traits<Input>::Index,
                           internal::traits<Input>::NumDimensions>,
        const Eigen::TensorShufflingOp<
            const Eigen::array<typename internal::traits<Input>::Index,
                               internal::traits<Input>::NumDimensions>,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename internal::traits<Input>::Index,
                                    internal::traits<Input>::NumDimensions>,
                const TensorContractionOp<
                    const array<
                        IndexPair<typename internal::traits<Input>::Index>, 1>,
                    const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     2>,
                        const Eigen::TensorShufflingOp<
                            const Eigen::array<
                                typename internal::traits<Input>::Index,
                                internal::traits<Input>::NumDimensions>,
                            const OutputBackward>>>,
                    const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     2>,
                        const TensorVolumePatchOp<
                            Dynamic, Dynamic, Dynamic,
                            const Eigen::TensorForcedEvalOp<
                                const Eigen::TensorShufflingOp<
                                    const Eigen::array<
                                        typename internal::traits<Input>::Index,
                                        internal::traits<Input>::NumDimensions>,
                                    const Input>>>>>>>>,
    const TensorReverseOp<
        const Eigen::array<typename internal::traits<Input>::Index,
                           internal::traits<Input>::NumDimensions>,
        const Eigen::TensorShufflingOp<
            const Eigen::array<typename internal::traits<Input>::Index,
                               internal::traits<Input>::NumDimensions>,
            const Eigen::TensorReshapingOp<
                const Eigen::DSizes<typename internal::traits<Input>::Index,
                                    internal::traits<Input>::NumDimensions>,
                const TensorContractionOp<
                    const array<
                        IndexPair<typename internal::traits<Input>::Index>, 1>,
                    const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     2>,
                        const TensorVolumePatchOp<
                            Dynamic, Dynamic, Dynamic,
                            const Eigen::TensorForcedEvalOp<
                                const Eigen::TensorShufflingOp<
                                    const Eigen::array<
                                        typename internal::traits<Input>::Index,
                                        internal::traits<Input>::NumDimensions>,
                                    const Input>>>>,
                    const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     2>,
                        const Eigen::TensorShufflingOp<
                            const Eigen::array<
                                typename internal::traits<Input>::Index,
                                internal::traits<Input>::NumDimensions>,
                            const OutputBackward>>>>>>>>::type
CuboidConvolutionBackwardKernel(
    const Input& input, const OutputBackward& output_backward,
    typename internal::traits<Input>::Index kernelPlanes,
    typename internal::traits<Input>::Index kernelRows,
    typename internal::traits<Input>::Index kernelCols,
    const DenseIndex stridePlanes = 1, const DenseIndex strideRows = 1,
    const DenseIndex strideCols = 1) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex>>
      in(input);
  TensorRef<Tensor<typename internal::traits<OutputBackward>::Scalar,
                   internal::traits<OutputBackward>::NumDimensions,
                   internal::traits<OutputBackward>::Layout, TensorIndex>>
      out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout ==
                          internal::traits<OutputBackward>::Layout,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  static const int NumDims = internal::traits<Input>::NumDimensions;
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions ==
                          internal::traits<OutputBackward>::NumDimensions,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  // We do not support higher dimensional backward convolutions, or convolutions
  // without batch dimension.
  // TODO(ezhulenev): Relax this constraint, and turn on tests without batch
  // dimension in eigen_backward_cuboid_convolutions_test.cc.
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 5,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  const TensorIndex inputPlanes =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputRows =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const TensorIndex inputCols =
      isColMajor ? in.dimension(3) : in.dimension(NumDims - 4);

  const TensorIndex outputPlanes =
      isColMajor ? out.dimension(1) : out.dimension(NumDims - 2);
  const TensorIndex outputRows =
      isColMajor ? out.dimension(2) : out.dimension(NumDims - 3);
  const TensorIndex outputCols =
      isColMajor ? out.dimension(3) : out.dimension(NumDims - 4);

  // Number of filters. This is the same as the output depth.
  const TensorIndex kernelFilters =
      isColMajor ? out.dimension(0) : out.dimension(NumDims - 1);
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? in.dimension(0) : in.dimension(NumDims - 1);

  // Number of batches in the input tensor.
  const TensorIndex batch =
      isColMajor ? in.dimension(4) : in.dimension(NumDims - 5);

  // TODO(ezhulenev): Add support for inflated strides. Without inflated strides
  // effective kernel planes/rows/cols are always the same as the kernel itself
  // (see eigen_spatial_convolutions for details).
  const TensorIndex kernelPlanesEff = kernelPlanes;
  const TensorIndex kernelRowsEff = kernelRows;
  const TensorIndex kernelColsEff = kernelCols;

  // Compute forward padding from input and output_backward dimensions.
  const TensorIndex padPlanes = numext::maxi<Index>(
      0, (outputPlanes - 1) * stridePlanes + kernelPlanesEff - inputPlanes);
  const TensorIndex padRows = numext::maxi<Index>(
      0, (outputRows - 1) * strideRows + kernelRowsEff - inputRows);
  const TensorIndex padCols = numext::maxi<Index>(
      0, (outputCols - 1) * strideCols + kernelColsEff - inputCols);

  const TensorIndex padding_top_z = padPlanes / 2;
  const TensorIndex padding_top = padRows / 2;
  const TensorIndex padding_left = padCols / 2;

  // Compute paddings for output_backward before extracting patches.
  const auto expanded_out_planes = (outputPlanes - 1) * stridePlanes + 1;
  const auto expanded_out_rows = (outputRows - 1) * strideRows + 1;
  const auto expanded_out_cols = (outputCols - 1) * strideCols + 1;
  const auto padded_out_planes = inputPlanes + kernelPlanes - 1;
  const auto padded_out_rows = inputRows + kernelRows - 1;
  const auto padded_out_cols = inputCols + kernelCols - 1;
  const auto top_pad_planes = kernelPlanes - 1 - padding_top_z;
  const auto top_pad_rows = kernelRows - 1 - padding_top;
  const auto left_pad_cols = kernelCols - 1 - padding_left;
  const auto bottom_pad_planes =
      padded_out_planes - expanded_out_planes - top_pad_planes;
  const auto bottom_pad_rows =
      padded_out_rows - expanded_out_rows - top_pad_rows;
  const auto right_pad_cols =
      padded_out_cols - expanded_out_cols - left_pad_cols;

  // Reorder output_backward dimensions.
  array<TensorIndex, 5> output_backward_shuffle;
  if (isColMajor) {
    // From: [out_depth, out_planes, out_rows, out_cols, batch]
    // To:   [batch, out_planes, out_rows, out_cols, out_depth]
    output_backward_shuffle = {4, 1, 2, 3, 0};
  } else {
    // From: [batch, out_cols, out_rows, out_planes, out_depth]
    // To:   [out_depth, out_cols, out_rows, out_planes, batch]
    output_backward_shuffle = {4, 1, 2, 3, 0};
  }

  // Reorder input dimensions.
  array<TensorIndex, 5> input_shuffle;
  if (isColMajor) {
    // From: [in_depth, in_planes, in_rows, in_cols, batch]
    // To:   [in_depth, batch, in_planes, in_rows, in_cols]
    input_shuffle = {0, 4, 1, 2, 3};
  } else {
    // From: [batch, in_cols, in_rows, in_planes, in_depth]
    // To:   [in_cols, in_rows, in_planes, batch, in_depth]
    input_shuffle = {1, 2, 3, 0, 4};
  }

  // Input is playing the role of a "kernel" in this convolution.
  DSizes<TensorIndex, 2> input_dims;
  if (isColMajor) {
    input_dims[0] = kernelChannels;
    input_dims[1] = batch * inputPlanes * inputRows * inputCols;
  } else {
    input_dims[1] = kernelChannels;
    input_dims[0] = inputCols * inputRows * inputPlanes * batch;
  }

  // Molds the output of the patch extraction result into a 2D tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = batch * inputPlanes * inputRows * inputCols;
    pre_contract_dims[1] =
        kernelPlanes * kernelRows * kernelCols * kernelFilters;
  } else {
    pre_contract_dims[1] = inputCols * inputRows * inputPlanes * batch;
    pre_contract_dims[0] =
        kernelFilters * kernelCols * kernelRows * kernelPlanes;
  }

  // We will contract along the collapsed dimension that contains the
  // batch, inputPlanes, inputRows and inputCols.
  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  // Dimensions after contraction.
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelChannels;
    post_contract_dims[1] = kernelPlanes;
    post_contract_dims[2] = kernelRows;
    post_contract_dims[3] = kernelCols;
    post_contract_dims[4] = kernelFilters;
  } else {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = kernelCols;
    post_contract_dims[2] = kernelRows;
    post_contract_dims[3] = kernelPlanes;
    post_contract_dims[4] = kernelChannels;
  }

  // Reorder output of contraction to valid filter shape.
  array<TensorIndex, 5> kernel_shuffle;
  if (isColMajor) {
    // From: [in_depth, kernel_planes, kernel_rows, kernel_cols, out_depth]
    // To:   [out_depth, in_depth, kernel_planes, kernel_rows, kernel_cols]
    kernel_shuffle = {4, 0, 1, 2, 3};
  } else {
    // From: [out_depth, kernel_cols, kernel_rows, kernel_planes, in_depth]
    // To:   [kernel_cols, kernel_rows, kernel_planes, in_depth, out_depth]
    kernel_shuffle = {1, 2, 3, 4, 0};
  }

  // Reverse kernel backprop dimensions.
  array<TensorIndex, 5> kernel_reverse;
  if (isColMajor) {
    kernel_reverse = {false, false, true, true, true};
  } else {
    kernel_reverse = {true, true, true, false, false};
  }

  // Create convolution input (aka source of patches) from output backward
  // tensor by shuffling dimensions.
  const auto the_input =
      output_backward.shuffle(output_backward_shuffle).eval();

  // Create convolution kernel (aka filter) from input by shuffling and
  // reshaping.
  const auto the_kernel =
      input.shuffle(input_shuffle).reshape(input_dims).eval();

  return choose(Cond<internal::traits<Input>::Layout == ColMajor>(),
                the_kernel.contract(
                    the_input
                        .extract_volume_patches(
                            inputPlanes, inputRows, inputCols, 1, 1, 1,
                            stridePlanes, strideRows, strideCols,
                            top_pad_planes, bottom_pad_planes, top_pad_rows,
                            bottom_pad_rows, left_pad_cols, right_pad_cols)
                        .reshape(pre_contract_dims),
                    contract_dims),
                the_input
                    .extract_volume_patches(
                        inputPlanes, inputRows, inputCols, 1, 1, 1,
                        stridePlanes, strideRows, strideCols, top_pad_planes,
                        bottom_pad_planes, top_pad_rows, bottom_pad_rows,
                        left_pad_cols, right_pad_cols)
                    .reshape(pre_contract_dims)
                    .contract(the_kernel, contract_dims))
      .reshape(post_contract_dims)
      .shuffle(kernel_shuffle)
      .reverse(kernel_reverse);
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_
