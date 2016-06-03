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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_SPATIAL_CONVOLUTIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_SPATIAL_CONVOLUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace Eigen {

/** SpatialConvolutionBackwardInput
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Computes the backprop for the input of a 2D convolution.
  *
  * The output_backward parameter is expected to be a tensor with a rank of 3 or
 * more (channels, height, width, and optionally others)
  * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
  * The output_backward and the kernel must both be in col-major layout. The
 * result will also be in col-major layout.
  *
  * If row_in_stride, col_in_stride > 1, then applies convolution with holes
 * (aka atrous convolution), sampling every row_in_stride, col_in_stride input
 * pixels.
  *
  * The result can be assigned to a tensor of rank equal to the rank of the
 * output_backward. The dimensions of the result will be filters, height, width
 * (and others if applicable).
  *
  * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
  *
  */
#ifdef EIGEN_HAS_INDEX_LIST
typedef IndexList<type2index<0>, type2index<0>, type2index<1>, type2index<1> >
    ReverseColMajor;
typedef IndexList<type2index<1>, type2index<1>, type2index<0>, type2index<0> >
    ReverseRowMajor;
#else
typedef array<bool, 4> ReverseColMajor;
typedef array<bool, 4> ReverseRowMajor;
#endif

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
                        typename internal::traits<OutputBackward>::Index, 4>,
                    const TensorReverseOp<const ReverseColMajor,
                                          const Kernel> > > >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorImagePatchOp<Dynamic, Dynamic,
                                         const OutputBackward> > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<OutputBackward>::Index,
                     internal::traits<OutputBackward>::NumDimensions>,
        const TensorContractionOp<
            const array<
                IndexPair<typename internal::traits<OutputBackward>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorImagePatchOp<Dynamic, Dynamic,
                                         const OutputBackward> >,
            const Eigen::TensorForcedEvalOp<const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             2>,
                const TensorShufflingOp<
                    const array<
                        typename internal::traits<OutputBackward>::Index, 4>,
                    const TensorReverseOp<const ReverseRowMajor,
                                          const Kernel> > > > > > >::type
SpatialConvolutionBackwardInput(
    const Kernel& kernel, const OutputBackward& output_backward,
    typename internal::traits<OutputBackward>::Index inputRows,
    typename internal::traits<OutputBackward>::Index inputCols,
    const DenseIndex row_stride = 1, const DenseIndex col_stride = 1,
    const DenseIndex row_in_stride = 1, const DenseIndex col_in_stride = 1) {
  typedef typename internal::traits<OutputBackward>::Index TensorIndex;
  typedef typename internal::traits<OutputBackward>::Scalar OutScalar;
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);
  TensorRef<Tensor<OutScalar, internal::traits<OutputBackward>::NumDimensions,
                   internal::traits<OutputBackward>::Layout, TensorIndex> >
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
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  // This is the effective kernel size, taking into account the (*_in_stride -
  // 1) zero-values
  // inserted between consecutive kernel elements in atrous convolution
  const TensorIndex kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const TensorIndex kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  const TensorIndex outputRows = isColMajor
                                     ? output_backward.dimension(1)
                                     : output_backward.dimension(NumDims - 2);
  const TensorIndex outputCols = isColMajor
                                     ? output_backward.dimension(2)
                                     : output_backward.dimension(NumDims - 3);

  // Computing the forward padding
  const TensorIndex forward_pad_top = numext::maxi<Index>(
      0, ((outputRows - 1) * row_stride + kernelRowsEff - inputRows) / 2);
  const TensorIndex forward_pad_left = numext::maxi<Index>(
      0, ((outputCols - 1) * col_stride + kernelColsEff - inputCols) / 2);
  const TensorIndex padding_top = kernelRowsEff - 1 - forward_pad_top;
  const TensorIndex padding_left = kernelColsEff - 1 - forward_pad_left;

  const TensorIndex padding_bottom = inputRows - (outputRows - 1) * row_stride -
                                     2 - padding_top + kernelRowsEff;
  const TensorIndex padding_right = inputCols - (outputCols - 1) * col_stride -
                                    2 - padding_left + kernelColsEff;

  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The kernel has dimensions filters X channels X patch_rows X patch_cols
  // We need to reverse the kernel along dimensions corresponding to rows and
  // cols.
  // TODO(yangke): we can make things slightly faster by collapsing the
  // dimensions
  // where we don't reverse. Try that once we have a faster compiler.
  typedef typename internal::conditional<isColMajor, ReverseColMajor,
                                         ReverseRowMajor>::type Reverse;
  Reverse kernel_reverse;

#ifndef EIGEN_HAS_INDEX_LIST
  if (isColMajor) {
    kernel_reverse[0] = false;
    kernel_reverse[1] = false;
    kernel_reverse[2] = true;
    kernel_reverse[3] = true;
  } else {
    kernel_reverse[0] = true;
    kernel_reverse[1] = true;
    kernel_reverse[2] = false;
    kernel_reverse[3] = false;
  }
#endif

  // Reorder the dimensions to filters X patch_rows X patch_cols X channels
  array<TensorIndex, 4> kernel_shuffle;
  if (isColMajor) {
    kernel_shuffle[0] = 0;
    kernel_shuffle[1] = 2;
    kernel_shuffle[2] = 3;
    kernel_shuffle[3] = 1;
  } else {
    kernel_shuffle[0] = 2;
    kernel_shuffle[1] = 0;
    kernel_shuffle[2] = 1;
    kernel_shuffle[3] = 3;
  }

  // Collapse the dims
  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters * kernelRows * kernelCols;
    kernel_dims[1] = kernelChannels;
  } else {
    kernel_dims[1] = kernelFilters * kernelRows * kernelCols;
    kernel_dims[0] = kernelChannels;
  }

  // The output_backward has dimensions out_depth X out_rows X out_cols X OTHERS
  // When we extract the image patches from output_backward, it will have
  // dimensions
  //   out_depth X (patch_rows * patch_cols) X (input_rows * input_cols *
  //   OTHERS)
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelFilters * kernelRows * kernelCols;
    pre_contract_dims[1] = inputRows * inputCols;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelFilters * kernelRows * kernelCols;
    pre_contract_dims[0] = inputRows * inputCols;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // We will contract along the fused dimension that contains the kernelFilters,
  // the kernelRows and the kernelCols.
  array<IndexPair<TensorIndex>, 1> contract_dims;
  if (isColMajor) {
    // col-major: kernel.contract(output.patches)
    contract_dims[0] = IndexPair<TensorIndex>(0, 0);
  } else {
    // row-major: output.patches.contract(kernel)
    contract_dims[0] = IndexPair<TensorIndex>(1, 1);
  }

  // Post contraction, the dimensions of the input_backprop is
  //  channels X input_rows X input_cols X OTHERS
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelChannels;
    post_contract_dims[1] = inputRows;
    post_contract_dims[2] = inputCols;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = out.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelChannels;
    post_contract_dims[NumDims - 2] = inputRows;
    post_contract_dims[NumDims - 3] = inputCols;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = out.dimension(i);
    }
  }

  return choose(
      Cond<internal::traits<OutputBackward>::Layout == ColMajor>(),
      kernel.reverse(kernel_reverse)
          .shuffle(kernel_shuffle)
          .reshape(kernel_dims)
          .eval()
          .contract(
              output_backward
                  .extract_image_patches(
                      kernelRows, kernelCols, 1, 1, row_in_stride,
                      col_in_stride, row_stride, col_stride, padding_top,
                      padding_bottom, padding_left, padding_right, OutScalar(0))
                  .reshape(pre_contract_dims),
              contract_dims)
          .reshape(post_contract_dims),
      output_backward
          .extract_image_patches(kernelRows, kernelCols, 1, 1, row_in_stride,
                                 col_in_stride, row_stride, col_stride,
                                 padding_top, padding_bottom, padding_left,
                                 padding_right, OutScalar(0))
          .reshape(pre_contract_dims)
          .contract(kernel.reverse(kernel_reverse)
                        .shuffle(kernel_shuffle)
                        .reshape(kernel_dims)
                        .eval(),
                    contract_dims)
          .reshape(post_contract_dims));
}

/** SpatialConvolutionBackwardKernel
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Computes the backprop for the filter of a 2D convolution.
  *
  * The output_backward parameter is expected to be a tensor with a rank of 3 or
 * more (channels, height, width, and optionally others)
  * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
  * The output_backward and the kernel must both be in col-major layout. The
 * result will also be in col-major layout.
  *
  * If row_in_stride, col_stride > 1, then applies convolution with holes (aka
 * atrous convolution), sampling every row_in_stride, col_in_stride input
 * pixels.
  *
  * The result can be assigned to a tensor of rank equal to the rank of the
 * output_backward. The dimensions of the result will be filters, height, width
 * (and others if applicable).
  *
  * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
  *
  */

template <typename OutputBackward, typename Input>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<OutputBackward>::Layout == ColMajor,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index, 4>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const OutputBackward>,
            const TensorShufflingOp<
                const array<typename internal::traits<OutputBackward>::Index,
                            2>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic,
                                             const Input> > > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index, 4>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorShufflingOp<
                const array<typename internal::traits<OutputBackward>::Index,
                            2>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const OutputBackward> > > >::type
SpatialConvolutionBackwardKernel(
    const Input& input, const OutputBackward& output_backward,
    typename internal::traits<Input>::Index kernelRows,
    typename internal::traits<Input>::Index kernelCols,
    const DenseIndex row_stride = 1, const DenseIndex col_stride = 1,
    const DenseIndex row_in_stride = 1, const DenseIndex col_in_stride = 1) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  typedef typename internal::traits<OutputBackward>::Scalar OutScalar;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<OutScalar, internal::traits<OutputBackward>::NumDimensions,
                   internal::traits<OutputBackward>::Layout, TensorIndex> >
      out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout ==
                          internal::traits<OutputBackward>::Layout,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  // stride and in_stride cannot both be larger than 1
  eigen_assert(!(row_stride > 1 && row_in_stride > 1) &&
               !(col_stride > 1 && col_in_stride > 1));

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  static const int NumDims = internal::traits<Input>::NumDimensions;
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions ==
                          internal::traits<OutputBackward>::NumDimensions,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  const TensorIndex inputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  const TensorIndex outputRows = isColMajor
                                     ? output_backward.dimension(1)
                                     : output_backward.dimension(NumDims - 2);
  const TensorIndex outputCols = isColMajor
                                     ? output_backward.dimension(2)
                                     : output_backward.dimension(NumDims - 3);

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? out.dimensions()[0] : out.dimensions()[NumDims - 1];

  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? in.dimensions()[0] : in.dimensions()[NumDims - 1];

  // This is the effective kernel size, taking into account the (*_in_stride -
  // 1) zero-values
  // inserted between consecutive kernel elements in atrous convolution
  const TensorIndex kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const TensorIndex kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  // Computing the forward padding
  const TensorIndex padRows = numext::maxi<Index>(
      0, (outputRows - 1) * row_stride + kernelRowsEff - inputRows);
  const TensorIndex padCols = numext::maxi<Index>(
      0, (outputCols - 1) * col_stride + kernelColsEff - inputCols);
  const TensorIndex padding_top = padRows / 2;
  const TensorIndex padding_bottom = padRows - padding_top;
  const TensorIndex padding_left = padCols / 2;
  const TensorIndex padding_right = padCols - padding_left;

  // Reshaped out
  DSizes<TensorIndex, 2> output_dims;
  if (isColMajor) {
    output_dims[0] = kernelFilters;
    output_dims[1] = outputRows * outputCols;
    for (int i = 3; i < NumDims; ++i) {
      output_dims[1] *= out.dimension(i);
    }
  } else {
    output_dims[1] = kernelFilters;
    output_dims[0] = outputCols * outputRows;
    for (int i = 0; i < NumDims - 3; ++i) {
      output_dims[0] *= out.dimension(i);
    }
  }

  // Reshaped extract_image_patches(in)
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = outputRows * outputCols;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
    eigen_assert(output_dims[1] == pre_contract_dims[1]);
  } else {
    pre_contract_dims[1] = kernelCols * kernelRows * kernelChannels;
    pre_contract_dims[0] = outputRows * outputCols;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
    eigen_assert(output_dims[0] == pre_contract_dims[0]);
  }

  array<TensorIndex, 2> shuffle_dims;
  shuffle_dims[0] = 1;
  shuffle_dims[1] = 0;

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  // After the contraction, the kernel will have the desired shape
  // out_depth X in_shape X kernel_rows X kernel_cols
  DSizes<TensorIndex, 4> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels;
    kernel_dims[2] = kernelRows;
    kernel_dims[3] = kernelCols;
  } else {
    kernel_dims[3] = kernelFilters;
    kernel_dims[2] = kernelChannels;
    kernel_dims[1] = kernelRows;
    kernel_dims[0] = kernelCols;
  }

  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      output_backward.reshape(output_dims)
          .contract(
              input
                  .extract_image_patches(
                      kernelRows, kernelCols, row_stride, col_stride,
                      row_in_stride, col_in_stride, 1, 1, padding_top,
                      padding_bottom, padding_left, padding_right, OutScalar(0))
                  .reshape(pre_contract_dims)
                  .shuffle(shuffle_dims),
              contract_dims)
          .reshape(kernel_dims),
      input
          .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                                 row_in_stride, col_in_stride, 1, 1,
                                 padding_top, padding_bottom, padding_left,
                                 padding_right, OutScalar(0))
          .reshape(pre_contract_dims)
          .shuffle(shuffle_dims)
          .contract(output_backward.reshape(output_dims), contract_dims)
          .reshape(kernel_dims));
}

}  // end namespace Eigen

#endif  // EIGEN_CXX11_NEURAL_NETWORKS_BACKWARD_SPATIAL_CONVOLUTIONS_H
