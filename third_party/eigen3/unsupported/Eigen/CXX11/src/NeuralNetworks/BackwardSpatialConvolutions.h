// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015 Ke Yang <yangke@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_NEURAL_NETWORKS_BACKWARD_SPATIAL_CONVOLUTIONS_H
#define EIGEN_CXX11_NEURAL_NETWORKS_BACKWARD_SPATIAL_CONVOLUTIONS_H

namespace Eigen {

/** SpatialConvolutionBackwardInput
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Computes the backprop for the input of a 2D convolution.
  *
  * The output_backward parameter is expected to be a tensor with a rank of 3 or more (channels, height, width, and optionally others)
  * The kernel parameter is expected to be a 4D tensor (filters, channels, kernel_height, kernel_width)
  * The output_backward and the kernel must both be in col-major layout. The result will also be in col-major layout.
  *
  * If in_stride > 1, then applies convolution with holes (aka atrous convolution), sampling every in_stride input pixels.
  *
  * The result can be assigned to a tensor of rank equal to the rank of the output_backward. The dimensions of the result will be filters, height, width (and others if applicable).
  *
  * It is possible to swap the order of the width and height dimensions provided that the same order is used in the input, the kernel, and the output.
  *
  */

template <typename OutputBackward, typename Kernel>
EIGEN_ALWAYS_INLINE
static const typename internal::conditional<
  internal::traits<OutputBackward>::Layout == ColMajor,
  TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, internal::traits<OutputBackward>::NumDimensions>, const TensorContractionOp<const array<IndexPair<typename internal::traits<OutputBackward>::Index>, 2>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 3>, const TensorReverseOp<const array<bool, 4>, const Kernel> >, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 3>, const TensorImagePatchOp<Dynamic, Dynamic, const OutputBackward> > > >,
  TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, internal::traits<OutputBackward>::NumDimensions>, const TensorContractionOp<const array<IndexPair<typename internal::traits<OutputBackward>::Index>, 2>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 3>, const TensorImagePatchOp<Dynamic, Dynamic, const OutputBackward> >, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 3>, const TensorReverseOp<const array<bool, 4>, const Kernel> > > > >::type
SpatialConvolutionBackwardInput(const Kernel& kernel, const OutputBackward& output_backward, typename internal::traits<OutputBackward>::Index inputRows, typename internal::traits<OutputBackward>::Index inputCols, const DenseIndex stride = 1, const DenseIndex in_stride = 1) {

  typedef typename internal::traits<OutputBackward>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar, internal::traits<Kernel>::NumDimensions, internal::traits<Kernel>::Layout, TensorIndex> > kern(kernel);
  TensorRef<Tensor<typename internal::traits<OutputBackward>::Scalar, internal::traits<OutputBackward>::NumDimensions, internal::traits<OutputBackward>::Layout, TensorIndex> > out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Kernel>::Layout == internal::traits<OutputBackward>::Layout, YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor = (internal::traits<OutputBackward>::Layout == ColMajor);

  static const int NumDims = internal::traits<OutputBackward>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the result
  const TensorIndex kernelFilters = isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels = isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows = isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols = isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  // This is the effective kernel size, taking into account the (in_stride - 1) zero-values
  // inserted between consecutive kernel elements in atrous convolution
  const TensorIndex kernelRowsEff = kernelRows + (kernelRows - 1) * (in_stride - 1);
  const TensorIndex kernelColsEff = kernelCols + (kernelCols - 1) * (in_stride - 1);

  const TensorIndex outputRows = isColMajor ? output_backward.dimension(1) : output_backward.dimension(NumDims - 2);
  const TensorIndex outputCols = isColMajor ? output_backward.dimension(2) : output_backward.dimension(NumDims - 3);

  // Computing the forward padding
  const TensorIndex forward_pad_top = ((outputRows - 1) * stride + kernelRowsEff - inputRows) / 2;
  const TensorIndex forward_pad_left = ((outputCols - 1) * stride + kernelColsEff - inputCols) / 2;

  const TensorIndex padding_top = kernelRowsEff - 1 - forward_pad_top;
  const TensorIndex padding_left = kernelColsEff - 1 - forward_pad_left;
  const TensorIndex padding_bottom = inputRows + kernelRowsEff - 1 - (outputRows - 1) * stride - 1 - padding_top;
  const TensorIndex padding_right = inputCols + kernelColsEff - 1 - (outputCols - 1) * stride - 1 - padding_left;

  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The kernel has dimensions filters X channels X patch_rows X patch_cols
  // We need to reverse the kernel along dimensions corresponding to rows and
  // cols.
  // TODO(yangke): we can make things slightly faster by collapsing the dimensions
  // where we don't reverse. Try that once we have a faster compiler.
  array<bool, 4> kernel_reverse;
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

  DSizes<TensorIndex, 3> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels;
    kernel_dims[2] = kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelRows * kernelCols;
    kernel_dims[1] = kernelChannels;
    kernel_dims[2] = kernelFilters;
  }

  // The output_backward has dimensions out_depth X out_rows X out_cols X OTHERS
  // When we extract the image patches from output_backward, it will have dimensions
  //   out_depth X (patch_rows * patch_cols) X (input_rows * input_cols * OTHERS)
  DSizes<TensorIndex, 3> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelFilters;
    pre_contract_dims[1] = kernelRows * kernelCols;
    pre_contract_dims[2] = inputRows * inputCols;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[2] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[2] = kernelFilters;
    pre_contract_dims[1] = kernelRows * kernelCols;
    pre_contract_dims[0] = inputRows * inputCols;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // We will contract along dimensions (0, 2) in kernel and (0, 1) in
  // output_backward, if this is col-major, and
  // dimensions (0, 2) in kernel and (1, 2) in output_backward, if this row-major.
  array<IndexPair<TensorIndex>, 2> contract_dims;
  if (isColMajor) {
    // col-major: kernel.contract(output.patches)
    contract_dims[0] = IndexPair<TensorIndex>(0, 0);
    contract_dims[1] = IndexPair<TensorIndex>(2, 1);
  } else {
    // row-major: output.patches.contract(kernel)
    contract_dims[0] = IndexPair<TensorIndex>(1, 0);
    contract_dims[1] = IndexPair<TensorIndex>(2, 2);
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

  return choose(Cond<internal::traits<OutputBackward>::Layout == ColMajor>(),
                kernel.reverse(kernel_reverse).reshape(kernel_dims).contract(output_backward.extract_image_patches(kernelRows, kernelCols, 1, 1, in_stride, in_stride, stride, stride, padding_top, padding_bottom, padding_left, padding_right, 0).reshape(pre_contract_dims), contract_dims).reshape(post_contract_dims),
                output_backward.extract_image_patches(kernelRows, kernelCols, 1, 1, in_stride, in_stride, stride, stride, padding_top, padding_bottom, padding_left, padding_right, 0).reshape(pre_contract_dims).contract(kernel.reverse(kernel_reverse).reshape(kernel_dims), contract_dims).reshape(post_contract_dims));
}


/** SpatialConvolutionBackwardKernel
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Computes the backprop for the filter of a 2D convolution.
  *
  * The output_backward parameter is expected to be a tensor with a rank of 3 or more (channels, height, width, and optionally others)
  * The kernel parameter is expected to be a 4D tensor (filters, channels, kernel_height, kernel_width)
  * The output_backward and the kernel must both be in col-major layout. The result will also be in col-major layout.
  *
  * If in_stride > 1, then applies convolution with holes (aka atrous convolution), sampling every in_stride input pixels.
  *
  * The result can be assigned to a tensor of rank equal to the rank of the output_backward. The dimensions of the result will be filters, height, width (and others if applicable).
  *
  * It is possible to swap the order of the width and height dimensions provided that the same order is used in the input, the kernel, and the output.
  *
  */
// TODO(gpapan): Resolve a bug in TensorContractionInputMapper at SpatialConvolutions.h that yangke circumvented by using .reshape().reshape().
// This can significantly accelerate SpatialConvolutionBackwardKernel.

template <typename OutputBackward, typename Input>
EIGEN_ALWAYS_INLINE
static const typename internal::conditional<
  internal::traits<OutputBackward>::Layout == ColMajor,
  const TensorShufflingOp<const array<typename internal::traits<OutputBackward>::Index, 4>, const TensorReverseOp<const array<bool, 4>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorContractionOp<const array<IndexPair<typename internal::traits<Input>::Index>, 2>, const TensorReshapingOp<const DSizes<typename internal::traits<Input>::Index, 3>, const Input>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorImagePatchOp<Dynamic, Dynamic, const OutputBackward> > > > > > >,
  const TensorShufflingOp<const array<typename internal::traits<OutputBackward>::Index, 4>, const TensorReverseOp<const array<bool, 4>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorContractionOp<const array<IndexPair<typename internal::traits<Input>::Index>, 2>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorReshapingOp<const DSizes<typename internal::traits<OutputBackward>::Index, 4>, const TensorImagePatchOp<Dynamic, Dynamic, const OutputBackward> > >, const TensorReshapingOp<const DSizes<typename internal::traits<Input>::Index, 3>, const Input> > > > > >::type
SpatialConvolutionBackwardKernel(const Input& input, const OutputBackward& output_backward, typename internal::traits<Input>::Index kernelRows, typename internal::traits<Input>::Index kernelCols, const DenseIndex stride = 1, const DenseIndex in_stride = 1) {

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);
  TensorRef<Tensor<typename internal::traits<OutputBackward>::Scalar, internal::traits<OutputBackward>::NumDimensions, internal::traits<OutputBackward>::Layout, TensorIndex> > out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout == internal::traits<OutputBackward>::Layout, YOU_MADE_A_PROGRAMMING_MISTAKE);

  // stride and in_stride cannot both be larger than 1
  eigen_assert(!(stride > 1 && in_stride > 1));

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  static const int NumDims = internal::traits<Input>::NumDimensions;
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == internal::traits<OutputBackward>::NumDimensions, YOU_MADE_A_PROGRAMMING_MISTAKE);

  const TensorIndex inputRows = isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputCols = isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  const TensorIndex outputRows = isColMajor ? output_backward.dimension(1) : output_backward.dimension(NumDims - 2);
  const TensorIndex outputCols = isColMajor ? output_backward.dimension(2) : output_backward.dimension(NumDims - 3);

  // Number of filters to apply. This is the same as the output depth of the result
  const TensorIndex kernelFilters = isColMajor ? out.dimensions()[0] : out.dimensions()[NumDims - 1];

  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels = isColMajor ? in.dimensions()[0] : in.dimensions()[NumDims - 1];

  // This is the effective kernel size, taking into account the (in_stride - 1) zero-values
  // inserted between consecutive kernel elements in atrous convolution
  const TensorIndex kernelRowsEff = kernelRows + (kernelRows - 1) * (in_stride - 1);
  const TensorIndex kernelColsEff = kernelCols + (kernelCols - 1) * (in_stride - 1);

  // Computing the forward padding
  const TensorIndex forward_pad_top = ((outputRows - 1) * stride + kernelRowsEff - inputRows) / 2;
  const TensorIndex forward_pad_left = ((outputCols - 1) * stride + kernelColsEff - inputCols) / 2;

  // TODO: factor out the padding computation.
  const TensorIndex padding_top = kernelRowsEff - 1 - forward_pad_top;
  const TensorIndex padding_left = kernelColsEff - 1 - forward_pad_left;
  const TensorIndex padding_bottom = inputRows + kernelRowsEff - 1 - (outputRows - 1) * stride - 1 - padding_top;
  const TensorIndex padding_right = inputCols + kernelColsEff - 1 - (outputCols - 1) * stride - 1 - padding_left;

  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The output_backward has dimensions out_depth X out_rows X out_cols X OTHERS
  // When we extract the image patches from output_backward (with input as the
  // kernel), it will have dimensions
  //  (out_depth) X (input_rows * input_cols) X (kernel_rows * kernel_cols) X OTHERS
  DSizes<TensorIndex, 4> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelFilters;
    pre_contract_dims[1] = inputRows * inputCols;
    pre_contract_dims[2] = kernelRows * kernelCols;
    pre_contract_dims[3] = 1;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[3] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[3] = kernelFilters;
    pre_contract_dims[2] = inputRows * inputCols;
    pre_contract_dims[1] = kernelRows * kernelCols;
    pre_contract_dims[0] = 1;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // The input has dimensions in_depth X (input_rows * input_cols) X OTHERS
  DSizes<TensorIndex, 3> input_dims;
  if (isColMajor) {
    input_dims[0] = kernelChannels;
    input_dims[1] = inputRows * inputCols;
    input_dims[2] = 1;
    for (int i = 3; i < NumDims; ++i) {
      input_dims[2] *= in.dimension(i);
    }
    eigen_assert(input_dims[2] == pre_contract_dims[3]);
  } else {
    input_dims[2] = kernelChannels;
    input_dims[1] = inputRows * inputCols;
    input_dims[0] = 1;
    for (int i = 0; i < NumDims - 3; ++i) {
      input_dims[0] *= in.dimension(i);
    }
    eigen_assert(input_dims[0] == pre_contract_dims[0]);
  }

  // We will contract along dimensions (1, 2) in in and (1, 3) in out, if
  // this is col-major.
  // For row-major, it's dimensions (0, 1) in in and (0, 2) in out.
  array<IndexPair<TensorIndex>, 2> contract_dims;
  if (isColMajor) {
    // col-major: in.contract(output.patches)
    contract_dims[0] = IndexPair<TensorIndex>(1, 1);
    contract_dims[1] = IndexPair<TensorIndex>(2, 3);
  } else {
    // row-major: output.patches.contract(in)
    contract_dims[0] = IndexPair<TensorIndex>(0, 0);
    contract_dims[1] = IndexPair<TensorIndex>(2, 1);
  }

  // After the contraction, the kernel will have dimension
  // in_depth X out_depth X kernel_rows X kernel_cols
  // We will need to shuffle the first two dimensions and reverse the latter
  // two dimensions.
  // The end shape is
  // out_depth X in_shape X kernel_rows X kernel_cols

  // This is the shape of the kernel *before* the shuffling.
  DSizes<TensorIndex, 4> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelChannels;
    kernel_dims[1] = kernelFilters;
    kernel_dims[2] = kernelRows;
    kernel_dims[3] = kernelCols;
  } else {
    kernel_dims[0] = kernelCols;
    kernel_dims[1] = kernelRows;
    kernel_dims[2] = kernelFilters;
    kernel_dims[3] = kernelChannels;
  }

  array<TensorIndex, 4> kernel_shuffle;
  if (isColMajor) {
    kernel_shuffle[0] = 1;
    kernel_shuffle[1] = 0;
    kernel_shuffle[2] = 2;
    kernel_shuffle[3] = 3;
  } else {
    kernel_shuffle[0] = 0;
    kernel_shuffle[1] = 1;
    kernel_shuffle[2] = 3;
    kernel_shuffle[3] = 2;
  }

  array<bool, 4> kernel_reverse;
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

  return choose(Cond<internal::traits<Input>::Layout == ColMajor>(),
                input.reshape(input_dims).contract(output_backward.extract_image_patches(inputRows, inputCols, in_stride, in_stride, 1, 1, stride, stride, padding_top, padding_bottom, padding_left, padding_right, 0).reshape(pre_contract_dims).reshape(pre_contract_dims), contract_dims).reshape(kernel_dims).reverse(kernel_reverse).shuffle(kernel_shuffle),
                output_backward.extract_image_patches(inputRows, inputCols, in_stride, in_stride, 1, 1, stride, stride, padding_top, padding_bottom, padding_left, padding_right, 0).reshape(pre_contract_dims).reshape(pre_contract_dims).contract(input.reshape(input_dims), contract_dims).reshape(kernel_dims).reverse(kernel_reverse).shuffle(kernel_shuffle));
}

} // end namespace Eigen

#endif // EIGEN_CXX11_NEURAL_NETWORKS_BACKWARD_SPATIAL_CONVOLUTIONS_H
