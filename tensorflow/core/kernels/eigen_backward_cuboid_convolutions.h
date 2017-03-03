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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_patch_3d.h"

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
                IndexPair<typename internal::traits<OutputBackward>::Index>, 2>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             3>,
                const TensorReverseOp<const array<bool, 5>, const Kernel> >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             3>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const OutputBackward> > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<OutputBackward>::Index,
                     internal::traits<OutputBackward>::NumDimensions>,
        const TensorContractionOp<
            const array<
                IndexPair<typename internal::traits<OutputBackward>::Index>, 2>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             3>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const OutputBackward> >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             3>,
                const TensorReverseOp<const array<bool, 5>,
                                      const Kernel> > > > >::type
CuboidConvolutionBackwardInput(
    const Kernel& kernel, const OutputBackward& output_backward,
    typename internal::traits<OutputBackward>::Index inputPlanes,
    typename internal::traits<OutputBackward>::Index inputRows,
    typename internal::traits<OutputBackward>::Index inputCols,
    const DenseIndex stridePlanes = 1, const DenseIndex strideRows = 1,
    const DenseIndex strideCols = 1) {
  typedef typename internal::traits<OutputBackward>::Index TensorIndex;
  const TensorRef<const Tensor<typename internal::traits<Kernel>::Scalar,
                               internal::traits<Kernel>::NumDimensions,
                               internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);
  const TensorRef<
      const Tensor<typename internal::traits<OutputBackward>::Scalar,
                   internal::traits<OutputBackward>::NumDimensions,
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

  TensorIndex forward_pad_z, forward_pad_y, forward_pad_x;
  const TensorIndex size_z =
      Eigen::divup(inputPlanes, static_cast<TensorIndex>(stridePlanes));
  const TensorIndex size_y =
      Eigen::divup(inputRows, static_cast<TensorIndex>(strideRows));
  const TensorIndex size_x =
      Eigen::divup(inputCols, static_cast<TensorIndex>(strideCols));

  // Infer padding type.
  if (size_z == outputPlanes && size_y == outputRows && size_x == outputCols) {
    // SAME padding.
    const TensorIndex dz = numext::maxi<TensorIndex>(
        0, (size_z - 1) * stridePlanes + kernelPlanes - inputPlanes);
    const TensorIndex dy = numext::maxi<TensorIndex>(
        0, (size_y - 1) * strideRows + kernelRows - inputRows);
    const TensorIndex dx = numext::maxi<TensorIndex>(
        0, (size_x - 1) * strideCols + kernelCols - inputCols);

    forward_pad_z = dz / 2;
    forward_pad_y = dy / 2;
    forward_pad_x = dx / 2;
  } else {
    // VALID padding.
    forward_pad_z = 0;
    forward_pad_y = 0;
    forward_pad_x = 0;
  }
  const TensorIndex padding_ztop = kernelPlanes - 1 - forward_pad_z;
  const TensorIndex padding_top = kernelRows - 1 - forward_pad_y;
  const TensorIndex padding_left = kernelCols - 1 - forward_pad_x;

  const TensorIndex padding_zbottom = inputPlanes + kernelPlanes - 1 -
                                      (outputPlanes - 1) * stridePlanes - 1 -
                                      padding_ztop;
  const TensorIndex padding_bottom = inputRows + kernelRows - 1 -
                                     (outputRows - 1) * strideRows - 1 -
                                     padding_top;
  const TensorIndex padding_right = inputCols + kernelCols - 1 -
                                    (outputCols - 1) * strideCols - 1 -
                                    padding_left;

  eigen_assert(padding_ztop >= 0);
  eigen_assert(padding_zbottom >= 0);
  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The kernel has dimensions filters X channels X patch_planes X patch_rows X
  // patch_cols.
  // We need to reverse the kernel along the spatial dimensions.
  array<bool, 5> kernel_reverse;
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

  DSizes<TensorIndex, 3> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels;
    kernel_dims[2] = kernelRows * kernelCols * kernelPlanes;
  } else {
    kernel_dims[0] = kernelRows * kernelCols * kernelPlanes;
    kernel_dims[1] = kernelChannels;
    kernel_dims[2] = kernelFilters;
  }

  // The output_backward has dimensions out_depth X out_planes X out_rows X
  // out_cols X OTHERS
  // When we extract the image patches from output_backward, it will have
  // dimensions:
  //   out_depth X (patch_planes * patch_rows * patch_cols) X (input_planes *
  //   input_rows * input_cols * OTHERS)
  DSizes<TensorIndex, 3> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelFilters;
    pre_contract_dims[1] = kernelRows * kernelCols * kernelPlanes;
    pre_contract_dims[2] = inputRows * inputCols * inputPlanes;
    for (int i = 4; i < NumDims; ++i) {
      pre_contract_dims[2] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[2] = kernelFilters;
    pre_contract_dims[1] = kernelRows * kernelCols * kernelPlanes;
    pre_contract_dims[0] = inputRows * inputCols * inputPlanes;
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // We will contract along dimensions (0, 2) in kernel and (0, 1) in
  // output_backward, if this is col-major, and
  // dimensions (0, 2) in kernel and (1, 2) in output_backward, if this
  // row-major.
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

  DSizes<TensorIndex, NumDims> strides;
  for (int i = 0; i < NumDims; i++) {
    strides[i] = 1;
  }
  if (isColMajor) {
    strides[1] = stridePlanes;
    strides[2] = strideRows;
    strides[3] = strideCols;
  } else {
    strides[NumDims - 2] = stridePlanes;
    strides[NumDims - 3] = strideRows;
    strides[NumDims - 4] = strideCols;
  }

  return choose(
      Cond<internal::traits<OutputBackward>::Layout == ColMajor>(),
      kernel.reverse(kernel_reverse)
          .reshape(kernel_dims)
          .contract(output_backward
                        .extract_volume_patches(
                            kernelPlanes, kernelRows, kernelCols, 1, 1, 1,
                            stridePlanes, strideRows, strideCols, padding_ztop,
                            padding_zbottom, padding_top, padding_bottom,
                            padding_left, padding_right)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      output_backward
          .extract_volume_patches(kernelPlanes, kernelRows, kernelCols, 1, 1, 1,
                                  stridePlanes, strideRows, strideCols,
                                  padding_ztop, padding_zbottom, padding_top,
                                  padding_bottom, padding_left, padding_right)
          .reshape(pre_contract_dims)
          .contract(kernel.reverse(kernel_reverse).reshape(kernel_dims),
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
    internal::traits<OutputBackward>::Layout == ColMajor,
    const TensorShufflingOp<
        const array<typename internal::traits<OutputBackward>::Index, 5>,
        const TensorReverseOp<
            const array<bool, 5>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             5>,
                const TensorContractionOp<
                    const array<
                        IndexPair<typename internal::traits<Input>::Index>, 2>,
                    const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     3>,
                        const Input>,
                    const TensorReshapingOp<
                        const DSizes<
                            typename internal::traits<OutputBackward>::Index,
                            4>,
                        const TensorVolumePatchOp<
                            Dynamic, Dynamic, Dynamic,
                            const OutputBackward> > > > > >,
    const TensorShufflingOp<
        const array<typename internal::traits<OutputBackward>::Index, 5>,
        const TensorReverseOp<
            const array<bool, 5>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<OutputBackward>::Index,
                             5>,
                const TensorContractionOp<
                    const array<
                        IndexPair<typename internal::traits<Input>::Index>, 2>,
                    const TensorReshapingOp<
                        const DSizes<
                            typename internal::traits<OutputBackward>::Index,
                            4>,
                        const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                                  const OutputBackward> >,
                    const TensorReshapingOp<
                        const DSizes<typename internal::traits<Input>::Index,
                                     3>,
                        const Input> > > > > >::type
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
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<OutputBackward>::Scalar,
                   internal::traits<OutputBackward>::NumDimensions,
                   internal::traits<OutputBackward>::Layout, TensorIndex> >
      out(output_backward);

  EIGEN_STATIC_ASSERT(internal::traits<Input>::Layout ==
                          internal::traits<OutputBackward>::Layout,
                      YOU_MADE_A_PROGRAMMING_MISTAKE);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  static const int NumDims = internal::traits<Input>::NumDimensions;
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions ==
                          internal::traits<OutputBackward>::NumDimensions,
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

  const TensorIndex kernelFilters =
      isColMajor ? out.dimension(0) : out.dimension(NumDims - 1);
  const TensorIndex kernelChannels =
      isColMajor ? in.dimension(0) : in.dimension(NumDims - 1);

  TensorIndex forward_pad_z, forward_pad_y, forward_pad_x;
  const TensorIndex size_z =
      Eigen::divup(inputPlanes, static_cast<TensorIndex>(stridePlanes));
  const TensorIndex size_y =
      Eigen::divup(inputRows, static_cast<TensorIndex>(strideRows));
  const TensorIndex size_x =
      Eigen::divup(inputCols, static_cast<TensorIndex>(strideCols));

  // Infer padding type.
  if (size_z == outputPlanes && size_y == outputRows && size_x == outputCols) {
    // SAME padding.
    const TensorIndex dz = numext::maxi<TensorIndex>(
        0, (size_z - 1) * stridePlanes + kernelPlanes - inputPlanes);
    const TensorIndex dy = numext::maxi<TensorIndex>(
        0, (size_y - 1) * strideRows + kernelRows - inputRows);
    const TensorIndex dx = numext::maxi<TensorIndex>(
        0, (size_x - 1) * strideCols + kernelCols - inputCols);

    forward_pad_z = dz / 2;
    forward_pad_y = dy / 2;
    forward_pad_x = dx / 2;
  } else {
    // VALID padding.
    forward_pad_z = 0;
    forward_pad_y = 0;
    forward_pad_x = 0;
  }

  const TensorIndex padding_ztop = kernelPlanes - 1 - forward_pad_z;
  const TensorIndex padding_top = kernelRows - 1 - forward_pad_y;
  const TensorIndex padding_left = kernelCols - 1 - forward_pad_x;

  const TensorIndex padding_zbottom = inputPlanes + kernelPlanes - 1 -
                                      (outputPlanes - 1) * stridePlanes - 1 -
                                      padding_ztop;
  const TensorIndex padding_bottom = inputRows + kernelRows - 1 -
                                     (outputRows - 1) * strideRows - 1 -
                                     padding_top;
  const TensorIndex padding_right = inputCols + kernelCols - 1 -
                                    (outputCols - 1) * strideCols - 1 -
                                    padding_left;

  eigen_assert(padding_ztop >= 0);
  eigen_assert(padding_zbottom >= 0);
  eigen_assert(padding_top >= 0);
  eigen_assert(padding_left >= 0);
  eigen_assert(padding_bottom >= 0);
  eigen_assert(padding_right >= 0);

  // The output_backward has dimensions out_depth X out_plaens X out_rows X
  // out_cols X OTHERS
  // When we extract the image patches from output_backward (with input as the
  // kernel), it will have dimensions
  //  (out_depth) X (input_planes * input_rows * input_cols) X (kernel_planes *
  //  kernel_rows * kernel_cols) X OTHERS
  DSizes<TensorIndex, 4> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelFilters;
    pre_contract_dims[1] = inputRows * inputCols * inputPlanes;
    pre_contract_dims[2] = kernelRows * kernelCols * kernelPlanes;
    pre_contract_dims[3] = 1;
    for (int i = 4; i < NumDims; ++i) {
      pre_contract_dims[3] *= out.dimension(i);
    }
  } else {
    pre_contract_dims[3] = kernelFilters;
    pre_contract_dims[2] = inputRows * inputCols * inputPlanes;
    pre_contract_dims[1] = kernelRows * kernelCols * kernelPlanes;
    pre_contract_dims[0] = 1;
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_contract_dims[0] *= out.dimension(i);
    }
  }

  // The input has dimensions in_depth X (input_planes * input_rows *
  // input_cols) X OTHERS
  DSizes<TensorIndex, 3> input_dims;
  if (isColMajor) {
    input_dims[0] = kernelChannels;
    input_dims[1] = inputRows * inputCols * inputPlanes;
    input_dims[2] = 1;
    for (int i = 4; i < NumDims; ++i) {
      input_dims[2] *= in.dimension(i);
    }
    eigen_assert(input_dims[2] == pre_contract_dims[3]);
  } else {
    input_dims[2] = kernelChannels;
    input_dims[1] = inputRows * inputCols * inputPlanes;
    input_dims[0] = 1;
    for (int i = 0; i < NumDims - 4; ++i) {
      input_dims[0] *= in.dimension(i);
    }
    eigen_assert(input_dims[0] == pre_contract_dims[0]);
  }

  // We will contract along dimensions (1, 2) in and (1, 3) in out, if
  // this is col-major.
  // For row-major, it's dimensions (0, 1) in and (0, 2) in out.
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
  //   in_depth X out_depth X kernel_patches X kernel_rows X kernel_cols
  // We will need to shuffle the first two dimensions and reverse the spatial
  // dimensions.
  // The end shape is:
  //   out_depth X in_shape X kernel_planes X kernel_rows X kernel_cols

  // This is the shape of the kernel *before* the shuffling.
  DSizes<TensorIndex, 5> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelChannels;
    kernel_dims[1] = kernelFilters;
    kernel_dims[2] = kernelPlanes;
    kernel_dims[3] = kernelRows;
    kernel_dims[4] = kernelCols;
  } else {
    kernel_dims[0] = kernelCols;
    kernel_dims[1] = kernelRows;
    kernel_dims[2] = kernelPlanes;
    kernel_dims[3] = kernelFilters;
    kernel_dims[4] = kernelChannels;
  }

  // Flip filters and channels.
  array<TensorIndex, 5> kernel_shuffle;
  if (isColMajor) {
    kernel_shuffle[0] = 1;
    kernel_shuffle[1] = 0;
    kernel_shuffle[2] = 2;
    kernel_shuffle[3] = 3;
    kernel_shuffle[4] = 4;
  } else {
    kernel_shuffle[0] = 0;
    kernel_shuffle[1] = 1;
    kernel_shuffle[2] = 2;
    kernel_shuffle[3] = 4;
    kernel_shuffle[4] = 3;
  }

  // Reverse the spatial dimensions.
  array<bool, 5> kernel_reverse;
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

  DSizes<TensorIndex, NumDims> strides;
  for (int i = 0; i < NumDims; i++) {
    strides[i] = 1;
  }
  if (isColMajor) {
    strides[1] = stridePlanes;
    strides[2] = strideRows;
    strides[3] = strideCols;
  } else {
    strides[NumDims - 2] = stridePlanes;
    strides[NumDims - 3] = strideRows;
    strides[NumDims - 4] = strideCols;
  }
  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      input.reshape(input_dims)
          .contract(output_backward
                        .extract_volume_patches(
                            inputPlanes, inputRows, inputCols, 1, 1, 1,
                            stridePlanes, strideRows, strideCols,

                            padding_ztop, padding_zbottom, padding_top,
                            padding_bottom, padding_left, padding_right)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(kernel_dims)
          .reverse(kernel_reverse)
          .shuffle(kernel_shuffle),
      output_backward
          .extract_volume_patches(inputPlanes, inputRows, inputCols, 1, 1, 1,
                                  stridePlanes, strideRows, strideCols,
                                  padding_ztop, padding_zbottom, padding_top,
                                  padding_bottom, padding_left, padding_right)
          .reshape(pre_contract_dims)
          .contract(input.reshape(input_dims), contract_dims)
          .reshape(kernel_dims)
          .reverse(kernel_reverse)
          .shuffle(kernel_shuffle));
}

}  // end namespace Eigen

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_EIGEN_BACKWARD_CUBOID_CONVOLUTIONS_H_
