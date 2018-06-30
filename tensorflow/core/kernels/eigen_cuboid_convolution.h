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

#ifndef TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_
#define TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/eigen_volume_patch.h"

namespace Eigen {

/** CuboidConvolution
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a 3D convolution over a multichannel input voxel block.
 *
 * The input parameter is expected to be a tensor with a rank of 4 or more
 * (channels, depth, height, width, and optionally others).
 * The kernel parameter is expected to be a 5D tensor (filters, channels,
 * kernel_depth, kernel_height, kernel_width).
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, depth, height, width
 * (and others if applicable).
 *
 * The input and kernel have to be in the same layout, and both row-major and
 * col-major are supported. The shapes given above are for col-major layout.
 * For row-major, all dimensions should be reversed.
 *
 * It is possible to swap the order of the depth, width, and height dimensions
 * provided that the same order is used in the input, the kernel, and the
 * output.
 */
template <typename Input, typename Kernel>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<Input>::Layout == ColMajor,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const Input> > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorVolumePatchOp<Dynamic, Dynamic, Dynamic,
                                          const Input> >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel> > > >::type
CuboidConvolution(const Input& input, const Kernel& kernel,
                  const DenseIndex stridePlanes = 1,
                  const DenseIndex strideRows = 1,
                  const DenseIndex strideCols = 1,
                  const PaddingType padding_type = PADDING_SAME) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result.
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[4];
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[3];

  // Spatial size of the kernel.
  const TensorIndex kernelDepth =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[4] : kern.dimensions()[0];

  if (isColMajor) {
    eigen_assert(kernelChannels == in.dimension(0));
  } else {
    eigen_assert(kernelChannels == in.dimension(NumDims - 1));
  }

  const TensorIndex inputPlanes =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex inputRows =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);
  const TensorIndex inputCols =
      isColMajor ? in.dimension(3) : in.dimension(NumDims - 4);

  TensorIndex out_depth;
  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID:
      out_depth = Eigen::divup(inputPlanes - kernelDepth + 1,
                               static_cast<TensorIndex>(stridePlanes));
      out_height = Eigen::divup(inputRows - kernelRows + 1,
                                static_cast<TensorIndex>(strideRows));
      out_width = Eigen::divup(inputCols - kernelCols + 1,
                               static_cast<TensorIndex>(strideCols));
      break;
    case PADDING_SAME:
      out_depth =
          Eigen::divup(inputPlanes, static_cast<TensorIndex>(stridePlanes));
      out_height =
          Eigen::divup(inputRows, static_cast<TensorIndex>(strideRows));
      out_width = Eigen::divup(inputCols, static_cast<TensorIndex>(strideCols));
      break;
    default:
      out_depth = 0;
      out_height = 0;
      out_width = 0;
      eigen_assert(false && "unexpected padding");
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelDepth * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelDepth * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }

  // Molds the output of the patch extraction result into a 2D tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] =
        kernelChannels * kernelDepth * kernelRows * kernelCols;
    pre_contract_dims[1] = out_depth * out_height * out_width;
    for (int i = 4; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] =
        kernelChannels * kernelDepth * kernelRows * kernelCols;
    pre_contract_dims[0] = out_depth * out_height * out_width;
    for (int i = 0; i < NumDims - 4; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  // Molds the output of the contraction into the shape expected by the user
  // (assuming ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output depth
  // - 3nd dim: output height
  // - 4rd dim: output width
  // - 5th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_depth;
    post_contract_dims[2] = out_height;
    post_contract_dims[3] = out_width;
    for (int i = 4; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_depth;
    post_contract_dims[NumDims - 3] = out_height;
    post_contract_dims[NumDims - 4] = out_width;
    for (int i = 0; i < NumDims - 4; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      kernel.reshape(kernel_dims)
          .contract(input
                        .extract_volume_patches(
                            kernelDepth, kernelRows, kernelCols, stridePlanes,
                            strideRows, strideCols, padding_type)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      input
          .extract_volume_patches(kernelDepth, kernelRows, kernelCols,
                                  stridePlanes, strideRows, strideCols,
                                  padding_type)
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims));
}

}  // end namespace Eigen

#endif  // TENSORFLOW_CORE_KERNELS_EIGEN_CUBOID_CONVOLUTION_H_
