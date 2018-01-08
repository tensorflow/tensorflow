/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Copied from tensorflow/core/kernels/eigen_spatial_convolutions.h.
// TODO(petewarden) - move this to a common location in Eigen itself.

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_

#define EIGEN_USE_CUSTOM_THREAD_POOL
#define EIGEN_USE_THREADS

// NOTE: Eigen is slightly different internally and externally. We need to
// hack the unsupported/Eigen/CXX11/Tensor header instantiation macros at
// specific places, so we need two copies of the hacked file, one for
// internal and one for external.
// If you have trouble simply undef out the reducer macro e.g.
// TFLITE_REDUCE_INSTANTIATIONS_GOOGLE, but be aware this will make
// the binary much bigger!
#define TFLITE_REDUCE_INSTANTIATIONS_OPEN_SOURCE
#define Eigen EigenForTFLite
#if defined(TFLITE_REDUCE_INSTANTIATIONS_GOOGLE)
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_tensor_reduced_instantiations_google.h"
#elif defined(TFLITE_REDUCE_INSTANTIATIONS_OPEN_SOURCE)
#include "tensorflow/contrib/lite/kernels/internal/optimized/eigen_tensor_reduced_instantiations_oss.h"
#else
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#endif


namespace Eigen {

/** SpatialConvolution
 * \ingroup CXX11_NeuralNetworks_Module
 *
 * \brief Applies a 2D convolution over a multichannel input image.
 *
 * The input parameter is expected to be a tensor with a rank of 3 or more
 * (channels, height, width, and optionally others)
 * The kernel parameter is expected to be a 4D tensor (filters, channels,
 * kernel_height, kernel_width)
 * The input and the kernel must both be in col-major layout. The result will
 * also be in col-major layout.
 *
 * If col_in_stride, row_in_stride > 1, then applies convolution with holes
 * (aka atrous convolution), sampling every col_in_stride, row_in_stride input
 * pixels.
 *
 * The result can be assigned to a tensor of rank equal to the rank of the
 * input. The dimensions of the result will be filters, height, width (and
 * others if applicable).
 *
 * It is possible to swap the order of the width and height dimensions provided
 * that the same order is used in the input, the kernel, and the output.
 *
 */
template <typename Input, typename Kernel>
EIGEN_DEVICE_FUNC
    EIGEN_ALWAYS_INLINE static const typename internal::conditional<
        internal::traits<Input>::Layout == ColMajor,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic,
                                             const Input> > > >,
        TensorReshapingOp<
            const DSizes<typename internal::traits<Input>::Index,
                         internal::traits<Input>::NumDimensions>,
            const TensorContractionOp<
                const array<IndexPair<typename internal::traits<Input>::Index>,
                            1>,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
                const TensorReshapingOp<
                    const DSizes<typename internal::traits<Input>::Index, 2>,
                    const Kernel> > > >::type
    SpatialConvolution(const Input& input, const Kernel& kernel,
                       const DenseIndex row_stride = 1,
                       const DenseIndex col_stride = 1,
                       const PaddingType padding_type = PADDING_SAME,
                       const DenseIndex row_in_stride = 1,
                       const DenseIndex col_in_stride = 1) {
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
  const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  const int NumDims = internal::traits<Input>::NumDimensions;

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

  const DenseIndex kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const DenseIndex kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID:
      out_height = numext::ceil((InputRows - kernelRowsEff + 1.f) /
                                static_cast<float>(row_stride));
      out_width = numext::ceil((InputCols - kernelColsEff + 1.f) /
                               static_cast<float>(col_stride));
      break;
    case PADDING_SAME:
      out_height = numext::ceil(InputRows / static_cast<float>(row_stride));
      out_width = numext::ceil(InputCols / static_cast<float>(col_stride));
      break;
    default:
      // Initialize unused variables to avoid a compiler warning
      out_height = 0;
      out_width = 0;
      eigen_assert(false && "unexpected padding");
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }
  // TODO(yangke): choose() is defined in TensorContraction.h -- consider
  // moving it to somewhere more "common".
  return
      input
          .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                                 row_in_stride, col_in_stride, padding_type)
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims);
}

}  // end namespace Eigen

// clang-format on

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_KERNELS_INTERNAL_OPTIMIZED_EIGEN_SPATIAL_CONVOLUTIONS_H_
