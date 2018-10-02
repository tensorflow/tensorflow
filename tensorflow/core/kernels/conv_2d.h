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

#ifndef TENSORFLOW_CORE_KERNELS_CONV_2D_H_
#define TENSORFLOW_CORE_KERNELS_CONV_2D_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
#include "tensorflow/core/kernels/eigen_spatial_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace functor {

// TODO(yangke): revisit these operations and in particular, see if we can
// combine all of them into just one operation without causing nvcc to
// timeout.
template <typename Device, typename T, int Dims, typename IndexType>
struct ShuffleAndReverse {
  void operator()(const Device& d,
                  typename TTypes<T, Dims, IndexType>::ConstTensor input,
                  const Eigen::DSizes<IndexType, Dims>& order,
                  const Eigen::array<bool, Dims>& reverse_dims,
                  typename TTypes<T, Dims, IndexType>::Tensor output) {
    output.device(d) = input.shuffle(order).reverse(reverse_dims);
  }
};

template <typename Device, typename T, int Dims, typename IndexType>
struct InflatePadAndShuffle {
  void operator()(
      const Device& d, typename TTypes<T, Dims, IndexType>::ConstTensor input,
      const Eigen::DSizes<IndexType, Dims>& strides,
      const Eigen::array<Eigen::IndexPair<IndexType>, Dims>& pad_dims,
      const Eigen::DSizes<IndexType, Dims>& order,
      typename TTypes<T, Dims, IndexType>::Tensor output) {
    output.device(d) = input.inflate(strides).pad(pad_dims).shuffle(order);
  }
};

template <typename Device, typename Input, typename Filter, typename Output>
void SpatialConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int row_stride, int col_stride,
                            int row_dilation, int col_dilation,
                            const Eigen::PaddingType& padding) {
  // Need to swap row/col when calling Eigen.
  output.device(d) =
      Eigen::SpatialConvolution(input, filter, col_stride, row_stride, padding,
                                col_dilation, row_dilation);
}

template <typename Device, typename T>
struct SpatialConvolution {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, int row_dilation, int col_dilation,
                  const Eigen::PaddingType& padding) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           row_dilation, col_dilation, padding);
  }
};

template <typename Device>
struct SpatialConvolution<Device, Eigen::half> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding) {
    output.device(d) =
        Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                  col_stride, row_stride, padding, col_dilation,
                                  row_dilation)
            .cast<Eigen::half>();
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardInput {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation) {
    // Need to swap row/col when calling Eigen.
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
        kernel, output_backward, input_backward.dimension(2),
        input_backward.dimension(1), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardFilter {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor kernel_backward,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation) {
    // Need to swap row/col when calling Eigen.
    kernel_backward.device(d) = Eigen::SpatialConvolutionBackwardKernel(
        input, output_backward, kernel_backward.dimension(1),
        kernel_backward.dimension(0), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};

// TODO(vrv): Figure out how to use the MatMulFunctor in matmul_op.h.
// My initial attempt to do this compiled but failed in the pytest
// due to a swigdeps error.
template <typename Device, typename T>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename TTypes<T, 2>::Tensor out,
      typename TTypes<T, 2>::ConstTensor in0,
      typename TTypes<T, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair) {
    out.device(d) = in0.contract(in1, dim_pair);
  }
};

// Shuffles a filter tensor from TensorFlow format HWIO to dst_filter_format.
//
// Note: Currently OIHW is the only supported destination format. Support for
// OHWI format will be added in a follow-up change.
template <typename Device, typename T, typename IndexType, int NDIMS>
struct TransformFilter {
  void operator()(const Device& d, FilterTensorFormat dst_filter_format,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out) {
    // Merge the spatial dimensions together to speed up the shuffle operation.
    Eigen::DSizes<IndexType, 3> merged_dims;
    merged_dims[0] = in.dimension(0);  // spatial dimensions
    for (int i = 1; i < NDIMS - 2; ++i) {
      merged_dims[0] *= in.dimension(i);
    }
    merged_dims[1] = in.dimension(NDIMS - 2);  // input filters
    merged_dims[2] = in.dimension(NDIMS - 1);  // output filters

    CHECK(dst_filter_format == FORMAT_OIHW)
        << "Unsupported destination filter format: "
        << ToString(dst_filter_format);
    // Source filter format is FORMAT_HWIO and spatial dimensions HW are merged
    // in the beginning.
    Eigen::DSizes<IndexType, 3> shuffling_perm =
        Eigen::DSizes<IndexType, 3>(2, 1, 0);

    Eigen::DSizes<IndexType, NDIMS> expanded_dims;
    int out_index = 0;
    for (int merged_dim = 0; merged_dim < merged_dims.rank(); ++merged_dim) {
      if (shuffling_perm[merged_dim] == 0) {
        for (int spatial_dim = 0; spatial_dim < NDIMS - 2; ++spatial_dim) {
          expanded_dims[out_index++] = in.dimension(spatial_dim);
        }
      } else {
        constexpr int kLastSpatialDim = NDIMS - 3;
        expanded_dims[out_index++] =
            in.dimension(kLastSpatialDim + shuffling_perm[merged_dim]);
      }
    }

    out.device(d) =
        in.reshape(merged_dims).shuffle(shuffling_perm).reshape(expanded_dims);
  }
};

template <typename Device, typename T, typename IndexType>
struct TransformDepth {
  void operator()(const Device& d,
                  typename TTypes<T, 4, IndexType>::ConstTensor in,
                  const Eigen::DSizes<IndexType, 4>& shuffle,
                  typename TTypes<T, 4, IndexType>::Tensor out) {
    Eigen::DSizes<IndexType, 3> merged_dims;
    Eigen::DSizes<IndexType, 4> expanded_dims;
    Eigen::DSizes<IndexType, 3> new_shuffle;

    // Merge dimensions that won't be shuffled together to speed things up.
    if (shuffle[1] == 2 && shuffle[2] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = shuffle[0];
      new_shuffle[1] = 2;
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(shuffle[0]);
      expanded_dims[1] = in.dimension(2);
      expanded_dims[2] = in.dimension(3);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 2 && shuffle[1] == 3) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1);
      merged_dims[2] = in.dimension(2) * in.dimension(3);
      new_shuffle[0] = 2;
      new_shuffle[1] = shuffle[2];
      new_shuffle[2] = shuffle[3];
      expanded_dims[0] = in.dimension(2);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(shuffle[2]);
      expanded_dims[3] = in.dimension(shuffle[3]);
    } else if (shuffle[0] == 0 && shuffle[1] == 3 && shuffle[2] == 1 &&
               shuffle[3] == 2) {
      merged_dims[0] = in.dimension(0);
      merged_dims[1] = in.dimension(1) * in.dimension(2);
      merged_dims[2] = in.dimension(3);
      new_shuffle[0] = 0;
      new_shuffle[1] = 2;
      new_shuffle[2] = 1;
      expanded_dims[0] = in.dimension(0);
      expanded_dims[1] = in.dimension(3);
      expanded_dims[2] = in.dimension(1);
      expanded_dims[3] = in.dimension(2);
    } else {
      assert(false && "unexpected shuffle");
    }

    out.device(d) =
        in.reshape(merged_dims).shuffle(new_shuffle).reshape(expanded_dims);
  }
};

template <typename Device, typename T, typename IndexType, int NDIMS>
struct PadInput {
  void operator()(const Device& d,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  const std::array<int, NDIMS - 2>& padding_left,
                  const std::array<int, NDIMS - 2>& padding_right,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out,
                  TensorFormat format) {
    Eigen::array<Eigen::IndexPair<IndexType>, NDIMS> padding;
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'N')] = {0, 0};
    for (int i = 0; i < NDIMS - 2; ++i) {
      padding[GetTensorDimIndex<NDIMS - 2>(format, '0' + i)] = {
          padding_left[i], padding_right[i]};
    }
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'C')] = {0, 0};
    out.device(d) = in.pad(padding);
  }
};

// Converts a tensor from:
//   [batch, <spatial>, filters]
// to:
//   [batch, filters, <spatial>]
template <typename Device, typename T, int NDIMS>
struct NHWCToNCHW {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

// Converts a tensor from:
//   [batch, filters, <spatial>]
// to:
//   [batch, <spatial>, filters]
template <typename Device, typename T, int NDIMS>
struct NCHWToNHWC {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

// Converts a tensor from:
//   [dim0, dim1, dim2]
// to:
//   [dim0, dim2, dim1]
template <typename Device, typename T, bool conjugate = false>
struct SwapDimension1And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& input_dims, T* out);
};

// Converts a tensor from:
//   [dim0, dim1, dim2]
// to:
//   [dim2, dim1, dim0]
template <typename Device, typename T, bool conjugate = false>
struct SwapDimension0And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64>& input_dims, T* out);
};

// Transforms back filter from OIHW to HWOI format to reverse effect of
// TransformFilter above.
// TODO(hinsu): Support reverse transformation from filter format OHWI as well.
template <typename Device, typename T, int NDIMS>
struct ReverseTransformFilter {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

}  // namespace functor

template <class T>
class ConvAlgorithmMap;

template <>
class ConvAlgorithmMap<Eigen::ThreadPoolDevice> {};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_2D_H_
