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

#include "absl/strings/string_view.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_backward_spatial_convolutions.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/tsl/framework/convolution/eigen_spatial_convolutions.h"

// Returns true if TF_CONV2D_USE_FP16_ACCUMULATE == 1, false otherwise.
static bool Conv2dUseFp16Accumulate() {
  static bool use_fp16_accumulate = []() {
    const char* env = std::getenv("TF_CONV2D_USE_FP16_ACCUMULATE");
    return (env != nullptr) && (absl::string_view(env) == "1");
  }();
  return use_fp16_accumulate;
}

namespace tensorflow {
namespace functor {

template <typename Device, typename Input, typename Filter, typename Output,
          typename OutputKernel>
void SpatialConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int row_stride, int col_stride,
                            int row_dilation, int col_dilation,
                            const Eigen::PaddingType& padding,
                            const OutputKernel& output_kernel,
                            int padding_top = 0, int padding_bottom = 0,
                            int padding_left = 0, int padding_right = 0) {
  // Need to swap row/col, padding_top/padding_left, and
  // padding_bottom/padding_right when calling Eigen. Eigen expects the tensor
  // in NWHC format, but the tensor given is in NHWC.
  output.device(d) = Eigen::SpatialConvolution(
      input, filter, col_stride, row_stride, padding, col_dilation,
      row_dilation, output_kernel, padding_left, padding_right, padding_top,
      padding_bottom);
}

// TODO(ezhulenev): Non-templated `operator()` are required by explicit template
// instantiations for the GPU device. However they are almost certainly not used
// in any of the kernel implementation. Check if they can be removed.
template <typename Device, typename T,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct SpatialConvolution {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, int row_dilation, int col_dilation,
                  const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           row_dilation, col_dilation, padding, output_kernel);
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(d, output, input, filter, row_stride, col_stride,
                           row_dilation, col_dilation, padding, output_kernel);
  }

  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int row_stride,
                  int col_stride, int row_dilation, int col_dilation,
                  int padding_top, int padding_bottom, int padding_left,
                  int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(
        d, output, input, filter, row_stride, col_stride, row_dilation,
        col_dilation, Eigen::PaddingType::PADDING_VALID, output_kernel,
        padding_top, padding_bottom, padding_left, padding_right);
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    SpatialConvolutionFunc(
        d, output, input, filter, row_stride, col_stride, row_dilation,
        col_dilation, Eigen::PaddingType::PADDING_VALID, output_kernel,
        padding_top, padding_bottom, padding_left, padding_right);
  }
};

template <typename Device, typename OutputKernel>
struct SpatialConvolution<Device, Eigen::half, OutputKernel> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride, padding, col_dilation,
          row_dilation, output_kernel);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                    col_stride, row_stride, padding,
                                    col_dilation, row_dilation, output_kernel)
              .template cast<Eigen::half>();
    }
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride, padding, col_dilation,
          row_dilation, output_kernel);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(input.template cast<float>(),
                                    filter.template cast<float>(), col_stride,
                                    row_stride, padding, col_dilation,
                                    row_dilation, output_kernel)
              .template cast<Eigen::half>();
    }
  }

  void operator()(const Device& d,
                  typename TTypes<Eigen::half, 4>::Tensor output,
                  typename TTypes<Eigen::half, 4>::ConstTensor input,
                  typename TTypes<Eigen::half, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride,
          Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
          output_kernel, padding_left, padding_right, padding_top,
          padding_bottom);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(
              input.cast<float>(), filter.cast<float>(), col_stride, row_stride,
              Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
              output_kernel, padding_left, padding_right, padding_top,
              padding_bottom)
              .template cast<Eigen::half>();
    }
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      output.device(d) = Eigen::SpatialConvolution(
          input, filter, col_stride, row_stride,
          Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
          output_kernel, padding_left, padding_right, padding_top,
          padding_bottom);
    } else {
      output.device(d) =
          Eigen::SpatialConvolution(
              input.template cast<float>(), filter.template cast<float>(),
              col_stride, row_stride, Eigen::PaddingType::PADDING_VALID,
              col_dilation, row_dilation, output_kernel, padding_left,
              padding_right, padding_top, padding_bottom)
              .template cast<Eigen::half>();
    }
  }
};

// Use float32 accumulation for bfloat16 to deal with precision accumulation
// issues.
template <typename Device, typename OutputKernel>
struct SpatialConvolution<Device, Eigen::bfloat16, OutputKernel> {
  void operator()(const Device& d,
                  typename TTypes<Eigen::bfloat16, 4>::Tensor output,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor input,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(input.cast<float>(), filter.cast<float>(),
                                  col_stride, row_stride, padding, col_dilation,
                                  row_dilation, output_kernel)
            .template cast<Eigen::bfloat16>();
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, const Eigen::PaddingType& padding,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(input.template cast<float>(),
                                  filter.template cast<float>(), col_stride,
                                  row_stride, padding, col_dilation,
                                  row_dilation, output_kernel)
            .template cast<Eigen::bfloat16>();
  }

  void operator()(const Device& d,
                  typename TTypes<Eigen::bfloat16, 4>::Tensor output,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor input,
                  typename TTypes<Eigen::bfloat16, 4>::ConstTensor filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(
            input.cast<float>(), filter.cast<float>(), col_stride, row_stride,
            Eigen::PaddingType::PADDING_VALID, col_dilation, row_dilation,
            output_kernel, padding_left, padding_right, padding_top,
            padding_bottom)
            .template cast<Eigen::bfloat16>();
  }

  template <typename Input, typename Filter, typename Output>
  void operator()(const Device& d, Output output, Input input, Filter filter,
                  int row_stride, int col_stride, int row_dilation,
                  int col_dilation, int padding_top, int padding_bottom,
                  int padding_left, int padding_right,
                  const OutputKernel& output_kernel = OutputKernel()) {
    output.device(d) =
        Eigen::SpatialConvolution(
            input.template cast<float>(), filter.template cast<float>(),
            col_stride, row_stride, Eigen::PaddingType::PADDING_VALID,
            col_dilation, row_dilation, output_kernel, padding_left,
            padding_right, padding_top, padding_bottom)
            .template cast<Eigen::bfloat16>();
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardInputFunc {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation) {
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
        filter, output_backward, input_backward.dimension(2),
        input_backward.dimension(1), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};

// GPU version requires all tensors to be indexable by int32.
template <typename T>
struct SpatialConvolutionBackwardInputFunc<Eigen::GpuDevice, T> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation) {
    To32Bit(input_backward).device(d) = Eigen::SpatialConvolutionBackwardInput(
        To32Bit(filter), To32Bit(output_backward), input_backward.dimension(2),
        input_backward.dimension(1), col_stride, row_stride, col_dilation,
        row_dilation);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex padded_cols, Eigen::DenseIndex padded_rows,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation, Eigen::DenseIndex pad_left,
                  Eigen::DenseIndex pad_top) {
    // We have to slice the result of a spatial convolution backward
    // input, before assigning it to the `input_backward` to remove padding.
    //
    // TODO(ezhulenev): Pass explicit paddings to Eigen and do not materialize
    // intermediate result in memory before slicing.
    input_backward.device(d) =
        Eigen::SpatialConvolutionBackwardInput(
            filter, output_backward, padded_cols, padded_rows, col_stride,
            row_stride, col_dilation, row_dilation)
            .eval()
            .slice(Eigen::DSizes<Eigen::DenseIndex, 4>{0, pad_left, pad_top, 0},
                   input_backward.dimensions());
  }
};

// GPU version requires all tensors to be indexable by int32.
template <typename T>
struct SpatialConvolutionBackwardInputWithExplicitPaddingFunc<Eigen::GpuDevice,
                                                              T> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  Eigen::DenseIndex padded_cols, Eigen::DenseIndex padded_rows,
                  Eigen::DenseIndex col_stride, Eigen::DenseIndex row_stride,
                  Eigen::DenseIndex col_dilation,
                  Eigen::DenseIndex row_dilation, Eigen::DenseIndex pad_left,
                  Eigen::DenseIndex pad_top) {
    To32Bit(input_backward).device(d) =
        Eigen::SpatialConvolutionBackwardInput(
            To32Bit(filter), To32Bit(output_backward), padded_cols, padded_rows,
            col_stride, row_stride, col_dilation, row_dilation)
            .eval()
            .slice(Eigen::DSizes<Eigen::DenseIndex, 4>{0, pad_left, pad_top, 0},
                   input_backward.dimensions());
  }
};

// TODO(vrv): Figure out how to use the MatMulFunctor in matmul_op.h.
// My initial attempt to do this compiled but failed in the pytest
// due to a swigdeps error.
template <typename Device, typename T,
          typename OutputKernel = const Eigen::NoOpOutputKernel>
struct MatMulConvFunctor {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename TTypes<T, 2>::Tensor out,
      typename TTypes<T, 2>::ConstTensor in0,
      typename TTypes<T, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    out.device(d) = in0.contract(in1, dim_pair, output_kernel);
  }
};

// Use float32 accumulation for float16 by default to deal with precision
// accumulation issues.  To enable float16 accumulation, set the environment
// variable TF_CONV2D_USE_FP16_ACCUMULATE.
template <typename Device, typename OutputKernel>
struct MatMulConvFunctor<Device, Eigen::half, OutputKernel> {
  // Computes on device "d": out = in0 * in1, where * is matrix
  // multiplication.
  void operator()(
      const Device& d, typename TTypes<Eigen::half, 2>::Tensor out,
      typename TTypes<Eigen::half, 2>::ConstTensor in0,
      typename TTypes<Eigen::half, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    if (Conv2dUseFp16Accumulate()) {
      out.device(d) = in0.contract(in1, dim_pair, output_kernel);
    } else {
      out.device(d) =
          in0.cast<float>()
              .contract(in1.template cast<float>(), dim_pair, output_kernel)
              .template cast<Eigen::half>();
    }
  }
};

// Use float32 accumulation for bfloat16 to deal with precision accumulation
// issues.
template <typename Device, typename OutputKernel>
struct MatMulConvFunctor<Device, Eigen::bfloat16, OutputKernel> {
  void operator()(
      const Device& d, typename TTypes<Eigen::bfloat16, 2>::Tensor out,
      typename TTypes<Eigen::bfloat16, 2>::ConstTensor in0,
      typename TTypes<Eigen::bfloat16, 2>::ConstTensor in1,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      const OutputKernel& output_kernel = OutputKernel()) {
    out.device(d) = in0.cast<float>()
                        .contract(in1.cast<float>(), dim_pair, output_kernel)
                        .template cast<Eigen::bfloat16>();
  }
};

// Shuffles a filter tensor from TensorFlow format HWIO to dst_filter_format.
//
// Note: Currently supports OIHW and OHWI destination formats.
template <typename Device, typename T, typename IndexType, int NDIMS>
struct TransformFilter {
  void operator()(const Device& d, FilterTensorFormat dst_filter_format,
                  typename TTypes<T, NDIMS, IndexType>::ConstTensor in,
                  typename TTypes<T, NDIMS, IndexType>::Tensor out) {
    // NOTE: Source filter format is always HWIO.
    Eigen::DSizes<IndexType, NDIMS - 2> spatial_dims;
    for (int i = 0; i < spatial_dims.rank(); ++i) {
      spatial_dims[i] = in.dimension(i);
    }

    // Merge the spatial dimensions together to speed up the shuffle operation.
    Eigen::DSizes<IndexType, 3> merged_dims;
    merged_dims[0] = spatial_dims.TotalSize();  // product of spatial dims [H*W]
    merged_dims[1] = in.dimension(NDIMS - 2);   // input filters           [I]
    merged_dims[2] = in.dimension(NDIMS - 1);   // output filters          [O]

    // Shuffle tensor with merged spatial dimensions.
    Eigen::DSizes<IndexType, 3> shuffling_perm;
    // Expand shuffled tensor into final dimensions.
    Eigen::DSizes<IndexType, NDIMS> expanded_dims;

    if (dst_filter_format == FORMAT_OIHW) {
      shuffling_perm = Eigen::DSizes<IndexType, 3>(2, 1, 0);

      expanded_dims[0] = merged_dims[2];  // [O]
      expanded_dims[1] = merged_dims[1];  // [I]
      for (int i = 0; i < spatial_dims.rank(); ++i) {
        expanded_dims[2 + i] = spatial_dims[i];
      }

    } else if (dst_filter_format == FORMAT_OHWI) {
      shuffling_perm = Eigen::DSizes<IndexType, 3>(2, 0, 1);

      expanded_dims[0] = merged_dims[2];          // [O]
      expanded_dims[NDIMS - 1] = merged_dims[1];  // [I]
      for (int i = 0; i < spatial_dims.rank(); ++i) {
        expanded_dims[1 + i] = spatial_dims[i];
      }

    } else {
      DCHECK(false) << "Unsupported destination filter format: "
                    << ToString(dst_filter_format);
    }

    out.device(d) =
        in.reshape(merged_dims).shuffle(shuffling_perm).reshape(expanded_dims);
  }
};

// TODO This functor is not used anywhere and should be removed,
// but it defines some eigen templates that are referenced in other kernels.
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
                  TensorFormat format, const T& padding_value) {
    Eigen::array<Eigen::IndexPair<IndexType>, NDIMS> padding;
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'N')] = {0, 0};
    for (int i = 0; i < NDIMS - 2; ++i) {
      padding[GetTensorDimIndex<NDIMS - 2>(format, '0' + i)] = {
          padding_left[i], padding_right[i]};
    }
    padding[GetTensorDimIndex<NDIMS - 2>(format, 'C')] = {0, 0};
    out.device(d) = in.pad(padding, padding_value);
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
                  const gtl::ArraySlice<int64_t>& input_dims, T* out);
};

// Converts a tensor from:
//   [dim0, dim1, dim2]
// to:
//   [dim2, dim1, dim0]
template <typename Device, typename T, bool conjugate = false>
struct SwapDimension0And2InTensor3 {
  void operator()(const Device& d, const T* in,
                  const gtl::ArraySlice<int64_t>& input_dims, T* out);
};

// Transforms back filter from OIHW or OHWI to HWOI format to reverse effect of
// TransformFilter above.
template <typename Device, typename T, int NDIMS>
struct ReverseTransformFilter {
  void operator()(const Device& d, FilterTensorFormat src_filter_format,
                  typename TTypes<T, NDIMS>::ConstTensor in,
                  typename TTypes<T, NDIMS>::Tensor out);
};

}  // namespace functor

template <class T>
class ConvAlgorithmMap;

template <>
class ConvAlgorithmMap<Eigen::ThreadPoolDevice> {};
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_CONV_2D_H_
