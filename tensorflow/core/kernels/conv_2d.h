#ifndef TENSORFLOW_KERNELS_CONV_2D_H_
#define TENSORFLOW_KERNELS_CONV_2D_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {
namespace functor {

// TODO(yangke): revisit these operations and in particular, see if we can
// combine all of them into just one operation without causing nvcc to
// timeout.
template <typename Device, typename T, int Dims>
struct ShuffleAndReverse {
  void operator()(const Device& d, typename TTypes<T, Dims>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, Dims>& order,
                  const Eigen::array<bool, Dims>& reverse_dims,
                  typename TTypes<T, Dims>::Tensor output) {
    output.device(d) = input.shuffle(order).reverse(reverse_dims);
  }
};

template <typename Device, typename T, int Dims>
struct InflatePadAndShuffle {
  void operator()(
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,
      const Eigen::DSizes<Eigen::DenseIndex, Dims>& strides,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, Dims>& pad_dims,
      const Eigen::DSizes<Eigen::DenseIndex, Dims>& order,
      typename TTypes<T, Dims>::Tensor output) {
    output.device(d) = input.inflate(strides).pad(pad_dims).shuffle(order);
  }
};

template <typename Device, typename Input, typename Filter, typename Output>
void SpatialConvolutionFunc(const Device& d, Output output, Input input,
                            Filter filter, int stride,
                            const Eigen::PaddingType& padding) {
  output.device(d) = Eigen::SpatialConvolution(input, filter, stride, padding);
}

template <typename Device, typename T>
struct SpatialConvolution {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor filter, int stride,
                  const Eigen::PaddingType& padding) {
    SpatialConvolutionFunc(d, output, input, filter, stride, padding);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardInput {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor input_backward,
                  typename TTypes<T, 4>::ConstTensor kernel,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int input_rows, int input_cols, int stride) {
    input_backward.device(d) = Eigen::SpatialConvolutionBackwardInput(
        kernel, output_backward, input_rows, input_cols, stride);
  }
};

template <typename Device, typename T>
struct SpatialConvolutionBackwardKernel {
  void operator()(const Device& d,
                  typename TTypes<T, 4>::Tensor kernel_backward,
                  typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 4>::ConstTensor output_backward,
                  int kernel_rows, int kernel_cols, int stride) {
    kernel_backward.device(d) = Eigen::SpatialConvolutionBackwardKernel(
        input, output_backward, kernel_rows, kernel_cols, stride);
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

template <typename Device, typename T>
struct TransformFilter {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  typename TTypes<T, 4>::Tensor out) {
    out.device(d) = in.shuffle(Eigen::DSizes<Eigen::DenseIndex, 4>(3, 2, 0, 1));
  }
};

template <typename Device, typename T>
struct TransformDepth {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, 4>& shuffle,
                  typename TTypes<T, 4>::Tensor out) {
    out.device(d) = in.shuffle(shuffle);
  }
};

template <typename Device, typename T>
struct PadInput {
  void operator()(const Device& d, typename TTypes<T, 4>::ConstTensor in,
                  int padding_rows_left, int padding_rows_right,
                  int padding_cols_left, int padding_cols_right,
                  typename TTypes<T, 4>::Tensor out) {
    Eigen::array<std::pair<ptrdiff_t, ptrdiff_t>, 4> padding;
    padding[0] = std::make_pair(0, 0);
    padding[1] = std::make_pair(padding_rows_left, padding_rows_right);
    padding[2] = std::make_pair(padding_cols_left, padding_cols_right);
    padding[3] = std::make_pair(0, 0);
    out.device(d) = in.pad(padding);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_2D_H_
