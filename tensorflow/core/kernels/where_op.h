#ifndef TENSORFLOW_KERNELS_WHERE_OP_H_
#define TENSORFLOW_KERNELS_WHERE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace functor {

template <typename Device>
struct NumTrue {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<bool>::ConstFlat input,
      TTypes<int64>::Scalar num_true) {
    num_true.device(d) = input.template cast<int64>().sum();
  }
};

template <typename Device, int NDIM>
struct Where {
  EIGEN_ALWAYS_INLINE static void Compute(
      const Device& d, typename TTypes<bool, NDIM>::ConstTensor input,
      typename TTypes<int64>::Matrix output) {
    Eigen::DenseIndex true_n = 0;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
    Eigen::DSizes<Eigen::DenseIndex, NDIM> strides;

    // Calculate strides for RowMajor order.
    EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                         static_cast<int>(Eigen::RowMajor)),
                        INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);

    strides[NDIM - 1] = 1;
    for (int i = NDIM - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * dims[i + 1];
    }

    // Note, no bounds checking is done on true_n.  It is assumed that
    // the output was correctly sized via output of NumTrue::Compute.
    for (Eigen::DenseIndex n = 0; n < input.size(); ++n) {
      if (input.data()[n]) {
        WriteIndexRowMajor(output, strides, true_n, n);
        ++true_n;
      }
    }
  }

  EIGEN_ALWAYS_INLINE static void WriteIndexRowMajor(
      typename TTypes<int64>::Matrix output,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& strides,
      Eigen::DenseIndex true_n, Eigen::DenseIndex index) {
    for (int i = 0; i < NDIM; ++i) {
      output(true_n, i) = index / strides[i];
      index %= strides[i];
    }
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_WHERE_OP_H_
