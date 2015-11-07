#ifndef TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
#define TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
// Functor definition for MaxPoolingOp, must be compilable by nvcc.

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"

namespace tensorflow {
namespace functor {

template <typename Device, typename T>
struct SpatialMaxPooling {
  void operator()(const Device& d, typename TTypes<T, 4>::Tensor output,
                  typename TTypes<T, 4>::ConstTensor input, int window_rows,
                  int window_cols, int row_stride, int col_stride,
                  const Eigen::PaddingType& padding) {
    // Because we swap the layout, we swap the row/cols as well
    output.swap_layout().device(d) =
        Eigen::SpatialMaxPooling(input.swap_layout(), window_cols, window_rows,
                                 col_stride, row_stride, padding);
  }
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_MAXPOOLING_OP_H_
