#ifndef TENSORFLOW_KERNELS_ARGMAX_OP_H_
#define TENSORFLOW_KERNELS_ARGMAX_OP_H_
// Generator definition for ArgMaxOp, must be compilable by nvcc.

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/port.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct ArgMax {
#define DECLARE_COMPUTE_SPEC(Dims)                                     \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                        \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,    \
      const int32 dimension,                                           \
      typename TTypes<int64, Dims - 1>::Tensor output) {               \
    output.device(d) = input.argmax(dimension).template cast<int64>(); \
  }

  DECLARE_COMPUTE_SPEC(1);
  DECLARE_COMPUTE_SPEC(2);
  DECLARE_COMPUTE_SPEC(3);
  DECLARE_COMPUTE_SPEC(4);
  DECLARE_COMPUTE_SPEC(5);

#undef DECLARE_COMPUTE_SPEC
};

template <typename Device, typename T>
struct ArgMin {
#define DECLARE_COMPUTE_SPEC(Dims)                                     \
  EIGEN_ALWAYS_INLINE static void Reduce##Dims(                        \
      const Device& d, typename TTypes<T, Dims>::ConstTensor input,    \
      const int32 dimension,                                           \
      typename TTypes<int64, Dims - 1>::Tensor output) {               \
    output.device(d) = input.argmin(dimension).template cast<int64>(); \
  }

  DECLARE_COMPUTE_SPEC(1);
  DECLARE_COMPUTE_SPEC(2);
  DECLARE_COMPUTE_SPEC(3);
  DECLARE_COMPUTE_SPEC(4);
  DECLARE_COMPUTE_SPEC(5);

#undef DECLARE_COMPUTE_SPEC
};

}  // namespace functor

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_ARGMAX_OP_H_
