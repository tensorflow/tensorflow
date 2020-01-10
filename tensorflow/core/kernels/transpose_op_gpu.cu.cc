#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/kernels/transpose_op_functor.h"

namespace tensorflow {
namespace functor {

template <typename T, int NDIMS>
struct TransposeFunctor<Eigen::GpuDevice, T, NDIMS> {
  void operator()(const Eigen::GpuDevice& d,
                  typename TTypes<T, NDIMS>::Tensor out,
                  typename TTypes<T, NDIMS>::ConstTensor in, const int* perm) {
    Transpose<Eigen::GpuDevice, T, NDIMS>(d, out, in, perm);
  }
};

#define DEFINE(T, N) template struct TransposeFunctor<Eigen::GpuDevice, T, N>;
#define DEFINE_DIM(T) \
  DEFINE(T, 1);       \
  DEFINE(T, 2);       \
  DEFINE(T, 3);       \
  DEFINE(T, 4);       \
  DEFINE(T, 5);       \
  DEFINE(T, 6);       \
  DEFINE(T, 7);       \
  DEFINE(T, 8);
DEFINE_DIM(uint8);
DEFINE_DIM(int8);
DEFINE_DIM(int16);
DEFINE_DIM(int32);
DEFINE_DIM(int64);
DEFINE_DIM(float);
DEFINE_DIM(double);
#undef DEFINE_DIM
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
