#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/tile_ops.h"
#include <stdio.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

#define DEFINE_TYPE(T) \
  DEFINE_DIM(T, 1)     \
  DEFINE_DIM(T, 2)     \
  DEFINE_DIM(T, 3)     \
  DEFINE_DIM(T, 4)     \
  DEFINE_DIM(T, 5)

#define DEFINE_DIM(T, NDIM)                     \
  template struct Tile<GPUDevice, T, NDIM>;     \
  template struct TileGrad<GPUDevice, T, NDIM>; \
  template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>;

DEFINE_TYPE(float)
DEFINE_TYPE(double)
DEFINE_TYPE(int64)
DEFINE_TYPE(int32)
DEFINE_TYPE(int16)
// NOTE(keveman): Eigen's int8 and string versions don't compile yet with nvcc.

#undef DEFINE_DIM
#undef DEFINE_TYPE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
