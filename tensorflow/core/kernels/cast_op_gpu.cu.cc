#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/bfloat16.h"
#include "tensorflow/core/kernels/cast_op.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename O, typename I>
struct CastFunctor<GPUDevice, O, I> {
  void operator()(const GPUDevice& d, typename TTypes<O>::Flat o,
                  typename TTypes<I>::ConstFlat i) {
    Cast<GPUDevice, O, I>(d, o, i);
  }
};

#define DEFINE(O, I) template struct CastFunctor<GPUDevice, O, I>;
DEFINE(float, double);
DEFINE(float, int32);
DEFINE(float, int64);
DEFINE(double, float);
DEFINE(double, int32);
DEFINE(double, int64);
DEFINE(int32, float);
DEFINE(int32, double);
DEFINE(int32, int64);
DEFINE(int64, float);
DEFINE(int64, double);
DEFINE(int64, int32);
DEFINE(int32, bool);
DEFINE(float, bool);
DEFINE(float, uint8);
DEFINE(uint8, float);
DEFINE(float, bfloat16);
DEFINE(bfloat16, float);
#undef DEFINE

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
