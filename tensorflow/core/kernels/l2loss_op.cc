// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/l2loss_op.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class L2LossOp : public OpKernel {
 public:
  explicit L2LossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // The input tensor can be of any number of dimensions, even though it's
    // 2D in most typical applications.
    const Tensor& input = context->input(0);
    // The output is a single number.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({}), &output));
    functor::L2Loss<Device, T>()(context->eigen_device<Device>(),
                                 input.flat<T>(), output->scalar<T>());
  }
};

#define REGISTER_KERNEL(T)                                      \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      L2LossOp<CPUDevice, T>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void L2Loss<GPUDevice, T>::operator()(const GPUDevice& d,                    \
                                        typename TTypes<T>::ConstTensor input, \
                                        typename TTypes<T>::Scalar output);    \
  extern template struct L2Loss<GPUDevice, T>;

DECLARE_GPU_SPEC(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                      \
      Name("L2Loss").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      L2LossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
