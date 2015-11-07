// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/kernels/softmax_op.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftmaxOp : public OpKernel {
 public:
  explicit SoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(0, logits_in.shape(), &softmax_out));
    functor::SoftmaxFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), logits_in.matrix<T>(),
            softmax_out->matrix<T>());
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from SoftmaxEigenImpl.
namespace functor {
template <typename T>
struct SoftmaxFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::Matrix softmax) {
    SoftmaxEigenImpl<CPUDevice, T>::Compute(d, logits, softmax);
  }
};
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("Softmax")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        SoftmaxOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("Softmax")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        SoftmaxOp<CPUDevice, double>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Softmax")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        SoftmaxOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
