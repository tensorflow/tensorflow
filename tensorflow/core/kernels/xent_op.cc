// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/kernels/xent_op.h"
#include "tensorflow/core/public/tensor.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class SoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& logits_in = context->input(0);
    const Tensor& labels_in = context->input(1);
    OP_REQUIRES(context, logits_in.IsSameSize(labels_in),
                errors::InvalidArgument(
                    "logits and labels must be same size: logits_size=",
                    logits_in.shape().DebugString(), " labels_size=",
                    labels_in.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits_in.shape()),
                errors::InvalidArgument("logits must be 2-dimensional"));
    // As we already tested that both inputs have the same shape no need to
    // check that "labels" is a matrix too.

    // loss is 1-D (one per example), and size is batch_size.

    Tensor scratch;
    OP_REQUIRES_OK(
        context, context->allocate_temp(DataTypeToEnum<T>::value,
                                        TensorShape({logits_in.dim_size(0), 1}),
                                        &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({logits_in.dim_size(0)}), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(1, logits_in.shape(), &back_out));

    functor::XentFunctor<Device, T> functor;
    functor(context->eigen_device<Device>(), logits_in.matrix<T>(),
            labels_in.matrix<T>(), scratch.matrix<T>(), loss_out->vec<T>(),
            back_out->matrix<T>());
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T>
struct XentFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<T>::ConstMatrix labels,
                  typename TTypes<T>::Matrix scratch,
                  typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    XentEigenImpl<CPUDevice, T>::Compute(d, logits, labels, scratch, loss,
                                         backprop);
  }
};
}  // namespace functor

REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T"),
                        SoftmaxXentWithLogitsOp<CPUDevice, float>);
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T"),
                        SoftmaxXentWithLogitsOp<CPUDevice, double>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("SoftmaxCrossEntropyWithLogits")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<float>("T"),
                        SoftmaxXentWithLogitsOp<GPUDevice, float>);
#endif  // GOOGLE_CUDA

}  // namespace tensorflow
