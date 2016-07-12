
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/norm_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"

namespace tensorflow {
typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct Norm1D<CPUDevice, T> {
   void operator()(const CPUDevice& d, typename TTypes<T>::ConstFlat x,
		   typename TTypes<T>::ConstFlat mean,
		   typename TTypes<T>::ConstFlat stdd,
		   typename TTypes<T>::Flat out) {
	   auto new_x = x - mean;
	   auto z = - new_x.square() / (stdd.constant(2.0) * stdd.square());
	   const T pi = std::acos(-1);
	   auto denom = stdd.constant(2.0 * pi).sqrt() * stdd;    
	   out.device(d) = z.exp() / denom;
   }
};
} // end namespace functor

template <typename Device, typename T>
class Norm1DOp : public OpKernel {
  public:
    explicit Norm1DOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

    void Compute(OpKernelContext* ctx) override {
      const Tensor& x = ctx->input(0);
      const Tensor& mean = ctx->input(1);
      const Tensor& stdd = ctx->input(2);

      OP_REQUIRES(ctx, x.shape().IsSameSize(mean.shape()),
          errors::InvalidArgument("x and mean do not have the same shape",
            x.shape().DebugString(), " ", mean.shape().DebugString()));
      OP_REQUIRES(ctx, x.shape().IsSameSize(stdd.shape()),
          errors::InvalidArgument("x and std do not have the same shape",
            x.shape().DebugString(), " ", stdd.shape().DebugString()));
     
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, x.shape(), &out));

      const Device& device = ctx->template eigen_device<Device>();
      functor::Norm1D<Device, T>()(device, x.flat<T>(), mean.flat<T>(),
          stdd.flat<T>(), out->flat<T>());
    }

};

#define REGISTER_CPU(T)                         \
  REGISTER_KERNEL_BUILDER(Name("Norm1D")        \
      .Device(DEVICE_CPU)                       \
      .TypeConstraint<T>("T"),                  \
      Norm1DOp<CPUDevice, T>);

TF_CALL_float(REGISTER_CPU);
TF_CALL_double(REGISTER_CPU);

#undef REGISTER_CPU

} // end namespace tensorflow
