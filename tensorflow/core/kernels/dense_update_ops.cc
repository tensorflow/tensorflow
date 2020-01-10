#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/assign_op.h"
#include "tensorflow/core/kernels/dense_update_ops.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

template <typename Device, typename T>
class AssignOpT : public AssignOp {
 public:
  using AssignOp::AssignOp;

  void Copy(OpKernelContext* context, Tensor* lhs, const Tensor& rhs) override {
    functor::DenseUpdate<Device, T, ASSIGN> copy;
    copy(context->eigen_device<Device>(), lhs->flat<T>(), rhs.flat<T>());
  }
};

// TODO(jeff): Get rid of use_exclusive_lock_ option
template <typename Device, typename T, DenseUpdateType OP>
class DenseUpdateOp : public OpKernel {
 public:
  explicit DenseUpdateOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("use_locking", &use_exclusive_lock_));
    const DataType dt = DataTypeToEnum<T>::v();
    OP_REQUIRES_OK(context, context->MatchSignature({MakeRefType(dt), dt},
                                                    {MakeRefType(dt)}));
  }

  void Compute(OpKernelContext* context) override {
    // We always return the input ref.
    context->forward_ref_input_to_ref_output(0, 0);

    if (use_exclusive_lock_) {
      mutex_lock l(*context->input_ref_mutex(0));
      DoUpdate(context);
    } else {
      DoUpdate(context);
    }
  }

 private:
  void DoUpdate(OpKernelContext* context) {
    Tensor Tparams = context->mutable_input(0, use_exclusive_lock_);
    const Tensor& Tupdate = context->input(1);
    OP_REQUIRES(context, Tparams.IsInitialized(),
                errors::FailedPrecondition("Attempting to use uninitialized "
                                           "parameters: ",
                                           def().input(0)));
    OP_REQUIRES(
        context, Tparams.IsSameSize(Tupdate),
        errors::InvalidArgument("Parameters and update must be the same size"));

    functor::DenseUpdate<Device, T, OP> update_functor;
    update_functor(context->eigen_device<Device>(), Tparams.flat<T>(),
                   Tupdate.flat<T>());
  }

  bool use_exclusive_lock_;
};

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(type)                                     \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      AssignOpT<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Only register 'Assign' on GPU for the subset of types also supported by
// 'Variable' (see variable_ops.cc.)
#define REGISTER_GPU_KERNELS(type)                                 \
  namespace functor {                                              \
  template <>                                                      \
  void DenseUpdate<GPUDevice, type, ASSIGN>::operator()(           \
      const GPUDevice& d, typename TTypes<type>::Flat lhs,         \
      typename TTypes<type>::ConstFlat rhs);                       \
  extern template struct DenseUpdate<GPUDevice, type, ASSIGN>;     \
  }                                                                \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("Assign").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      AssignOpT<GPUDevice, type>);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

#define REGISTER_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<CPUDevice, type, DenseUpdateType::SUB>);

TF_CALL_NUMBER_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC_FOR_OP(T, OP)                     \
  template <>                                              \
  void DenseUpdate<GPUDevice, T, OP>::operator()(          \
      const GPUDevice& d, typename TTypes<T>::Flat params, \
      typename TTypes<T>::ConstFlat update);               \
  extern template struct DenseUpdate<GPUDevice, T, OP>
#define DECLARE_GPU_SPEC(T)                         \
  DECLARE_GPU_SPEC_FOR_OP(T, DenseUpdateType::ADD); \
  DECLARE_GPU_SPEC_FOR_OP(T, DenseUpdateType::SUB)
TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_FOR_OP
}  // namespace functor

#define REGISTER_GPU_KERNELS(type)                                    \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::ADD>);          \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("AssignSub").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      DenseUpdateOp<GPUDevice, type, DenseUpdateType::SUB>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // end GOOGLE_CUDA

}  // namespace tensorflow
