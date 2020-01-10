// See docs in ../ops/image_ops.cc
#define EIGEN_USE_THREADS

#include <memory>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/adjust_contrast_op.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
#include "tensorflow/core/public/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AdjustContrastOp : public OpKernel {
 public:
  explicit AdjustContrastOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& factor = context->input(1);
    const Tensor& min_value = context->input(2);
    const Tensor& max_value = context->input(3);
    OP_REQUIRES(context, input.dims() >= 3,
                errors::InvalidArgument("input must be at least 3-D, got shape",
                                        input.shape().ShortDebugString()));
    const int64 height = input.dim_size(input.dims() - 3);
    const int64 width = input.dim_size(input.dims() - 2);
    const int64 channels = input.dim_size(input.dims() - 1);

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(factor.shape()),
                errors::InvalidArgument("contrast_factor must be scalar: ",
                                        factor.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(min_value.shape()),
                errors::InvalidArgument("min_value must be scalar: ",
                                        min_value.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(max_value.shape()),
                errors::InvalidArgument("max_value must be scalar: ",
                                        max_value.shape().ShortDebugString()));

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

    Tensor mean_values;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<float>::value,
                                                   TensorShape(input.shape()),
                                                   &mean_values));

    if (input.NumElements() > 0) {
      const int64 batch = input.NumElements() / (height * width * channels);
      const int64 shape[4] = {batch, height, width, channels};
      functor::AdjustContrast<Device, T>()(
          context->eigen_device<Device>(), input.shaped<T, 4>(shape),
          factor.scalar<float>(), min_value.scalar<float>(),
          max_value.scalar<float>(), mean_values.shaped<float, 4>(shape),
          output->shaped<float, 4>(shape));
    }
  }
};

#define REGISTER_KERNEL(T)                                              \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<CPUDevice, T>);

REGISTER_KERNEL(uint8);
REGISTER_KERNEL(int8);
REGISTER_KERNEL(int16);
REGISTER_KERNEL(int32);
REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_SPEC(T)                                         \
  template <>                                                       \
  void AdjustContrast<GPUDevice, T>::operator()(                    \
      const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input, \
      typename TTypes<float>::ConstScalar contrast_factor,          \
      typename TTypes<float>::ConstScalar min_value,                \
      typename TTypes<float>::ConstScalar max_value,                \
      typename TTypes<float, 4>::Tensor mean_values,                \
      typename TTypes<float, 4>::Tensor output);                    \
  extern template struct AdjustContrast<GPUDevice, T>;

DECLARE_GPU_SPEC(uint8);
DECLARE_GPU_SPEC(int8);
DECLARE_GPU_SPEC(int16);
DECLARE_GPU_SPEC(int32);
DECLARE_GPU_SPEC(float);
DECLARE_GPU_SPEC(double);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("AdjustContrast").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      AdjustContrastOp<GPUDevice, T>);
REGISTER_GPU_KERNEL(uint8);
REGISTER_GPU_KERNEL(int8);
REGISTER_GPU_KERNEL(int16);
REGISTER_GPU_KERNEL(int32);
REGISTER_GPU_KERNEL(float);
REGISTER_GPU_KERNEL(double);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
