// See docs in ../ops/math_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/aggregate_ops.h"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/register_types.h"

#include "tensorflow/core/platform/logging.h"
namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class AddNOp : public OpKernel {
 public:
  explicit AddNOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* ctx) override {
    if (!ctx->ValidateInputsAreSameShape(this)) return;

    const Tensor& input0 = ctx->input(0);
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, input0.shape(), &output));
    auto To = output->flat<T>();

    const int num = ctx->num_inputs();
    if (num == 1) {
      *output = input0;
      return;
    }

#define I(IDX) ctx->input(IDX).flat<T>()

#if defined(PLATFORM_POSIX_ANDROID) || defined(PLATFORM_GOOGLE_ANDROID)
    // On Android, we only support additions of two arguments, so we
    // can reduce the number of template instantiations.
    OP_REQUIRES(ctx, num == 2,
                errors::InvalidArgument("Only additions of two arguments "
                                        "supported. Num inputs: ",
                                        num));
    functor::Add2Functor<Device, T> functor2;
    functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
#else
    static const int kWidth = 8;
    int r = num % kWidth;

    switch (r) {
      case 2: {
        functor::Add2Functor<Device, T> functor2;
        functor2(ctx->template eigen_device<Device>(), To, I(0), I(1));
        break;
      }
      case 3: {
        functor::Add3Functor<Device, T> functor3;
        functor3(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2));
        break;
      }
      case 4: {
        functor::Add4Functor<Device, T> functor4;
        functor4(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3));
        break;
      }
      case 5: {
        functor::Add5Functor<Device, T> functor5;
        functor5(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4));
        break;
      }
      case 6: {
        functor::Add6Functor<Device, T> functor6;
        functor6(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5));
        break;
      }
      case 7: {
        functor::Add7Functor<Device, T> functor7;
        functor7(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6));
        break;
      }
      case 0: {
        functor::Add8Functor<Device, T> functor8;
        functor8(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7));
        r = 8;
        break;
      }
      case 1: {
        functor::Add9Functor<Device, T> functor9;
        functor9(ctx->template eigen_device<Device>(), To, I(0), I(1), I(2),
                 I(3), I(4), I(5), I(6), I(7), I(8));
        r = 9;
        break;
      }
    }

    for (; r < num; r += kWidth) {
      functor::Add8pFunctor<Device, T> functor8p;
      functor8p(ctx->template eigen_device<Device>(), To, I(r), I(r + 1),
                I(r + 2), I(r + 3), I(r + 4), I(r + 5), I(r + 6), I(r + 7));
    }
#endif  // defined(PLATFORM_POSIX_ANDROID) || defined(PLATFORM_GOOGLE_ANDROID)

#undef I
  }
};

// Partial specializations for a CPUDevice, that uses the Eigen implementation
// from AddNEigenImpl.
namespace functor {
template <typename T>
struct Add2Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2) {
    Add2EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2);
  }
};
template <typename T>
struct Add3Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3) {
    Add3EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3);
  }
};
template <typename T>
struct Add4Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4) {
    Add4EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4);
  }
};
template <typename T>
struct Add5Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5) {
    Add5EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5);
  }
};
template <typename T>
struct Add6Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6) {
    Add6EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6);
  }
};
template <typename T>
struct Add7Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T>::Flat out,
                  typename TTypes<T>::ConstFlat in1,
                  typename TTypes<T>::ConstFlat in2,
                  typename TTypes<T>::ConstFlat in3,
                  typename TTypes<T>::ConstFlat in4,
                  typename TTypes<T>::ConstFlat in5,
                  typename TTypes<T>::ConstFlat in6,
                  typename TTypes<T>::ConstFlat in7) {
    Add7EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7);
  }
};

template <typename T>
struct Add8Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8);
  }
};

template <typename T>
struct Add8pFunctor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8) {
    Add8pEigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                          in7, in8);
  }
};

template <typename T>
struct Add9Functor<CPUDevice, T> {
  void operator()(
      const CPUDevice& d, typename TTypes<T>::Flat out,
      typename TTypes<T>::ConstFlat in1, typename TTypes<T>::ConstFlat in2,
      typename TTypes<T>::ConstFlat in3, typename TTypes<T>::ConstFlat in4,
      typename TTypes<T>::ConstFlat in5, typename TTypes<T>::ConstFlat in6,
      typename TTypes<T>::ConstFlat in7, typename TTypes<T>::ConstFlat in8,
      typename TTypes<T>::ConstFlat in9) {
    Add9EigenImpl<CPUDevice, T>::Compute(d, out, in1, in2, in3, in4, in5, in6,
                                         in7, in8, in9);
  }
};

}  // namespace functor

#define REGISTER_ADDN(type, dev)                                   \
  REGISTER_KERNEL_BUILDER(                                         \
      Name("AddN").Device(DEVICE_##dev).TypeConstraint<type>("T"), \
      AddNOp<dev##Device, type>)

#define REGISTER_ADDN_CPU(type) REGISTER_ADDN(type, CPU)

TF_CALL_NUMBER_TYPES(REGISTER_ADDN_CPU);
#undef REGISTER_ADDN_CPU

#if GOOGLE_CUDA
REGISTER_ADDN(float, GPU);
#endif  // GOOGLE_CUDA

#undef REGISTER_ADDN

}  // namespace tensorflow
