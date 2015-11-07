// See docs in ../ops/math_ops.cc.

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {

int32 GetValue(int32 v) { return v; }

template <typename T>
class RangeOp : public OpKernel {
 public:
  explicit RangeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& start_in = context->input(0);
    const Tensor& limit_in = context->input(1);
    const Tensor& delta_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyScalar(start_in.shape()),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyScalar(limit_in.shape()),
                errors::InvalidArgument("limit must be a scalar, not shape ",
                                        limit_in.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsLegacyScalar(delta_in.shape()),
                errors::InvalidArgument("delta must be a scalar, not shape ",
                                        delta_in.shape().ShortDebugString()));
    const int32 start = GetValue(start_in.scalar<T>()());
    const int32 limit = GetValue(limit_in.scalar<T>()());
    OP_REQUIRES(context, start <= limit,
                errors::InvalidArgument("Requires start <= limit: ", start, "/",
                                        limit));
    const int32 delta = GetValue(delta_in.scalar<T>()());
    OP_REQUIRES(context, delta > 0,
                errors::InvalidArgument("Requires delta > 0: ", delta));
    int32 size = (limit - start + delta - 1) / delta;
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({size}), &out));
    auto flat = out->flat<T>();
    int32 val = start;
    for (int32 i = 0; i < size; ++i) {
      flat(i) = T(val);
      val += delta;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Range")
                            .Device(DEVICE_CPU)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta")
                            .HostMemory("output"),
                        RangeOp<int32>);

#if GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(Name("Range")
                            .Device(DEVICE_GPU)
                            .HostMemory("start")
                            .HostMemory("limit")
                            .HostMemory("delta")
                            .HostMemory("output"),
                        RangeOp<int32>);
#endif  // GOOGLE_CUDA

template <typename T>
class LinSpaceOp : public OpKernel {
 public:
  explicit LinSpaceOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& start_in = context->input(0);
    const Tensor& stop_in = context->input(1);
    const Tensor& num_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(start_in.shape()),
                errors::InvalidArgument("start must be a scalar, not shape ",
                                        start_in.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(stop_in.shape()),
                errors::InvalidArgument("stop must be a scalar, not shape ",
                                        stop_in.shape().ShortDebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(num_in.shape()),
                errors::InvalidArgument("num must be a scalar, not shape ",
                                        num_in.shape().ShortDebugString()));
    const T start = start_in.scalar<T>()();
    const T stop = stop_in.scalar<T>()();
    const int32 num = num_in.scalar<int32>()();
    OP_REQUIRES(context, num > 0,
                errors::InvalidArgument("Requires num > 0: ", num));
    Tensor* out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape({num}), &out));
    auto flat = out->flat<T>();
    if (num == 1) {
      flat(0) = start;
    } else {
      const T step = (stop - start) / (num - 1);
      for (int32 i = 0; i < num; ++i) flat(i) = start + step * i;
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("LinSpace")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("T")
                            .HostMemory("start")
                            .HostMemory("stop")
                            .HostMemory("num")
                            .HostMemory("output"),
                        LinSpaceOp<float>);
REGISTER_KERNEL_BUILDER(Name("LinSpace")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<double>("T")
                            .HostMemory("start")
                            .HostMemory("stop")
                            .HostMemory("num")
                            .HostMemory("output"),
                        LinSpaceOp<double>);

}  // namespace tensorflow
