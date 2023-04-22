/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/in_topk_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename TARGET_T>
class InTopK : public OpKernel {
 public:
  explicit InTopK(OpKernelConstruction* context) : OpKernel(context) {
    if (context->num_inputs() == 2) {
      OP_REQUIRES_OK(context, context->GetAttr("k", &k_));
    }
  }

  void Compute(OpKernelContext* context) override {
    const auto& predictions_in = context->input(0);
    const auto& targets_in = context->input(1);

    int64 k_value = k_;
    const Tensor* k_tensor = nullptr;

    if (context->num_inputs() == 3) {
      const auto& k_in = context->input(2);

      OP_REQUIRES(context, TensorShapeUtils::IsScalar(k_in.shape()),
                  errors::InvalidArgument("k must be 0-D, got shape ",
                                          k_in.shape().DebugString()));

      k_tensor = &k_in;
    }

    OP_REQUIRES(context, predictions_in.dims() == 2,
                errors::InvalidArgument("predictions must be 2-dimensional"));
    OP_REQUIRES(context, targets_in.dims() == 1,
                errors::InvalidArgument("targets must be 1-dimensional"));
    OP_REQUIRES(context, predictions_in.dim_size(0) == targets_in.dim_size(0),
                errors::InvalidArgument("First dimension of predictions ",
                                        predictions_in.dim_size(0),
                                        " must match length of targets ",
                                        targets_in.dim_size(0)));

    const auto predictions = predictions_in.matrix<T>();
    const auto targets = targets_in.vec<TARGET_T>();

    Tensor* t_out = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({targets_in.dim_size(0)}), &t_out));
    auto out = t_out->vec<bool>();

    functor::InTopKFunctor<Device, T, TARGET_T> f;
    functor::TopKArg arg;
    arg.k_value = k_value;
    arg.k_tensor = k_tensor;
    f(context, predictions, targets, arg, out);
  }

 private:
  int k_;
};

REGISTER_KERNEL_BUILDER(Name("InTopK")
                            .Device(DEVICE_CPU)
                            .HostMemory("predictions")
                            .HostMemory("targets")
                            .HostMemory("precision")
                            .TypeConstraint<int32>("T"),
                        InTopK<CPUDevice, float, int32>);
REGISTER_KERNEL_BUILDER(Name("InTopK")
                            .Device(DEVICE_CPU)
                            .HostMemory("predictions")
                            .HostMemory("targets")
                            .HostMemory("precision")
                            .TypeConstraint<int64>("T"),
                        InTopK<CPUDevice, float, int64>);

REGISTER_KERNEL_BUILDER(Name("InTopKV2")
                            .Device(DEVICE_CPU)
                            .HostMemory("predictions")
                            .HostMemory("targets")
                            .HostMemory("k")
                            .HostMemory("precision")
                            .TypeConstraint<int32>("T"),
                        InTopK<CPUDevice, float, int32>);
REGISTER_KERNEL_BUILDER(Name("InTopKV2")
                            .Device(DEVICE_CPU)
                            .HostMemory("predictions")
                            .HostMemory("targets")
                            .HostMemory("k")
                            .HostMemory("precision")
                            .TypeConstraint<int64>("T"),
                        InTopK<CPUDevice, float, int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, TARGET_T)                               \
  template <>                                                       \
  void InTopKFunctor<GPUDevice, T, TARGET_T>::operator()(           \
      OpKernelContext* context,                                     \
      typename TTypes<T, 2>::ConstTensor predictions,               \
      typename TTypes<TARGET_T>::ConstVec targets, const TopKArg k, \
      typename TTypes<bool>::Vec output);                           \
  extern template struct InTopKFunctor<GPUDevice, T, TARGET_T>;

DECLARE_GPU_SPEC(float, int32);
DECLARE_GPU_SPEC(float, int64);

#undef DECLARE_GPU_SPEC
}  // namespace functor

REGISTER_KERNEL_BUILDER(
    Name("InTopKV2").Device(DEVICE_GPU).TypeConstraint<int32>("T"),
    InTopK<GPUDevice, float, int32>);
REGISTER_KERNEL_BUILDER(
    Name("InTopKV2").Device(DEVICE_GPU).TypeConstraint<int64>("T"),
    InTopK<GPUDevice, float, int64>);

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
