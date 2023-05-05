/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/tensor_to_hash_bucket_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class TensorToHashBucketOp : public OpKernel {
 public:
  explicit TensorToHashBucketOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_buckets", &num_buckets_));

    DataType dtype;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype));
    OP_REQUIRES(ctx,
                dtype == DT_INT8 || dtype == DT_UINT8 || dtype == DT_INT16 ||
                    dtype == DT_UINT16 || dtype == DT_INT32 ||
                    dtype == DT_UINT32 || dtype == DT_INT64 ||
                    dtype == DT_UINT64,
                errors::InvalidArgument("TensorToHashBucketOp doesn't support "
                                        "datatype ",
                                        DataTypeString(dtype)));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor* input_tensor;
    OP_REQUIRES_OK(context, context->input("input", &input_tensor));
    const auto& input_flat = input_tensor->flat<T>();

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output("output", input_tensor->shape(),
                                            &output_tensor));
    auto output_flat = output_tensor->flat<int64_t>();

    functor::LaunchTensorToHashBucket<Device, T>()(
        context, num_buckets_, input_flat.data(), input_tensor->NumElements(),
        output_flat.data());
  }

 private:
  int64_t num_buckets_;

  TF_DISALLOW_COPY_AND_ASSIGN(TensorToHashBucketOp);
};

#define REGISTER_CPU_KERNELS(type)                        \
  REGISTER_KERNEL_BUILDER(Name("_TensorToHashBucketFast") \
                              .Device(DEVICE_CPU)         \
                              .TypeConstraint<type>("T"), \
                          TensorToHashBucketOp<CPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_CPU_KERNELS);

#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                        \
  REGISTER_KERNEL_BUILDER(Name("_TensorToHashBucketFast") \
                              .Device(DEVICE_GPU)         \
                              .TypeConstraint<type>("T"), \
                          TensorToHashBucketOp<GPUDevice, type>);

TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);

#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
