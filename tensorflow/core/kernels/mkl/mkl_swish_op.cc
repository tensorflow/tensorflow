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

// See docs in ../ops/mkl_nn_ops.cc.

#ifdef INTEL_MKL

#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"

namespace tensorflow {

template <typename Device, typename T>
class MklSwishOp
    : public MklEltwiseFwdActivationOpBase<Device, T,
                                           dnnl::algorithm::eltwise_swish> {
 public:
  ~MklSwishOp() {}

  explicit MklSwishOp(OpKernelConstruction* context)
      : MklEltwiseFwdActivationOpBase<Device, T,
                                      dnnl::algorithm::eltwise_swish>(
            context, 1.0f, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const Tensor& src_tensor = context->input(0);

    // Get shapes of input tensors
    TensorShape src_shape = src_tensor.shape();

    Tensor* dst_tensor = nullptr;
    void* user_i =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));

    TensorShape dst_shape = src_shape;

    OP_REQUIRES_OK(context, context->allocate_output(
                                GetTensorDataIndex(0, context->num_outputs()),
                                dst_shape, &dst_tensor));

    // swish(x) =  x * sigmoid(x).
    void* out_o = static_cast<void*>(dst_tensor->flat<T>().data());
    T feature = (static_cast<T*>(user_i))[0];
    T e1 = Eigen::numext::exp(-feature);
    (static_cast<T*>(out_o))[0] = feature / (static_cast<T>(1) + e1);
    return;
  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES(type)              \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_MklSwish").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MklSwishOp<CPUDevice, type>);
TF_CALL_float(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_SWISH_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif
