/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && !defined(ENABLE_ONEDNN_V3)

#include "tensorflow/core/kernels/mkl/mkl_eltwise_activation_base_op.h"

namespace tensorflow {

template <typename Device, typename T>
class MklFusedMishOp
    : public MklEltwiseFwdActivationOpBase<Device, T,
                                           dnnl::algorithm::eltwise_mish> {
 public:
  ~MklFusedMishOp() {}

  explicit MklFusedMishOp(OpKernelConstruction* context)
      : MklEltwiseFwdActivationOpBase<Device, T, dnnl::algorithm::eltwise_mish>(
            context, 1.0f, 0.0f) {}

  virtual void Compute_Scalar(OpKernelContext* context) {
    const Tensor& src_tensor = context->input(0);

    // Get input tensor shape
    TensorShape src_shape = src_tensor.shape();

    Tensor* dst_tensor = nullptr;
    void* src_buf =
        static_cast<void*>(const_cast<T*>(src_tensor.flat<T>().data()));

    TensorShape dst_shape = src_shape;
    OP_REQUIRES_OK(context, context->allocate_output(
                                GetTensorDataIndex(0, context->num_outputs()),
                                dst_shape, &dst_tensor));

    T* dst_buf = static_cast<T*>(dst_tensor->flat<T>().data());
    // mish(x) = x . tanh(softplus(x))
    // y = softplus(x) = ln(1+e^x)
    // tanh(y) = (e^y - e^(-y))/ (e^y + e^(-y))
    // mish(x) = x * tanh(y)
    T feature = (static_cast<T*>(src_buf))[0];
    T s1 = Eigen::numext::exp(feature);
    T s2 = Eigen::numext::log(s1 + 1);
    T e1 = Eigen::numext::exp(s2);
    T e2 = Eigen::numext::exp(-s2);
    (static_cast<T*>(dst_buf))[0] = feature * ((e1 - e2) / (e1 + e2));
    return;
  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_MISH_MKL_SUPPORTED_KERNELS_TYPES(type)                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("_MklFusedMish").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      MklFusedMishOp<CPUDevice, type>);
TF_CALL_float(REGISTER_MISH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_MISH_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL && !ENABLE_ONEDNN_V3
