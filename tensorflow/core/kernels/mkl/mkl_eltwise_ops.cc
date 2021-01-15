/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifdef INTEL_MKL

#include "tensorflow/core/kernels/mkl/mkl_eltwise_ops.h"

#include <unordered_map>

#include "mkldnn.hpp"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/mkl_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

using mkldnn::algorithm;
using mkldnn::eltwise_forward;
using mkldnn::memory;
using mkldnn::prop_kind;
using mkldnn::stream;

using EltwiseFwdPd = mkldnn::eltwise_forward::primitive_desc;
using EltwiseBwdPd = mkldnn::eltwise_backward::primitive_desc;

namespace tensorflow {

template <typename Device, typename T>
class MklReluOp
    : public MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_relu> {
 public:
  ~MklReluOp() {}

  explicit MklReluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_relu>(
            context, 0.0f, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, this->KSrcIndex, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDstIndex, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = static_cast<T*>(dst_tensor->flat<T>().data());
    out_o[0] = std::max(user_i[0], static_cast<T>(0));
    return;
  }
};

template <typename Device, typename T>
class MklReluGradOp
    : public MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_relu> {
 public:
  ~MklReluGradOp() {}

  explicit MklReluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_relu>(
            context, 0.0f, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    const Tensor& diff_dst_tensor = MklGetInput(context, this->KDiffDstIndex);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, this->KDiffDstIndex, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDiffSrcIndex, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = static_cast<T*>(diff_src_tensor->flat<T>().data());
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    T* user_g =
        static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    out_o[0] = user_g[0] * static_cast<T>((user_i[0] > static_cast<T>(0)));
    return;
  }
};

template <typename Device, typename T>
class MklEluOp
    : public MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_elu> {
 public:
  ~MklEluOp() {}

  explicit MklEluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_elu>(
            context, 0.0f, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, this->KSrcIndex, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDstIndex, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = static_cast<T*>(dst_tensor->flat<T>().data());
    // return exp(feature) - 1 if feature > 0; feature otherwise
    T& feature = user_i[0];
    if (feature < static_cast<T>(0))
      out_o[0] = Eigen::numext::exp(feature);
    else
      out_o[0] = feature;
    return;
  }
};

template <typename Device, typename T>
class MklEluGradOp
    : public MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_elu> {
 public:
  ~MklEluGradOp() {}

  explicit MklEluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_elu>(
            context, 0.0f, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    const Tensor& diff_dst_tensor = MklGetInput(context, this->KDiffDstIndex);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, this->KDiffDstIndex, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDiffSrcIndex, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = static_cast<T*>(diff_src_tensor->flat<T>().data());
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    T* user_g =
        static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    // gradient of elu(x) = 1 if x > 0; elu(x) + 1 otherwise
    T& feature = user_i[0];
    if (feature > static_cast<T>(0)) {
      out_o[0] = user_g[0];
    } else {
      T elu = Eigen::numext::exp(feature) - static_cast<T>(1);
      out_o[0] = user_g[0] * (elu + static_cast<T>(1));
    }
  }
};

template <typename Device, typename T>
class MklTanhOp
    : public MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_tanh> {
 public:
  ~MklTanhOp() {}

  explicit MklTanhOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_tanh>(
            context, 0.0f, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, this->KSrcIndex, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDstIndex, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = static_cast<T*>(dst_tensor->flat<T>().data());
    // tanh(x) = (e^x - e^(-x))/ (e^x + e^(-x)) = (e^2x - 1) / (e^2x + 1)
    T e1 = Eigen::numext::exp(T(2) * user_i[0]);
    out_o[0] = (e1 - T(1)) / (e1 + T(1));
    return;
  }
};

template <typename Device, typename T>
class MklTanhGradOp
    : public MklEltwiseGradOpBase<
          Device, T, mkldnn::algorithm::eltwise_tanh_use_dst_for_bwd> {
 public:
  ~MklTanhGradOp() {}

  explicit MklTanhGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T,
                             mkldnn::algorithm::eltwise_tanh_use_dst_for_bwd>(
            context, 0.0f, 0.0f) {}

  int GetDiffDstIndex() const override { return 1; }
  int GetSrcIndex() const override { return 0; }
  int GetDiffSrcIndex() const override { return 0; }

  // TanhGrad gets 'y' from Tanh, where 'y' is output of Tanh(x).
  int GetTypeOfInputTensorFromFwdOp() const override { return MKLDNN_ARG_DST; }

  void Compute_Scalar(OpKernelContext* context) override {
    // y and dy for Tanh is in reverse order compared with order for
    // Relu/Elu/other element-wise ops.
    // This is because Tanh is math op in Tensorflow; others are NN ops.
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    const Tensor& diff_dst_tensor = MklGetInput(context, this->KDiffDstIndex);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, this->KDiffDstIndex, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDiffSrcIndex, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = static_cast<T*>(diff_src_tensor->flat<T>().data());
    T* user_i = static_cast<T*>(const_cast<T*>(src_tensor.flat<T>().data()));
    // gradient of tanh(x) = 1 - tanh(x)^2
    // Input to TanhGrad is output of Tanh. So we do not need to compute
    // Tanh again.
    T* user_g =
        static_cast<T*>(const_cast<T*>(diff_dst_tensor.flat<T>().data()));
    out_o[0] = user_g[0] * (static_cast<T>(1) - user_i[0] * user_i[0]);
  }
};

#define RELU6_UPPER_BOUND 6.0f
template <typename Device, typename T>
class MklRelu6Op
    : public MklEltwiseOpBase<Device, T,
                              mkldnn::algorithm::eltwise_bounded_relu> {
 public:
  ~MklRelu6Op() {}

  explicit MklRelu6Op(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, this->KSrcIndex, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDstIndex, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = std::min(std::max(user_i[0], static_cast<T>(0)),
                        static_cast<T>(RELU6_UPPER_BOUND));
    return;
  }
};

template <typename Device, typename T>
class MklRelu6GradOp
    : public MklEltwiseGradOpBase<Device, T,
                                  mkldnn::algorithm::eltwise_bounded_relu> {
 public:
  ~MklRelu6GradOp() {}

  explicit MklRelu6GradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T,
                             mkldnn::algorithm::eltwise_bounded_relu>(
            context, RELU6_UPPER_BOUND, 0.0f) {}

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    const Tensor& diff_dst_tensor = MklGetInput(context, this->KDiffDstIndex);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, this->KDiffDstIndex, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDiffSrcIndex, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_g[0] *
               static_cast<T>(user_i[0] > static_cast<T>(0) &&
                              (user_i[0] < static_cast<T>(RELU6_UPPER_BOUND)));
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluOp
    : public MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_relu> {
 public:
  ~MklLeakyReluOp() {}

  explicit MklLeakyReluOp(OpKernelConstruction* context)
      : MklEltwiseOpBase<Device, T, mkldnn::algorithm::eltwise_relu>(
            context, 0.0f, 0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    MklDnnShape dnn_shape_src;
    GetMklShape(context, this->KSrcIndex, &dnn_shape_src);

    Tensor* dst_tensor = nullptr;
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    MklDnnShape dnn_shape_dst;
    dnn_shape_dst.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDstIndex, &dst_tensor,
                              src_tensor.shape(), dnn_shape_dst);
    T* out_o = dst_tensor->flat<T>().data();
    out_o[0] = user_i[0] >= T(0) ? user_i[0] : user_i[0] * T(this->alpha_);
    return;
  }
};

template <typename Device, typename T>
class MklLeakyReluGradOp
    : public MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_relu> {
 public:
  ~MklLeakyReluGradOp() {}

  explicit MklLeakyReluGradOp(OpKernelConstruction* context)
      : MklEltwiseGradOpBase<Device, T, mkldnn::algorithm::eltwise_relu>(
            context, 0.0f, 0.0f) {
    float alpha;
    OP_REQUIRES_OK(context, context->GetAttr("alpha", &alpha));
    OP_REQUIRES(
        context, alpha <= 1,
        errors::InvalidArgument("MKL LeakyRelu only supports alpha <= 1. "
                                "alpha is: ",
                                alpha));

    this->alpha_ = alpha;
  }

  void Compute_Scalar(OpKernelContext* context) override {
    const Tensor& src_tensor = MklGetInput(context, this->KSrcIndex);
    const Tensor& diff_dst_tensor = MklGetInput(context, this->KDiffDstIndex);
    Tensor* diff_src_tensor = nullptr;

    MklDnnShape dnn_shape_diff_dst;
    GetMklShape(context, this->KDiffDstIndex, &dnn_shape_diff_dst);

    MklDnnShape dnn_shape_diff_src;
    dnn_shape_diff_src.SetMklTensor(false);
    AllocateOutputSetMklShape(context, this->KDiffSrcIndex, &diff_src_tensor,
                              diff_dst_tensor.shape(), dnn_shape_diff_src);
    T* out_o = diff_src_tensor->flat<T>().data();
    T* user_i = const_cast<T*>(src_tensor.flat<T>().data());
    T* user_g = const_cast<T*>(diff_dst_tensor.flat<T>().data());
    out_o[0] = user_i[0] >= static_cast<T>(0)
                   ? user_g[0]
                   : user_g[0] * static_cast<T>(this->alpha_);
    return;
  }
};

// register dnn kernels for supported operations and supported types
#define REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklReluGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU_MKL_SUPPORTED_KERNELS_TYPES);

// register dnn kernels for supported operations and supported types
#define REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES(type)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklElu")                                          \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluOp<CPUDevice, type>);                              \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklEluGrad")                                      \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklEluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_ELU_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES(type)        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanh")                                         \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhOp<CPUDevice, type>);                             \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklTanhGrad")                                     \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklTanhGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_TANH_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES(type)       \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6")                                        \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6Op<CPUDevice, type>);                            \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklRelu6Grad")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklRelu6GradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_RELU6_MKL_SUPPORTED_KERNELS_TYPES);

#define REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES(type)   \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyRelu")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluOp<CPUDevice, type>);                        \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklLeakyReluGrad")                                \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklLeakyReluGradOp<CPUDevice, type>);
TF_CALL_float(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_LeakyRelu_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
