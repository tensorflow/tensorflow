/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// See docs in ../ops/math_ops.cc.

// This file uses oneDNN InnerProduct for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) with bias (BiasAdd) operations.
#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// Fuse Operation
template <typename Device, typename T, bool native_format = false>
class MklFusedMatMulOp : public MklDnnMatMulOpBase<T, T> {
 public:
  explicit MklFusedMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    if (AreWeightsFrozen()) {
      this->is_weight_const_ = true;
    } else {
      OP_REQUIRES_OK(
          ctx, ctx->GetAttr("is_filter_const", &(this->is_weight_const_)));
    }

    OP_REQUIRES(ctx, fused_ops_.size() <= 2,
                errors::InvalidArgument(
                    "MklFusedMatMul must have 2 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAdd",
        errors::InvalidArgument(
            "The 1st post-argument of MklFusedMatMul must be BiasAdd."));
    if (fused_ops_.size() > 1 && fused_ops_[1] == "Add") fuse_add_ = true;
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        errors::InvalidArgument("In[0] of MklMatMul can't be transposed."));
    if (fused_ops_.size() == 2 && fused_ops_[1] == "LeakyRelu") {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("leakyrelu_alpha", &leakyrelu_alpha));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMul has 3 inputs: src, weights, bias
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);

    MklDnnShape src_mkl_shape;
    MklDnnShape weight_mkl_shape;
    GetMklShape(ctx, this->kInputIndexSrc, &src_mkl_shape, native_format);
    GetMklShape(ctx, this->kInputIndexWeight, &weight_mkl_shape, native_format);
    OP_REQUIRES(ctx, !weight_mkl_shape.IsMklTensor(),
                errors::InvalidArgument("Weight should not be in MKL Layout"));

    // Get shapes of input tensors
    auto src_tf_shape = src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                                    : src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                errors::InvalidArgument("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                errors::InvalidArgument("In[1] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(bias_tensor.shape()),
                errors::InvalidArgument("Biases must be 1D"));

    // Expression: [batch, k] * [k, channel] + [channel] = [batch, channel]
    //
    // Get dimension size of each matrix, dim_pair[] is the location of k
    // in the inputs, we have constraint that k of the two inputs are
    // the same
    const int dim_pair[] = {1, transpose_b_ ? 1 : 0};
    const int batch = src_tf_shape.dim_size(1 - dim_pair[0]);
    const int k = src_tf_shape.dim_size(dim_pair[0]);
    const int channel = weight_tf_shape.dim_size(1 - dim_pair[1]);

    OP_REQUIRES(
        ctx, k == weight_tf_shape.dim_size(dim_pair[1]),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
            ", In[1]: ", weight_tf_shape.DebugString()));
    OP_REQUIRES(ctx, bias_tensor.shape().dim_size(0) == channel,
                errors::InvalidArgument(
                    "Must provide as many biases as the channel size: ",
                    bias_tensor.shape().DebugString(), " vs. ", channel));

    // For inputs s[batch, k], w[k, channel] and b[channel], the primitive
    // dims should be described like this:
    //   s[batch, k] * w^T[channel, k] + b[channel] = dst[batch, channel]
    //    [n,    ic] *    [oc,     ic] +  [oc]      =    [n,          oc]
    memory::dims src_dims = memory::dims({batch, k});
    // Reverse the weights dims from [k, channel] to [channel, k].
    memory::dims weight_dims = memory::dims({channel, k});
    memory::dims bias_dims = memory::dims({channel});
    memory::dims dst_dims = memory::dims({batch, channel});
    memory::format_tag src_format = memory::format_tag::nc;
    memory::format_tag weight_format =
        transpose_b_ ? memory::format_tag::oi : memory::format_tag::io;

    // Set weight format `any` for primitive as per oneDNN recommendation.
    MklDnnMatMulFwdParams matmul_params(
        src_dims, weight_dims, bias_dims, dst_dims, src_format,
        (this->is_weight_const_) ? memory::format_tag::any : weight_format,
        memory::format_tag::nc, this->is_weight_const_);
    // Extend the basic parameters for data types and fusions.
    ExtendMklDnnMatMulFwdParams(ctx, matmul_params);
#ifdef DNNL_AARCH64_USE_ACL
    // TODO(milpuz01): Remove once Arm Compute Library provides support for
    // in-place updates
    matmul_params.weight_hash =
        Hash64(weight_tensor.tensor_data().data(),
               std::min(kWeightTensorHashLength,
                        static_cast<int>(weight_tensor.tensor_data().size())));
#endif
    MklDnnMatMulFwdPrimitive<T, T, T, T, T>* matmul_prim =
        MklDnnMatMulFwdPrimitiveFactory<T, T, T, T, T>::Get(matmul_params, 0);

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    std::shared_ptr<dnnl::inner_product_forward::primitive_desc> matmul_pd =
        matmul_prim->GetPrimitiveDesc();

    // The output shape of MatMul is same both for MKL and TF version.
    // They are all NC format, no matter what's the format of input.
    // And the shape of AddOp is also the same with output's shape.
    MklDnnShape output_mkl_shape;
    output_mkl_shape.SetMklTensor(false);

    TensorShape output_tf_shape({batch, channel});

    if (fuse_add_) {
      const Tensor& add_tensor = MklGetInput(ctx, kInputIndex_Add);
      MklDnnShape add_mkl_shape;
      GetMklShape(ctx, kInputIndex_Add, &add_mkl_shape, native_format);

      // For native format, we need not to set metadata.
      if (native_format && ctx->forward_input_to_output_with_shape(
                               kInputIndex_Add, kOutputIndex_Dst,
                               output_tf_shape, &dst_tensor)) {
        ;  // Need to do nothing for native format
      } else if (!native_format && ForwardMklTensorInToOutWithMklShape(
                                       ctx, kInputIndex_Add, kOutputIndex_Dst,
                                       &dst_tensor, output_mkl_shape, false)) {
        ;  // If it's not native format, need to forward and set meta first
      } else {
        // If forward is not successful, we should use reorder to copy add
        // tensor to dst tensor
        AllocateOutputSetMklShape(ctx, kOutputIndex_Dst, &dst_tensor,
                                  output_tf_shape, output_mkl_shape,
                                  native_format);
        auto output_format_tag =
            MklTensorFormatToMklDnnDataFormat(MklTensorFormat::FORMAT_NC);
        auto add_md =
            add_mkl_shape.IsMklTensor()
                ? add_mkl_shape.GetMklLayout()
                : memory::desc(dst_dims, MklDnnType<T>(), output_format_tag);
        auto dst_md =
            memory::desc(dst_dims, MklDnnType<T>(), output_format_tag);

        void* add_buf =
            static_cast<void*>(const_cast<T*>(add_tensor.flat<T>().data()));
        void* dst_buf = static_cast<void*>((dst_tensor)->flat<T>().data());

        if (native_format) {
          // We are simply deep copying the add_tensor to dst_tensor without
          // changing memory layout, hence using same memory descriptor.
          add_md = dst_md =
              memory::desc({add_tensor.NumElements()}, MklDnnType<T>(),
                           dnnl::memory::format_tag::x);
        }

        auto fuse_add_src_ = memory(add_md, this->cpu_engine_, add_buf);
        auto fuse_add_dst_ = memory(dst_md, this->cpu_engine_, dst_buf);
        auto reorder_desc =
            ReorderPd(this->cpu_engine_, add_md, this->cpu_engine_, dst_md);

        CreateAndExecuteReorder(reorder_desc, fuse_add_src_, fuse_add_dst_,
                                this->cpu_engine_, ctx);
      }
    } else {
      AllocateOutputSetMklShape(ctx, 0, &dst_tensor, output_tf_shape,
                                output_mkl_shape, native_format);
    }

    // if there's nothing to compute, just return.
    if (batch == 0 || channel == 0) {
      return;
    }

    try {
      // Prepare the input and output for primitive.
      T* src_data = const_cast<T*>(src_tensor.flat<T>().data());
      T* weight_data = const_cast<T*>(weight_tensor.flat<T>().data());
      T* bias_data = const_cast<T*>(bias_tensor.flat<T>().data());
      T* dst_data = const_cast<T*>(dst_tensor->flat<T>().data());

      // Reorder input if necessary.
      MklDnnData<T> src_mkl(&(this->cpu_engine_));
      MklDnnData<T> weight_mkl(&(this->cpu_engine_));

      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T>(), src_format);

      if (src_md != matmul_pd->src_desc()) {
        src_mkl.SetUsrMem(src_md, src_data);
        src_mkl.CheckReorderToOpMem(matmul_pd.get()->src_desc(),
                                    this->cpu_engine_, ctx);
        src_data = reinterpret_cast<T*>(src_mkl.GetOpMem().get_data_handle());
      }

      // Get cached data when weight is const.
      const memory::desc weight_md =
          memory::desc(weight_dims, MklDnnType<T>(), weight_format);
      if (weight_md != matmul_pd->weights_desc()) {
        T* cached_weight_data = nullptr;

        if (this->is_weight_const_) {
          if (this->IsWeightCacheEmpty(ctx)) {
            this->CacheWeight(ctx, matmul_pd, cached_weight_data, weight_tensor,
                              weight_mkl, weight_md);
          }
          cached_weight_data =
              this->GetCachedWeight(ctx, matmul_pd->weights_desc());
        }

        // Cache weight may fail when it gets different format in different
        // iteration. Fallback to reoder if it happens.
        // Also do generel reorder if weight isn't const.
        if (cached_weight_data != nullptr) {
          weight_data = cached_weight_data;
        } else {
          weight_mkl.SetUsrMem(weight_md, weight_data);
          weight_mkl.CheckReorderToOpMem(matmul_pd.get()->weights_desc(),
                                         this->cpu_engine_, ctx);
          weight_data =
              reinterpret_cast<T*>(weight_mkl.GetOpMem().get_data_handle());
        }
      }
      std::shared_ptr<stream> cpu_stream;
      auto st = ExecuteSingleThreadedGemm(batch, channel, k, sizeof(T));
      MklDnnThreadPool eigen_tp(ctx, st ? 1 : -1);
      cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

      UserScratchPad<unsigned char> scratch_pad;
      scratch_pad.AllocateSPTensor(matmul_prim, ctx);

      // Execute fused matmul op.
      matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                           scratch_pad.Get(), cpu_stream);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          ctx, errors::Aborted("Operation received an exception:", error_msg));
    }
  }

  void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                   MklDnnMatMulFwdParams& params) {
    if (fused_ops_.size() == 2) {
      string post_op = fused_ops_[1];

      if (post_op == "Relu") {
        params.post_op_params.push_back({"relu", {1.0, 0.0, 0.0}});
      } else if (post_op == "Relu6") {
        params.post_op_params.push_back({"relu6", {1.0, 6.0, 0.0}});
      } else if (post_op == "Elu") {
        params.post_op_params.push_back({"elu", {1.0, 1.0, 0.0}});
      } else if (post_op == "GeluApproximate") {
        params.post_op_params.push_back({"gelu_approximate", {1.0, 1.0, 0.0}});
      } else if (post_op == "GeluExact") {
        params.post_op_params.push_back({"gelu_exact", {1.0, 1.0, 0.0}});
      } else if (post_op == "Tanh") {
        params.post_op_params.push_back({"tanh", {1.0, 0.0, 0.0}});
      } else if (post_op == "Add") {
        params.post_op_params.push_back({"sum", {1.0}});
      } else if (post_op == "LeakyRelu") {
        params.post_op_params.push_back(
            {"leakyrelu", {1.0, leakyrelu_alpha, 0.0}});
      } else if (post_op == "Sigmoid") {
        params.post_op_params.push_back({"logistic", {1.0, 0.0, 0.0}});
      } else {
        OP_REQUIRES_OK(
            ctx, errors::InvalidArgument(
                     "Unsupported post-argument in MklFusedMatMul: ", post_op));
      }
    }
  }

 private:
  bool fuse_add_ = false;
  bool transpose_a_;
  bool transpose_b_;
  float leakyrelu_alpha = 0.2;
  std::vector<string> fused_ops_;
  const int kInputIndex_Add = 3;
  const int kOutputIndex_Dst = 0;
#ifdef DNNL_AARCH64_USE_ACL
  const int kWeightTensorHashLength = 1024;
#endif
};  // namespace tensorflow

// Register mkl kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type)                \
  REGISTER_KERNEL_BUILDER(                                                    \
      Name("_MklFusedMatMul")                                                 \
          .Device(DEVICE_CPU)                                                 \
          .TypeConstraint<type>("T")                                          \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),                \
      MklFusedMatMulOp<CPUDevice, type>);                                     \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeFusedMatMul")                       \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<type>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklFusedMatMulOp<CPUDevice, type, true>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
