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

// This file uses MKL-DNN InnerProduct for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) with bias (BiasAdd) operations.
#ifdef INTEL_MKL

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {

// Fuse Operation
template <typename Device, typename T>
class MklFusedMatMulOp : public MklDnnMatMulOpBase<T, T> {
 public:
  explicit MklFusedMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T, T>(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("fused_ops", &fused_ops_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("is_filter_const", &(this->is_weight_const_)));

    OP_REQUIRES(ctx, fused_ops_.size() <= 2,
                errors::InvalidArgument(
                    "MklFusedMatMul must have 2 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAdd",
        errors::InvalidArgument(
            "The 1st post-argument of MklFusedMatMul must be BiasAdd."));
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        errors::InvalidArgument("In[0] of MklMatMul can't be transposed."));
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMul has 3 inputs: src, weights, bias
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);

    MklDnnShape src_mkl_shape;
    MklDnnShape weight_mkl_shape;
    GetMklShape(ctx, this->kInputIndexSrc, &src_mkl_shape);
    GetMklShape(ctx, this->kInputIndexWeight, &weight_mkl_shape);
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

    OP_REQUIRES(ctx, k == weight_tf_shape.dim_size(dim_pair[1]),
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        src_tf_shape.DebugString(), ", In[1]: ",
                                        weight_tf_shape.DebugString()));
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
    memory::format weight_format =
        transpose_b_ ? memory::format::oi : memory::format::io;

    // Set weight format for primitive:
    //   1. const, let MKL-DNN determine format because it will be cached;
    //   2. var, keep the original format to avoid reordering.
    MklDnnMatMulFwdParams matmul_params(
        src_dims, weight_dims, bias_dims, dst_dims,
        (this->is_weight_const_) ? memory::format::any : weight_format);

    // Extend the basic parameters for data types and fusions.
    ExtendMklDnnMatMulFwdParams(ctx, matmul_params);
    MklDnnMatMulFwdPrimitive<T, T, T, T, T>* matmul_prim =
        MklDnnMatMulFwdPrimitiveFactory<T, T, T, T, T>::Get(matmul_params, 0);

    // Allocate output tensor.
    Tensor* dst_tensor = nullptr;
    std::shared_ptr<mkldnn::inner_product_forward::primitive_desc> matmul_pd =
        matmul_prim->GetPrimitiveDesc();

    if (src_mkl_shape.IsMklTensor()) {
      this->AllocateOutputTensor(ctx, *matmul_pd, dst_dims, memory::format::nc,
                                 &dst_tensor);
    } else {
      TensorShape dst_tensor_shape({batch, channel});
      MklDnnShape dst_mkl_shape;
      dst_mkl_shape.SetMklTensor(false);
      AllocateOutputSetMklShape(ctx, 0, &dst_tensor, dst_tensor_shape,
                                dst_mkl_shape);
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

      if (src_mkl_shape.IsMklTensor()) {
        memory::desc input_md = src_mkl_shape.GetMklLayout();

        if (input_md.data.format != memory::format::nc) {
          src_mkl.SetUsrMem(input_md, src_data);
          src_mkl.CheckReorderToOpMem(matmul_pd.get()->src_primitive_desc());
          src_data = reinterpret_cast<T*>(src_mkl.GetOpMem().get_data_handle());
        }
      }

      // Get cached data when weight is const.
      memory::format expected_format = matmul_prim->GetweightMemoryFormat();
      DCHECK(expected_format != weight_format && this->is_weight_const_);
      if (this->is_weight_const_) {
        T* cached_weight_data = nullptr;
        if (this->IsWeightCacheEmpty(ctx)) {
          auto weight_md =
              memory::desc(weight_dims, MklDnnType<T>(), weight_format);
          this->CacheWeight(ctx, matmul_pd, cached_weight_data, weight_tensor,
                            weight_mkl, weight_md);
        }
        cached_weight_data = this->GetCachedWeight(ctx, expected_format);

        // Cache weight may fail when it gets different format in different
        // iteration. Fallback to reoder if it happens.
        // TODO: Fix this slow path.
        if (cached_weight_data != nullptr) {
          weight_data = cached_weight_data;
        } else {
          memory::desc input_md =
              memory::desc(weight_dims, MklDnnType<T>(), weight_format);

          weight_mkl.SetUsrMem(input_md, weight_data);
          weight_mkl.CheckReorderToOpMem(
              matmul_pd.get()->weights_primitive_desc());
          weight_data =
              reinterpret_cast<T*>(weight_mkl.GetOpMem().get_data_handle());
        }
      }

      matmul_prim->Execute(src_data, weight_data, bias_data, dst_data);
    } catch (mkldnn::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
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
      } else {
        OP_REQUIRES_OK(
            ctx, errors::InvalidArgument(
                     "Unsupported post-argument in MklFusedMatMul: ", post_op));
      }
    }
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  std::vector<string> fused_ops_;
};

// Register mkl kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type) \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklFusedMatMul")                                  \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                           \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel), \
      MklFusedMatMulOp<CPUDevice, type>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);

}  // namespace tensorflow

#endif  // INTEL_MKL
