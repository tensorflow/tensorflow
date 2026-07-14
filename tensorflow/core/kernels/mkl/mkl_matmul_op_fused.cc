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

// See docs in ../ops/math_ops.cc.

// This file uses oneDNN InnerProduct for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) with bias (BiasAdd) operations.
#if defined(INTEL_MKL)

#include <type_traits>

#include "oneapi/dnnl/dnnl.hpp"
#include "absl/container/inlined_vector.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/mkl/mkl_quantized_conv_ops.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {

// Fuse Operation
template <typename Device, typename T1, typename T2, typename Tbias,
          typename Toutput, typename U, bool native_format = false>
class MklFusedMatMulOp : public MklDnnMatMulOpBase<T2, Tbias, Toutput> {
 public:
  explicit MklFusedMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T2, Tbias, Toutput>(ctx) {
    if (std::is_same<T2, qint8>::value) {
      return;  // Quantized version will have own contstruction code.
    }
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
                absl::InvalidArgumentError(
                    "MklFusedMatMul must have 2 post-arguments at most."));
    OP_REQUIRES(
        ctx, fused_ops_[0] == "BiasAdd",
        absl::InvalidArgumentError(
            "The 1st post-argument of MklFusedMatMul must be BiasAdd."));
    if (fused_ops_.size() > 1 && fused_ops_[1] == "Add") fuse_add_ = true;
    OP_REQUIRES(
        ctx, transpose_a_ == false,
        absl::InvalidArgumentError("In[0] of MklMatMul can't be transposed."));
    if (fused_ops_.size() == 2 && fused_ops_[1] == "LeakyRelu") {
      OP_REQUIRES_OK(ctx, ctx->GetAttr("leakyrelu_alpha", &leakyrelu_alpha_));
    }
  }

  void Compute(OpKernelContext* ctx) override {
    // FusedMatMul has 3 inputs: src, weights, bias
    const Tensor& src_tensor = ctx->input(this->kInputIndexSrc);
    const Tensor& weight_tensor = ctx->input(this->kInputIndexWeight);
    const Tensor& bias_tensor = MklGetInput(ctx, this->kInputIndexBias);

    if (std::is_same<T1, float>::value) {
      (void)SetFPMathMode();
    }

    MklDnnShape src_mkl_shape;
    MklDnnShape weight_mkl_shape;
    GetMklShape(ctx, this->kInputIndexSrc, &src_mkl_shape, native_format);
    GetMklShape(ctx, this->kInputIndexWeight, &weight_mkl_shape, native_format);
    OP_REQUIRES(
        ctx, !weight_mkl_shape.IsMklTensor(),
        absl::InvalidArgumentError("Weight should not be in MKL Layout"));

    // Get shapes of input tensors
    auto src_tf_shape = src_mkl_shape.IsMklTensor() ? src_mkl_shape.GetTfShape()
                                                    : src_tensor.shape();
    auto weight_tf_shape = weight_tensor.shape();

    // Check the constraint of input matrix and bias
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(src_tf_shape),
                absl::InvalidArgumentError("In[0] is not a matrix"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(weight_tf_shape),
                absl::InvalidArgumentError("In[1] is not a matrix"));
    for (int i = 0; i < bias_tensor.dims() - 1; i++) {
      OP_REQUIRES(ctx, bias_tensor.dim_size(i) == 1,
                  absl::InvalidArgumentError(
                      absl::StrCat("For bias_dims > 1, all except the "
                                   "last dimension (channel) must be 1, got: ",
                                   bias_tensor.shape().DebugString())));
    }

    // Expression: [batch, k] * [k, channel] + [channel] = [batch, channel]
    //
    // Get dimension size of each matrix, dim_pair[] is the location of k
    // in the inputs, we have constraint that k of the two inputs are
    // the same
    const int64_t dim_pair[] = {1, transpose_b_ ? 1 : 0};
    const int64_t batch = src_tf_shape.dim_size(1 - dim_pair[0]);
    const int64_t k = src_tf_shape.dim_size(dim_pair[0]);
    const int64_t channel = weight_tf_shape.dim_size(1 - dim_pair[1]);

    OP_REQUIRES(
        ctx, k == weight_tf_shape.dim_size(dim_pair[1]),
        absl::InvalidArgumentError(absl::StrCat(
            "Matrix size-incompatible: In[0]: ", src_tf_shape.DebugString(),
            ", In[1]: ", weight_tf_shape.DebugString())));
    OP_REQUIRES(ctx, bias_tensor.dim_size(bias_tensor.dims() - 1) == channel,
                absl::InvalidArgumentError(absl::StrCat(
                    "Must provide as many biases as the channel size: ",
                    bias_tensor.shape().DebugString(), " vs. ", channel)));

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
    auto st = ExecuteSingleThreadedGemm(batch, channel, k, sizeof(T1));
    // Create the oneDNN wrapper over Eigen threadpool and set max threads
    // in oneDNN.
    Eigen::ThreadPoolInterface* eigen_interface =
        EigenThreadPoolFromTfContext(ctx);
    tsl::OneDnnThreadPool eigen_tp(eigen_interface, ThreadPoolUseCallerThread(),
                                   st ? 1 : -1);
    MklDnnMatMulFwdPrimitive<float, T1, T2, Tbias, Toutput>* matmul_prim =
        MklDnnMatMulFwdPrimitiveFactory<float, T1, T2, Tbias, Toutput>::Get(
            matmul_params, 0);

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
      const Tensor& add_tensor = MklGetInput(ctx, input_idx_add_);
      MklDnnShape add_mkl_shape;
      GetMklShape(ctx, input_idx_add_, &add_mkl_shape, native_format);

      // For native format, we need not to set metadata.
      if (native_format &&
          ctx->forward_input_to_output_with_shape(
              input_idx_add_, kOutputIndex_Dst, output_tf_shape, &dst_tensor)) {
        ;  // Need to do nothing for native format
      } else if (!native_format && ForwardMklTensorInToOutWithMklShape(
                                       ctx, input_idx_add_, kOutputIndex_Dst,
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
                : memory::desc(dst_dims, MklDnnType<U>(), output_format_tag);
        auto dst_md =
            memory::desc(dst_dims, MklDnnType<Toutput>(), output_format_tag);

        void* add_buf =
            static_cast<void*>(const_cast<U*>(add_tensor.flat<U>().data()));
        void* dst_buf =
            static_cast<void*>((dst_tensor)->flat<Toutput>().data());

        if (native_format) {
          // We are simply deep copying the add_tensor to dst_tensor without
          // changing memory layout, hence using same memory descriptor.
          add_md = dst_md =
              memory::desc({add_tensor.NumElements()}, MklDnnType<U>(),
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
      T1* src_data = const_cast<T1*>(src_tensor.flat<T1>().data());
      T2* weight_data = const_cast<T2*>(weight_tensor.flat<T2>().data());
      void* bias_data = static_cast<void*>(
          const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
      Toutput* dst_data =
          const_cast<Toutput*>(dst_tensor->flat<Toutput>().data());

      // Reorder input if necessary.
      MklDnnData<T1> src_mkl(&(this->cpu_engine_));
      MklDnnData<T2> weight_mkl(&(this->cpu_engine_));

      auto src_md = src_mkl_shape.IsMklTensor()
                        ? src_mkl_shape.GetMklLayout()
                        : memory::desc(src_dims, MklDnnType<T1>(), src_format);

      if (src_md != matmul_pd->src_desc()) {
        src_mkl.SetUsrMem(src_md, src_data);
        src_mkl.CheckReorderToOpMem(matmul_pd.get()->src_desc(),
                                    this->cpu_engine_, ctx);
        src_data = static_cast<T1*>(src_mkl.GetOpMem().get_data_handle());
      }

      // Get cached data when weight is const.
      const memory::desc weight_md =
          memory::desc(weight_dims, MklDnnType<T2>(), weight_format);
      if (weight_md != matmul_pd->weights_desc()) {
        T2* cached_weight_data = nullptr;

        if (this->is_weight_const_) {
          // TODO(intel-tf): When oneDNN major version changes to v4.x, weight
          // caching may not work as expected if the underlying memory
          // descriptor has changed (i.e. compared to v3.x). We have to return
          // a status here to catch oneDNN major version change to avoid
          // unexpected results.
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
              static_cast<T2*>(weight_mkl.GetOpMem().get_data_handle());
        }
      }
      std::shared_ptr<stream> cpu_stream;
      cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

      UserScratchPad<unsigned char> scratch_pad;
      scratch_pad.AllocateSPTensor(matmul_prim, ctx);

      // Temporary tensor for scaled bias when op is quantized version.
      Tensor temp_scaled_bias_tensor;
      if (std::is_same<T2, qint8>::value) {
        this->GetScaledBias(ctx, matmul_pd, bias_tensor,
                            &temp_scaled_bias_tensor, &bias_data);
      }

      // Execute fused matmul op.
      matmul_prim->Execute(src_data, weight_data, bias_data, dst_data,
                           matmul_params, scratch_pad.Get(), cpu_stream);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) +
                         ", message: " + string(e.message) + ", in file " +
                         string(__FILE__) + ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(ctx, absl::AbortedError(absl::StrCat(
                              "Operation received an exception:", error_msg)));
    }
  }

  virtual void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                           MklDnnMatMulFwdParams& params) {
    // Create a string from data types of input, weight, bias, and output.
    params.dtypes.append(typeid(T1).name());
    params.dtypes.append(typeid(T2).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());
    if (fused_ops_.size() == 2) {
      string post_op = fused_ops_[1];
      float scale = 1.0f;
      float alpha = 0.0f;
      float beta = 0.0f;
      if (post_op == "Relu6") {
        alpha = 6.0f;
      } else if (post_op == "LeakyRelu") {
        alpha = leakyrelu_alpha_;
      } else if (post_op == "Elu") {
        alpha = 1.0f;
      }
      if (post_op == "Relu" || post_op == "Relu6" || post_op == "LeakyRelu" ||
          post_op == "Elu" || post_op == "GeluApproximate" ||
          post_op == "GeluExact" || post_op == "Tanh" || post_op == "Sigmoid") {
        params.post_op_params.push_back({post_op, {scale, alpha, beta}});
      } else if (post_op == "Add") {
        params.post_op_params.push_back({"sum", {1.0}});
      } else {
        OP_REQUIRES_OK(ctx, absl::InvalidArgumentError(absl::StrCat(
                                "Unsupported post-argument in MklFusedMatMul: ",
                                post_op)));
      }
    }
  }

 protected:
  virtual void GetScaledBias(
      OpKernelContext*,
      std::shared_ptr<dnnl::inner_product_forward::primitive_desc>&,
      const Tensor&, Tensor*, void**) {}

  bool fuse_add_ = false;
  bool transpose_a_;
  bool transpose_b_;
  float leakyrelu_alpha_ = 0.2;
  std::vector<string> fused_ops_;
  int input_idx_add_ = 3;
  const int kOutputIndex_Dst = 0;
#ifdef DNNL_AARCH64_USE_ACL
  const int kWeightTensorHashLength = 1024;
#endif
};

namespace {

enum class FusedComputationType {
  kUndefined,
  kBiasAdd,
  kBiasAdd_Dequantize,
  kBiasAdd_Requantize,
  kBiasAdd_Activation,
  kBiasAdd_Activation_Dequantize,
  kBiasAdd_Activation_Requantize,
  kBiasAdd_Add,
  kBiasAdd_Add_Dequantize,
  kBiasAdd_Add_Requantize,
};

struct FusedComputationPattern {
  FusedComputationType fused_computation;
  std::vector<string> fused_ops;
};

}  // namespace

// OneDNN uses post-ops to implement different kind of fusions. The category of
// each individual post-op can be inferred from the fused_ops attribute. The
// following enum is used to identify list of required post-ops.
enum class PostOpKind { kActivation, kSum, kOutputScale, kLinear };

template <typename Device, typename T1, typename T2, typename Tbias,
          typename Toutput, typename U, bool native_format = true>
class QuantizedFusedMatMulOp
    : public MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, U,
                              native_format> {
 protected:
  string input_quant_mode_;   // 0-th input
  string output_quant_mode_;  // 0-th output
  string activation_type_;    // Activation op type

  // Initialize minmax tensor indices with default values for the most common
  // cases.
  int input_min_idx_ = 3;
  int input_max_idx_ = 4;
  int weight_min_idx_ = 5;
  int weight_max_idx_ = 6;

  struct PostOpInfo {
    PostOpKind post_op_kind;
    struct OperandInfo {
      int idx = -1;  // Operand tensor index if needed by a post-op.
      // Indices of min and max value tensors, if the operand is quantized.
      absl::InlinedVector<int, 4> min_max_indices;
    } operand_info;
    // Indices of output min and max value tensors. It is used when requantize
    // is fused.
    absl::InlinedVector<int, 4> min_max_indices;
  };

  absl::InlinedVector<PostOpInfo, 4> post_op_info_list_;

  void Initialize(OpKernelConstruction* context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_a", &this->transpose_a_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("transpose_b", &this->transpose_b_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("input_quant_mode", &input_quant_mode_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("output_quant_mode", &output_quant_mode_));
    OP_REQUIRES_OK(
        context, context->GetAttr("is_weight_const", &this->is_weight_const_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("is_bias_const", &this->is_bias_const_));
    if (context->HasAttr("leakyrelu_alpha")) {
      OP_REQUIRES_OK(context, context->GetAttr("leakyrelu_alpha",
                                               &this->leakyrelu_alpha_));
    }

    // Extract activation info and canonicalize activation types to
    // common name "Activation" in the fused_ops attribute.
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    for (auto it = fused_ops.begin(); it != fused_ops.end(); ++it) {
      if (*it == "Relu" || *it == "Relu6" || *it == "Elu" ||
          *it == "GeluApproximate" || *it == "GeluExact" || *it == "Tanh" ||
          *it == "LeakyRelu" || *it == "Sigmoid") {
        if (*it != "Relu") {
          string last_fusion = fused_ops.back();
          OP_REQUIRES(
              context,
              (last_fusion == "Dequantize" || last_fusion == "Requantize"),
              absl::UnimplementedError(absl::StrCat(
                  "Nonlinear activation except Relu can be ",
                  "supported only with Dequantize or Requantize fusion.")));
        }
        activation_type_ = *it;
        // Canonicalize all activation types into "Activation" for simplifying
        // post ops construction.
        *it = "Activation";
      }
    }

    using FCT = FusedComputationType;

    // TODO(intel-tf): Add more patterns when implemented.
    std::vector<FusedComputationPattern> patterns{
        {FCT::kBiasAdd, {"BiasAdd"}},
        {FCT::kBiasAdd_Dequantize, {"BiasAdd", "Dequantize"}},
        {FCT::kBiasAdd_Requantize, {"BiasAdd", "Requantize"}},
        {FCT::kBiasAdd_Activation, {"BiasAdd", "Activation"}},
        {FCT::kBiasAdd_Activation_Dequantize,
         {"BiasAdd", "Activation", "Dequantize"}},
        {FCT::kBiasAdd_Activation_Requantize,
         {"BiasAdd", "Activation", "Requantize"}},
        {FCT::kBiasAdd_Add_Dequantize, {"BiasAdd", "Add", "Dequantize"}},
    };

    FusedComputationType fused_computation = FusedComputationType::kUndefined;
    for (const auto& pattern : patterns) {
      if (fused_ops == pattern.fused_ops) {
        fused_computation = pattern.fused_computation;
        break;
      }
    }

    // Configure oneDNN post ops
    switch (fused_computation) {
      case FCT::kBiasAdd:
        // No post op is required.
        OP_REQUIRES(context, (std::is_same<Toutput, qint32>::value),
                    absl::UnimplementedError(absl::StrCat(
                        "Qunatized fusion: [", absl::StrJoin(fused_ops, ","),
                        "] needs output in qint32.")));
        break;
      case FCT::kBiasAdd_Dequantize:
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}}};
        break;
      case FCT::kBiasAdd_Requantize:
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kLinear, {}, {7, 8}}};
        break;
      case FCT::kBiasAdd_Activation:
        OP_REQUIRES(context,
                    (std::is_same<Toutput, qint32>::value &&
                     activation_type_ == "Relu"),
                    absl::UnimplementedError(absl::StrCat(
                        "Qunatized fusion: [", absl::StrJoin(fused_ops, ","),
                        "] needs output in qint32 and ",
                        "activation supported is only Relu")));
        post_op_info_list_ = {{PostOpKind::kActivation, {}, {}}};
        break;
      case FCT::kBiasAdd_Activation_Dequantize:
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kActivation, {}, {}}};
        break;
      case FCT::kBiasAdd_Activation_Requantize:
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kActivation, {}, {}},
                              {PostOpKind::kLinear, {}, {7, 8}}};
        break;
      case FCT::kBiasAdd_Add_Dequantize: {
        OP_REQUIRES(
            context,
            (std::is_same<U, float>::value || std::is_same<U, bfloat16>::value),
            absl::UnimplementedError(
                "Quantized addend tensor is not implemented yet."));
        // Addend tensor precedes all minmax tensors. Shift the indices from
        // default initilized values.
        input_min_idx_ += 1;
        input_max_idx_ += 1;
        weight_min_idx_ += 1;
        weight_max_idx_ += 1;
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kSum, {3, {}}, {}}};
      } break;
      default:
        OP_REQUIRES(context, false,
                    absl::UnimplementedError(
                        absl::StrCat("Fusion is not implemented: [",
                                     absl::StrJoin(fused_ops, ","), "]")));
    }
  }

 public:
  explicit QuantizedFusedMatMulOp(OpKernelConstruction* context)
      : MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, U, true>(context) {
    Initialize(context);
  }

  void Compute(OpKernelContext* ctx) override {
    MklFusedMatMulOp<Device, T1, T2, Tbias, Toutput, U, true>::Compute(ctx);
    // Compute additional outputs
    if (std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, quint8>::value ||
        std::is_same<Toutput, qint32>::value) {
      Tensor* min_output = nullptr;
      Tensor* max_output = nullptr;

      const float min_input = ctx->input(input_min_idx_).flat<float>()(0);
      const float max_input = ctx->input(input_max_idx_).flat<float>()(0);
      const Tensor& min_weight = ctx->input(weight_min_idx_);
      const Tensor& max_weight = ctx->input(weight_max_idx_);
      OP_REQUIRES(ctx, min_weight.shape() == max_weight.shape(),
                  absl::InvalidArgumentError(
                      "Shape of min-weight and max-weight must be same."));

      if (std::is_same<Toutput, qint32>::value) {
        TensorShape output_minmax_shape = min_weight.shape();
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(1, output_minmax_shape, &min_output));
        OP_REQUIRES_OK(
            ctx, ctx->allocate_output(2, output_minmax_shape, &max_output));
        if (min_weight.dims() == 0) {
          float min_output_value;
          float max_output_value;
          MklQuantizationRangeForMultiplication<T1, T2, qint32>(
              min_input, max_input, min_weight.flat<float>()(0),
              max_weight.flat<float>()(0), &min_output_value,
              &max_output_value);
          min_output->flat<float>()(0) = min_output_value;
          max_output->flat<float>()(0) = max_output_value;
        } else {
          MklQuantizationRangeForMultiplication<T1, T2, qint32>(
              min_input, max_input, min_weight, max_weight, &min_output,
              &max_output);
        }
      } else {
        // When output type is qint8 or quint8, the kernel is registered for
        // Requantize fusion.
        OP_REQUIRES_OK(ctx, ctx->allocate_output(1, {}, &min_output));
        OP_REQUIRES_OK(ctx, ctx->allocate_output(2, {}, &max_output));
        int output_min_idx = ctx->num_inputs() - 2;
        int output_max_idx = ctx->num_inputs() - 1;
        const float requested_min = ctx->input(output_min_idx).flat<float>()(0);
        const float requested_max = ctx->input(output_max_idx).flat<float>()(0);
        if (output_quant_mode_ == "SCALED") {
          const float range_output =
              std::max(std::abs(requested_min), std::abs(requested_max));
          if (std::is_same<Toutput, qint8>::value) {
            min_output->flat<float>()(0) = -range_output;
            max_output->flat<float>()(0) = range_output;
          } else {
            min_output->flat<float>()(0) = 0;
            max_output->flat<float>()(0) = range_output;
          }
        } else {
          min_output->flat<float>()(0) = requested_min;
          max_output->flat<float>()(0) = requested_max;
        }
      }
    } else if (std::is_same<Toutput, float>::value ||
               std::is_same<Toutput, bfloat16>::value) {
      // Kernel is registered for Dequantization fusion. Nothing to do.
    } else {
      OP_REQUIRES_OK(ctx,
                     absl::InvalidArgumentError("Unsupported output type."));
    }
  }

  void ExtendMklDnnMatMulFwdParams(OpKernelContext* ctx,
                                   MklDnnMatMulFwdParams& params) override {
    // Create a string from data types of input, weight, bias, and output.
    params.dtypes.append(typeid(T1).name());
    params.dtypes.append(typeid(T2).name());
    params.dtypes.append(typeid(Tbias).name());
    params.dtypes.append(typeid(Toutput).name());

    params.input_quant_mode = input_quant_mode_;

    for (const auto& post_op_info : post_op_info_list_) {
      auto post_op_kind = post_op_info.post_op_kind;
      switch (post_op_kind) {
        case PostOpKind::kOutputScale: {
          if constexpr (std::is_same<Toutput, qint32>::value) {
            // No scaling is required.
            break;
          }
          const float min_input = ctx->input(input_min_idx_).flat<float>()(0);
          const float max_input = ctx->input(input_max_idx_).flat<float>()(0);
          const Tensor& min_weight_tensor = ctx->input(weight_min_idx_);
          const Tensor& max_weight_tensor = ctx->input(weight_max_idx_);
          const float* min_weight = min_weight_tensor.flat<float>().data();
          const float* max_weight = max_weight_tensor.flat<float>().data();
          const size_t num_weight_scales = min_weight_tensor.NumElements();

          const float max_int8_input =
              (std::is_same<T1, quint8>::value) ? 255.0f : 127.0f;
          const float max_int8_weight =
              (std::is_same<T2, quint8>::value) ? 255.0f : 127.0f;
          const float range_input =
              (input_quant_mode_ == "MIN_FIRST")
                  ? max_input - min_input
                  : std::max(std::abs(min_input), std::abs(max_input));

          const float src_scale = range_input / max_int8_input;
          std::vector<float> wei_scales(num_weight_scales);
#ifndef ENABLE_ONEDNN_V3
          std::vector<float> output_scales(num_weight_scales);
#endif  // ENABLE_ONEDNN_V3
          for (size_t i = 0; i < num_weight_scales; ++i) {
            float range_weight =
                std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
            wei_scales[i] = range_weight / max_int8_weight;
#ifndef ENABLE_ONEDNN_V3
            output_scales[i] = src_scale * wei_scales[i];
#endif  // ENABLE_ONEDNN_V3
          }
          FactoryKeyCreator src_partial_key;
          src_partial_key.AddAsKey<float>(min_input);
          src_partial_key.AddAsKey<float>(max_input);

          FactoryKeyCreator wei_partial_key;
          wei_partial_key.AddAsKey<const float*>(min_weight);
          wei_partial_key.AddAsKey<const float*>(max_weight);
#ifndef ENABLE_ONEDNN_V3
          FactoryKeyCreator output_scales_partial_key;
          output_scales_partial_key.AddAsKey(src_partial_key.GetKey());
          output_scales_partial_key.AddAsKey(wei_partial_key.GetKey());
          params.post_op_params.push_back({"output_scale", output_scales,
                                           output_scales_partial_key.GetKey()});
#else
          params.post_op_params.push_back(
              {"src_scale", {src_scale}, src_partial_key.GetKey()});
          params.post_op_params.push_back(
              {"wei_scale", wei_scales, wei_partial_key.GetKey()});
#endif  // ENABLE_ONEDNN_V3
        } break;

        case PostOpKind::kActivation: {
          float scale = 1.0f;
          float alpha = 0.0f;
          float beta = 0.0f;
          if (activation_type_ == "LeakyRelu")
            alpha = this->leakyrelu_alpha_;
          else if (activation_type_ == "Relu6")
            alpha = 6.0f;
          else if (activation_type_ == "Elu")
            alpha = 1.0f;
          params.post_op_params.push_back(
              {activation_type_, {scale, alpha, beta}});
        } break;

        case PostOpKind::kLinear: {
          // Update output_scale for requantize fusion.
          auto output_min_idx = post_op_info.min_max_indices[0];
          auto output_max_idx = post_op_info.min_max_indices[1];
          const float min_output =
              ctx->input(output_min_idx).template flat<float>()(0);
          const float max_output =
              ctx->input(output_max_idx).template flat<float>()(0);
          const float max_int8_output =
              (std::is_same<Toutput, quint8>::value) ? 255.0f : 127.0f;
          const float range_output =
              (output_quant_mode_ == "MIN_FIRST")
                  ? max_output - min_output
                  : std::max(std::abs(min_output), std::abs(max_output));
          float req_scale = max_int8_output / range_output;
          float req_shift = 0.0f;
          if (output_quant_mode_ == "MIN_FIRST") {
            req_shift = -min_output * max_int8_output / range_output;
          }
          params.post_op_params.push_back(
              {"linear", {1.0, req_scale, req_shift}});
        } break;

        case PostOpKind::kSum: {
          this->fuse_add_ = true;
          this->input_idx_add_ = post_op_info.operand_info.idx;
          params.post_op_params.push_back({"sum", {1.0}});
        } break;

        default:
          OP_REQUIRES_OK(
              ctx, absl::InvalidArgumentError("Unsupported post-op-kind."));
      }
    }
  }

  void GetScaledBias(
      OpKernelContext* ctx,
      std::shared_ptr<dnnl::inner_product_forward::primitive_desc>& matmul_pd,
      const Tensor& bias_tensor, Tensor* temp_scaled_bias_tensor,
      void** bias_data) override {
#ifdef ENABLE_ONEDNN_V3
#define TSCALED_BIAS float
#else
#define TSCALED_BIAS Tbias
#endif  // ENABLE_ONEDNN_V3

#ifndef ENABLE_ONEDNN_V3
    if (std::is_same<Tbias, qint32>::value) {
      // Bias already has been scaled for quantized input and weight.
#else
    if ((std::is_same<Tbias, float>::value ||
         std::is_same<Tbias, bfloat16>::value) &&
        input_quant_mode_ == "SCALED") {
#endif  // !ENABLE_ONEDNN_V3
      return;
    } else {
      const float min_input = ctx->input(input_min_idx_).flat<float>()(0);
      const float max_input = ctx->input(input_max_idx_).flat<float>()(0);
      const Tensor& min_weight_tensor = ctx->input(weight_min_idx_);
      const Tensor& max_weight_tensor = ctx->input(weight_max_idx_);
      const float* min_weight = min_weight_tensor.flat<float>().data();
      const float* max_weight = max_weight_tensor.flat<float>().data();
      bool is_cached_bias_valid = false;
      bool is_bias_cache_empty = this->IsBiasCacheEmpty();
      if (!is_bias_cache_empty) {
        this->GetCachedBias(min_input, max_input, bias_data);
        is_cached_bias_valid = (*bias_data != nullptr);
      }
      if (!is_cached_bias_valid) {
        void* input_bias_buf = static_cast<void*>(
            const_cast<Tbias*>(bias_tensor.flat<Tbias>().data()));
        auto scaled_bias_md = matmul_pd->bias_desc();
        TensorShape scaled_bias_shape;
        scaled_bias_shape.AddDim((scaled_bias_md.get_size() / sizeof(float)));
        OP_REQUIRES_OK(ctx, ctx->allocate_temp(
                                DataTypeToEnum<TSCALED_BIAS>::v(),
                                scaled_bias_shape, temp_scaled_bias_tensor));
        void* scaled_bias_buf = static_cast<void*>(
            temp_scaled_bias_tensor->flat<TSCALED_BIAS>().data());

        const float max_int8_input =
            (std::is_same<T1, quint8>::value) ? 255.0f : 127.0f;
        const float max_int8_weight =
            (std::is_same<T2, quint8>::value) ? 255.0f : 127.0f;
        const float range_input =
            (input_quant_mode_ == "MIN_FIRST")
                ? max_input - min_input
                : std::max(std::abs(min_input), std::abs(max_input));
        const size_t num_weight_scales = min_weight_tensor.NumElements();
        std::vector<float> bias_scales(num_weight_scales, 1.0);
        for (size_t i = 0; i < num_weight_scales; ++i) {
          float range_weight =
              std::max(std::abs(min_weight[i]), std::abs(max_weight[i]));
          float scale_factor =
              (max_int8_input * max_int8_weight) / (range_input * range_weight);
          bias_scales[i] = scale_factor;
        }
        if (input_quant_mode_ == "MIN_FIRST") {
          Tbias* input_bias = (Tbias*)input_bias_buf;
          TSCALED_BIAS* adjusted_bias = (TSCALED_BIAS*)scaled_bias_buf;
          float q_min_input = max_int8_input * min_input / range_input;
          const Tensor& weight_tensor = ctx->input(1);
          int stride_ic = 1;
          int stride_oc = 1;
          int k = 0;
          int n = 0;
          if (this->transpose_b_) {
            k = weight_tensor.dim_size(1);
            n = weight_tensor.dim_size(0);
            stride_ic = 1;
            stride_oc = k;
          } else {
            k = weight_tensor.dim_size(0);
            n = weight_tensor.dim_size(1);
            stride_ic = n;
            stride_oc = 1;
          }
          T2* weight_buf = const_cast<T2*>(weight_tensor.flat<T2>().data());
          std::vector<float> scales(n);
          if (num_weight_scales == 1) {
            // Weights are quantized per_tensor. Scales need to be expanded to
            // number of output channels.
            std::fill(scales.begin(), scales.end(), bias_scales[0]);
          } else {
            scales = bias_scales;
          }
          // TODO(intel-tf): Paralellize loop for large weights.
          for (int j = 0; j < n; ++j) {
            int sum = 0;
            for (int i = 0; i < k; ++i) {
              sum += weight_buf[i * stride_ic + j * stride_oc];
            }
#ifndef ENABLE_ONEDNN_V3
            adjusted_bias[j] = static_cast<TSCALED_BIAS>(
                (static_cast<float>(input_bias[j]) * scales[j]) +
                (sum * q_min_input));
#else
            // TODO(intel-tf): Use zeropoint for quantized input tensor instead
            // of manual adjustments.
            if (std::is_same<Tbias, qint32>::value) {
              // Starting with oneDNN v3.0, bias is expected to be dequantized
              // to float32.
              adjusted_bias[j] = static_cast<float>(input_bias[j]) / scales[j];
            } else {
              // Bias is float32 or bfloat16 but still needs to be compensated.
              adjusted_bias[j] = static_cast<float>(input_bias[j]) +
                                 ((sum * q_min_input) / scales[j]);
            }
#endif  // !ENABLE_ONEDNN_V3
          }
        } else {
          memory::dims input_bias_dims =
              memory::dims({bias_tensor.shape().dim_size(0)});
          auto input_bias_md = dnnl::memory::desc(
              input_bias_dims, MklDnnType<Tbias>(), memory::format_tag::x);
          auto input_bias_mem =
              dnnl::memory(input_bias_md, this->cpu_engine_, input_bias_buf);
          auto scaled_bias_mem =
              dnnl::memory(scaled_bias_md, this->cpu_engine_, scaled_bias_buf);
          dnnl::primitive_attr bias_attr;
#ifndef ENABLE_ONEDNN_V3
          (num_weight_scales == 1)
              ? bias_attr.set_output_scales(0, bias_scales)
              : bias_attr.set_output_scales(1, bias_scales);
#else
          (num_weight_scales == 1) ? bias_attr.set_scales_mask(DNNL_ARG_SRC, 0)
                                   : bias_attr.set_scales_mask(DNNL_ARG_SRC, 1);
#endif  // !ENABLE_ONEDNN_V3
          auto reorder_prim =
              dnnl::reorder(input_bias_mem, scaled_bias_mem, bias_attr);
          std::unordered_map<int, memory> reorder_net_args = {
              {DNNL_ARG_FROM, input_bias_mem}, {DNNL_ARG_TO, scaled_bias_mem}};
#ifdef ENABLE_ONEDNN_V3
          auto scale_mem =
              memory({{1}, MklDnnType<float>(), memory::format_tag::x},
                     this->cpu_engine_, bias_scales.data());
          reorder_net_args.insert(
              {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, scale_mem});
#endif  // ENABLE_ONEDNN_V3
          reorder_prim.execute(dnnl::stream(this->cpu_engine_),
                               reorder_net_args);
        }

        *bias_data = temp_scaled_bias_tensor->flat<float>().data();

        if (is_bias_cache_empty) {
          // Only try to cache the bias in the first iteration.
          this->CacheBias(ctx, *temp_scaled_bias_tensor, min_input, max_input);
        }
      }
    }
  }

  bool IsCachedBiasValid(float current_min_input,
                         float current_max_input) override
      TF_LOCKS_EXCLUDED(this->bias_cache_mutex_) {
    tf_shared_lock lock(this->bias_cache_mutex_);
    if (this->is_bias_const_ && this->is_weight_const_ &&
        std::abs(current_min_input - this->saved_min_input_) < 1e-5 &&
        std::abs(current_max_input - this->saved_max_input_) < 1e-5) {
      return true;
    }
    return false;
  }
};

// Register mkl kernels for supported operations and types.
#define REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES(type)    \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_MklFusedMatMul")                                     \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<type>("T")                              \
          .Label(mkl_op_registry::kMklLayoutDependentOpLabel),    \
      MklFusedMatMulOp<CPUDevice, type, type, type, type, type>); \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("_MklNativeFusedMatMul")                               \
          .Device(DEVICE_CPU)                                     \
          .TypeConstraint<type>("T")                              \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),         \
      MklFusedMatMulOp<CPUDevice, type, type, type, type, type, true>);
TF_CALL_float(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_bfloat16(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
TF_CALL_half(REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES);
#undef REGISTER_FUSEDMATMUL_MKL_SUPPORTED_KERNELS_TYPES

#define REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type,       \
                                  output_type, additional_type)             \
  REGISTER_KERNEL_BUILDER(                                                  \
      Name("_QuantizedMatMul")                                              \
          .Device(DEVICE_CPU)                                               \
          .TypeConstraint<input_type>("T1")                                 \
          .TypeConstraint<weight_type>("T2")                                \
          .TypeConstraint<bias_type>("Tbias")                               \
          .TypeConstraint<output_type>("Tout")                              \
          .TypeConstraint<additional_type>("U"),                            \
      QuantizedFusedMatMulOp<CPUDevice, input_type, weight_type, bias_type, \
                             output_type, additional_type, true>);

#define REGISTER_ALL_OUTPUT_TYPES(input_type, weight_type, bias_type,     \
                                  additional_type)                        \
  REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type, qint8,    \
                            additional_type)                              \
  REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type, quint8,   \
                            additional_type)                              \
  REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type, qint32,   \
                            additional_type)                              \
  REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type, float,    \
                            additional_type)                              \
  REGISTER_QUANTIZED_MATMUL(input_type, weight_type, bias_type, bfloat16, \
                            additional_type)

#define REGISTER_ALL_BIAS_OUTPUT_TYPES(input_type, weight_type,              \
                                       additional_type)                      \
  REGISTER_ALL_OUTPUT_TYPES(input_type, weight_type, float, additional_type) \
  REGISTER_ALL_OUTPUT_TYPES(input_type, weight_type, bfloat16,               \
                            additional_type)                                 \
  REGISTER_ALL_OUTPUT_TYPES(input_type, weight_type, qint32, additional_type)

#define REGISTER_ALL_INPUT_BIAS_OUTPUT_TYPES(weight_type, additional_type) \
  REGISTER_ALL_BIAS_OUTPUT_TYPES(qint8, weight_type, additional_type)      \
  REGISTER_ALL_BIAS_OUTPUT_TYPES(quint8, weight_type, additional_type)

REGISTER_ALL_INPUT_BIAS_OUTPUT_TYPES(qint8, float);
REGISTER_ALL_INPUT_BIAS_OUTPUT_TYPES(qint8, bfloat16);

}  // namespace tensorflow

#endif  // INTEL_MKL
