/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This file uses oneDNN library for acceleration of Batch Matrix-Matrix
// Multiplication (MatMul) operations. We currently register this kernel only
// for oneDNN supported data types (float, bfloat16). The maximum number of
// dimensions (rank) for output tensor is DNNL_MAX_NDIMS = 12 in oneDNN.
// If output tensor rank exceeds 12, we exit with reporting an error message.

#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/mkl/mkl_batch_matmul_helper.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool v2_bcast>
class BatchMatMulMkl : public OpKernel {
 public:
  explicit BatchMatMulMkl(OpKernelConstruction* context) : OpKernel(context) {
    if (!context) return;

    if (context->HasAttr("transpose_a")) {
      // This is needed for using BatchMatMulMkl as the super class of
      // MklMatMulOp (below) whose context has a transpose_a attribute which is
      // effectively the same as adj_x_
      OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &adj_x_));
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    }

    if (context->HasAttr("transpose_b")) {
      // This is needed for using BatchMatMulMkl as the super class of
      // MklMatMulOp (below) whose context has a transpose_b attribute which is
      // effectively the same as adj_y_
      OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &adj_y_));
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
    }
  }

  virtual ~BatchMatMulMkl() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);

    if (std::is_same<Tlhs, float>::value) {
      (void)SetFPMathMode();
    }

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(
          ctx, lhs.dims() == rhs.dims(),
          absl::InvalidArgumentError(absl::StrCat(
              "In[0] and In[1] has different ndims: ",
              lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString())));
      const int ndims = lhs.dims();
      OP_REQUIRES(ctx, ndims >= 2,
                  absl::InvalidArgumentError(absl::StrCat(
                      "In[0] and In[1] ndims must be >= 2: ", ndims)));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                    absl::InvalidArgumentError(absl::StrCat(
                        "In[0].dim(", i, ") and In[1].dim(", i,
                        ") must be the same: ", lhs.shape().DebugString(),
                        " vs ", rhs.shape().DebugString())));
      }
    } else {
      OP_REQUIRES(ctx, lhs.dims() >= 2,
                  absl::InvalidArgumentError(
                      absl::StrCat("In[0] ndims must be >= 2: ", lhs.dims())));
      OP_REQUIRES(ctx, rhs.dims() >= 2,
                  absl::InvalidArgumentError(
                      absl::StrCat("In[1] ndims must be >= 2: ", rhs.dims())));
    }

    // lhs and rhs can have different dimensions
    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();

    // Get broadcast info
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        absl::InvalidArgumentError(absl::StrCat(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString())));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);
    OP_REQUIRES(
        ctx, lhs_cols == rhs_rows,
        absl::InvalidArgumentError(absl::StrCat(
            "Matrix size-incompatible: In[0]: ", lhs.shape().DebugString(),
            ", In[1]: ", rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_)));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);
    // The maximum number of DNNL tensor dimensions is DNNL_MAX_NDIMS = 12.
    OP_REQUIRES(
        ctx, out_shape.dims() <= DNNL_MAX_NDIMS,
        absl::InvalidArgumentError(absl::StrCat(
            "Rank of output tensor must be <= 12, but is ", out_shape.dims(),
            ". Current implementation supports upto rank 12 tensors.")));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Toutput> f;
      f(ctx->eigen_device<Device>(), out->flat<Toutput>());
      return;
    }

    // Compute parameters for DNNL matmul primitive.
    MklBatchMatMulHelper bmm;
    string prefix = "batchmatmul";
    auto params = bmm.CreateMatMulParams(prefix, lhs.shape(), rhs.shape(),
                                         out_shape, adj_x_, adj_y_);

    this->ExtendMklMatMulParams(ctx, *params);
    // Create the oneDNN wrapper over Eigen threadpool and set max threads
    // in oneDNN.
    Eigen::ThreadPoolInterface* eigen_interface =
        EigenThreadPoolFromTfContext(ctx);
    tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                   ThreadPoolUseCallerThread());
    // Create or retrieve matmul primitive from cache.
    MklMatMulPrimitive<Tlhs, Trhs, Toutput>* matmul_prim =
        MklMatMulPrimitiveFactory<float, Tlhs, Trhs, Toutput>::Get(
            *params, false /* value for do_not_cache */);

    Trhs* weight_data = const_cast<Trhs*>(rhs.flat<Trhs>().data());
// TODO(Arm, Intel): Reach agreement on whether this block should be deleted.
// https://github.com/tensorflow/tensorflow/pull/57987#discussion_r993731524
#ifdef DNNL_AARCH64_USE_ACL
    MklDnnData<Trhs> weights_mkl(&(this->cpu_engine_));
    auto weight_md =
        memory::desc(params->b_dims, MklDnnType<Trhs>(), params->b_strides);
    std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd =
        matmul_prim->GetPrimitiveDesc();
    // Reorder weights if necessary.
    // Check whether we need to do reorder.
    if (weight_md != matmul_pd->weights_desc()) {
      weights_mkl.SetUsrMem(weight_md, weight_data);
      weights_mkl.CheckReorderToOpMem(matmul_pd.get()->weights_desc(),
                                      this->cpu_engine_, ctx);
      weight_data =
          reinterpret_cast<Trhs*>(weights_mkl.GetOpMem().get_data_handle());
    }

#endif  // DNNL_AARCH64_USE_ACL

    UserScratchPad<unsigned char> scratch_pad;
    scratch_pad.AllocateSPTensor(matmul_prim, ctx);
    // Execute matmul primitive.
    std::shared_ptr<stream> cpu_stream;
    cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
    matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                         out->flat<Toutput>().data(), *params,
                         scratch_pad.Get(), this->fusion_data_);
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 protected:
  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {}
  std::vector<void*> fusion_data_;

 private:
  bool adj_x_;
  bool adj_y_;
};

// OneDNN uses post-ops to implement different kind of fusions. The category of
// each individual post-op can be inferred from the fused_ops attribute. The
// following enum is used to identify list of required post-ops.
namespace {

enum class FusedComputationType {
  kUndefined,
  kMul,
  kAdd,
  kMulAdd,
  kDequantize,
  kMul_Dequantize,
  kAdd_Dequantize,
  kMulAdd_Dequantize,
  kRequantize,
  kMul_Requantize,
  kAdd_Requantize,
  kMulAdd_Requantize,
};

struct FusedComputationPattern {
  FusedComputationType fused_computation;
  std::vector<string> fused_ops;
};

}  // namespace
enum class PostOpKind { kNone, kOutputScale, kMul, kAdd, kLinear };

// FusedBatchMatMul has additional inputs, currently forcing all the operands
// of fusion to have same type `U`.
template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          /*type of additional tensors*/ typename U, bool v2_bcast>
class FusedBatchMatMulMkl
    : public BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast> {
 public:
  explicit FusedBatchMatMulMkl(OpKernelConstruction* context)
      : BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast>(context) {
    InitializeFusion(context);
  }

  virtual ~FusedBatchMatMulMkl() {}

 protected:
  struct PostOpInfo {
    PostOpKind post_op_kind;
    int input_idx = -1;  // Operand tensor index if needed by a post-op.
  };

  std::vector<PostOpInfo> post_op_info_list_;

  // This function is called from constructor.
  void InitializeFusion(OpKernelConstruction* context) {
    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    OP_REQUIRES(context, !fused_ops.empty(),
                absl::InvalidArgumentError(
                    "Fused BatchMatMul must have at least one fused op."));

    using FCT = FusedComputationType;
    // TODO(intel-tf): Add more patterns when implemented. Refactor for
    // arbitrary fusion sequence when oneDNN is performant.
    std::vector<FusedComputationPattern> patterns{
        {FCT::kMul, {"Mul"}},
        {FCT::kAdd, {"Add"}},
        {FCT::kMulAdd, {"Mul", "Add"}},
    };
    FusedComputationType fused_computation = FusedComputationType::kUndefined;
    for (const auto& pattern : patterns) {
      if (fused_ops == pattern.fused_ops) {
        fused_computation = pattern.fused_computation;
        break;
      }
    }

    // Configure oneDNN post-ops. Refactor for arbitrary fusion sequence when
    // oneDNN is performant.
    switch (fused_computation) {
      case FCT::kMul:
        post_op_info_list_ = {{PostOpKind::kMul, 2}};
        break;
      case FCT::kAdd:
        post_op_info_list_ = {{PostOpKind::kAdd, 2}};
        break;
      case FCT::kMulAdd:
        post_op_info_list_ = {{PostOpKind::kMul, 2}, {PostOpKind::kAdd, 3}};
        break;
      default:
        OP_REQUIRES_OK(context, absl::UnimplementedError(absl::StrCat(
                                    "Fusion is not implemented: [",
                                    absl::StrJoin(fused_ops, ","), "]")));
    }

    int num_args = 0;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));
    this->fusion_data_.resize(num_args);
  }

  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {
    int idx = 0;
    for (const auto& post_op_info : this->post_op_info_list_) {
      switch (post_op_info.post_op_kind) {
        case PostOpKind::kMul: {
          const Tensor& multiplicand_tensor =
              ctx->input(post_op_info.input_idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported = multiplicand_tensor.NumElements() == 1 &&
                              params.c_dims.size() == 4;
          OP_REQUIRES(ctx, is_supported,
                      absl::UnimplementedError(absl::StrCat(
                          "Unimplemented multiplicand shape for Mul fusion: ",
                          multiplicand_tensor.shape().DebugString())));
          auto format_tag = memory::format_tag::abcd;
          memory::data_type data_type = MklDnnType<U>();
          memory::dims mul_dims(params.c_dims.size(), 1);
          params.post_op_params.push_back(
              {"mul", {}, mul_dims, data_type, format_tag});
          void* multiplicand_data = static_cast<void*>(
              const_cast<U*>(multiplicand_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = multiplicand_data;
        } break;
        case PostOpKind::kAdd: {
          const Tensor& addend_tensor = ctx->input(post_op_info.input_idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported = params.c_dims.size() == 4 &&
                              addend_tensor.dims() == params.c_dims.size();
          OP_REQUIRES(ctx, is_supported,
                      absl::UnimplementedError(absl::StrCat(
                          "Unimplemented addend shape for Add fusion: ",
                          addend_tensor.shape().DebugString())));
          auto format_tag = memory::format_tag::abcd;
          memory::data_type data_type = MklDnnType<U>();
          memory::dims addend_dims = TFShapeToMklDnnDims(addend_tensor.shape());
          params.post_op_params.push_back(
              {"add", {}, addend_dims, data_type, format_tag});
          void* addend_data = static_cast<void*>(
              const_cast<U*>(addend_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = addend_data;
        } break;
        default:
          OP_REQUIRES_OK(ctx,
                         absl::UnimplementedError("Unsupported post-op-kind."));
      }
    }
  }
};

template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          typename U>
class QuantizedBatchMatMulOp
    : public BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, /*v2_bcast*/ true> {
 public:
  explicit QuantizedBatchMatMulOp(OpKernelConstruction* context)
      : BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, true>(context) {
    InitializeFusion(context);
  }

  void Compute(OpKernelContext* ctx) override {
    BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, true>::Compute(ctx);
    if (std::is_same<Toutput, qint8>::value ||
        std::is_same<Toutput, quint8>::value) {
      Tensor* min_output = nullptr;
      Tensor* max_output = nullptr;
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
    } else if (std::is_same<Toutput, float>::value ||
               std::is_same<Toutput, bfloat16>::value) {
      // Kernel is registered for Dequantization fusion. Nothing to do.
    } else {
      OP_REQUIRES_OK(ctx,
                     absl::InvalidArgumentError("Unsupported output type."));
    }
  }

  virtual ~QuantizedBatchMatMulOp() {}

 protected:
  string input_quant_mode_;  // Both lhs and rhs are quantized with same mode.
  string output_quant_mode_;

  // Initialize minmax tensor indices with default values for the most common
  // cases.
  int lhs_min_idx_ = 2;
  int lhs_max_idx_ = 3;
  int rhs_min_idx_ = 4;
  int rhs_max_idx_ = 5;

  struct PostOpInfo {
    PostOpKind post_op_kind;
    struct OperandInfo {
      int idx = -1;  // Operand tensor index if needed by a post-op.
      // Indices of min and max value tensors, if the operand is quantized.
      std::vector<int> min_max_indices;
    } operand_info;
    // Indices of output min and max value tensors. It is used when requantize
    // is fused.
    std::vector<int> min_max_indices;
  };

  std::vector<PostOpInfo> post_op_info_list_;

  int num_operands_;  // Number of regular operands without minmax tensors.

  void UpdateInputMinMaxIndices(int offset) {
    lhs_min_idx_ += offset;
    lhs_max_idx_ += offset;
    rhs_min_idx_ += offset;
    rhs_max_idx_ += offset;
  }

  void InitializeFusion(OpKernelConstruction* context) {
    // Currently, tensor quantized with only SCALED mode is supported.
    OP_REQUIRES_OK(context, context->GetAttr("input_quant_mode",
                                             &this->input_quant_mode_));
    OP_REQUIRES(context, input_quant_mode_ == "SCALED",
                absl::UnimplementedError(
                    "Input tensors are not quantized with SCALED mode."));
    OP_REQUIRES_OK(context, context->GetAttr("output_quant_mode",
                                             &this->output_quant_mode_));

    std::vector<string> fused_ops;
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &fused_ops));
    if (fused_ops.empty())
      OP_REQUIRES_OK(context,
                     absl::InvalidArgumentError(
                         "Fused BatchMatMul must have at least one fused op."));

    using FCT = FusedComputationType;
    // TODO(intel-tf): Add more patterns when implemented.
    std::vector<FusedComputationPattern> patterns{
        {FCT::kDequantize, {"Dequantize"}},
        {FCT::kMul_Dequantize, {"Mul", "Dequantize"}},
        {FCT::kAdd_Dequantize, {"Add", "Dequantize"}},
        {FCT::kMulAdd_Dequantize, {"Mul", "Add", "Dequantize"}},
        {FCT::kRequantize, {"Requantize"}},
        {FCT::kMul_Requantize, {"Mul", "Requantize"}},
        {FCT::kAdd_Requantize, {"Add", "Requantize"}},
        {FCT::kMulAdd_Requantize, {"Mul", "Add", "Requantize"}},
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
      case FCT::kDequantize: {
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}}};
      } break;
      case FCT::kRequantize: {
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kLinear, {}, {6, 7}}};
      } break;
      case FCT::kMul_Dequantize: {
        this->UpdateInputMinMaxIndices(1);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kMul, {2, {}}, {}}};
        this->fusion_data_.resize(1);
      } break;
      case FCT::kMul_Requantize: {
        this->UpdateInputMinMaxIndices(1);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kMul, {2, {}}, {}},
                              {PostOpKind::kLinear, {}, {7, 8}}};
        this->fusion_data_.resize(1);
      } break;
      case FCT::kAdd_Dequantize: {
        this->UpdateInputMinMaxIndices(1);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kAdd, {2, {}}, {}}};
        this->fusion_data_.resize(1);
      } break;
      case FCT::kAdd_Requantize: {
        this->UpdateInputMinMaxIndices(1);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kAdd, {2, {}}, {}},
                              {PostOpKind::kLinear, {}, {7, 8}}};
        this->fusion_data_.resize(1);
      } break;
      case FCT::kMulAdd_Dequantize: {
        this->UpdateInputMinMaxIndices(2);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kMul, {2, {}}, {}},
                              {PostOpKind::kAdd, {3, {}}, {}}};
        this->fusion_data_.resize(2);
      } break;
      case FCT::kMulAdd_Requantize: {
        this->UpdateInputMinMaxIndices(2);
        post_op_info_list_ = {{PostOpKind::kOutputScale, {}, {}},
                              {PostOpKind::kMul, {2, {}}, {}},
                              {PostOpKind::kAdd, {3, {}}, {}},
                              {PostOpKind::kLinear, {}, {8, 9}}};
        this->fusion_data_.resize(2);
      } break;
      default:
        OP_REQUIRES(context, false,
                    absl::UnimplementedError(
                        absl::StrCat("Fusion is not implemented: [",
                                     absl::StrJoin(fused_ops, ","), "]")));
    }
  }

  void ExtendMklMatMulParams(OpKernelContext* ctx,
                             MklMatMulParams& params) override {
    int idx = 0;
    for (const auto& post_op_info : post_op_info_list_) {
      switch (post_op_info.post_op_kind) {
        case PostOpKind::kMul: {
          const Tensor& multiplicand_tensor =
              ctx->input(post_op_info.operand_info.idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported = multiplicand_tensor.NumElements() == 1 &&
                              params.c_dims.size() == 4;
          OP_REQUIRES(ctx, is_supported,
                      absl::UnimplementedError("Unimplemented"));
          auto format_tag = memory::format_tag::abcd;
          memory::data_type data_type = MklDnnType<U>();
          memory::dims mul_dims(params.c_dims.size(), 1);
          params.post_op_params.push_back(
              {"mul", {}, mul_dims, data_type, format_tag});
          void* multiplicand_data = static_cast<void*>(
              const_cast<U*>(multiplicand_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = multiplicand_data;
        } break;
        case PostOpKind::kAdd: {
          const Tensor& addend_tensor =
              ctx->input(post_op_info.operand_info.idx);
          // TODO(intel-tf): Relax restriction when oneDNN is performant for
          // arbitrary shapes.
          bool is_supported = params.c_dims.size() == 4 &&
                              addend_tensor.dims() == params.c_dims.size();
          OP_REQUIRES(ctx, is_supported,
                      absl::UnimplementedError("Unimplemented."));
          auto format_tag = memory::format_tag::abcd;
          memory::data_type data_type = MklDnnType<U>();
          memory::dims addend_dims = TFShapeToMklDnnDims(addend_tensor.shape());
          params.post_op_params.push_back(
              {"add", {}, addend_dims, data_type, format_tag});
          void* addend_data = static_cast<void*>(
              const_cast<U*>(addend_tensor.flat<U>().data()));
          this->fusion_data_[idx++] = addend_data;
        } break;
        case PostOpKind::kOutputScale: {
          const Tensor& lhs_min_tensor = ctx->input(lhs_min_idx_);
          const Tensor& lhs_max_tensor = ctx->input(lhs_max_idx_);
          const Tensor& rhs_min_tensor = ctx->input(rhs_min_idx_);
          const Tensor& rhs_max_tensor = ctx->input(rhs_max_idx_);
          // Currently, only per tensor quantization supported.
          OP_REQUIRES(ctx,
                      lhs_min_tensor.NumElements() == 1 &&
                          lhs_max_tensor.NumElements() == 1 &&
                          rhs_min_tensor.NumElements() == 1 &&
                          rhs_max_tensor.NumElements() == 1,
                      absl::UnimplementedError(
                          "Only supported is per-tensor quantization."));

          const float min_lhs = lhs_min_tensor.flat<float>()(0);
          const float max_lhs = lhs_max_tensor.flat<float>()(0);
          const float min_rhs = rhs_min_tensor.flat<float>()(0);
          const float max_rhs = rhs_max_tensor.flat<float>()(0);

          const float range_lhs =
              (input_quant_mode_ == "MIN_FIRST")
                  ? (max_lhs - min_lhs)
                  : std::max(std::abs(min_lhs), std::abs(max_lhs));
          const float range_rhs =
              (input_quant_mode_ == "MIN_FIRST")
                  ? (max_rhs - min_rhs)
                  : std::max(std::abs(min_rhs), std::abs(max_rhs));
          const float max_int8_lhs =
              (std::is_same<Tlhs, quint8>::value) ? 255.0f : 127.0f;
          const float max_int8_rhs =
              (std::is_same<Trhs, quint8>::value) ? 255.0f : 127.0f;
          const float lhs_scale = range_lhs / max_int8_lhs;
          const float rhs_scale = range_rhs / max_int8_rhs;
#ifndef ENABLE_ONEDNN_V3
          const float output_scale = lhs_scale * rhs_scale;
          params.post_op_params.push_back({"output_scale", { output_scale }});
#else
          const float dst_scale = 1.0;
          params.post_op_params.push_back({"lhs_scale", {lhs_scale}});
          params.post_op_params.push_back({"rhs_scale", {rhs_scale}});
          params.post_op_params.push_back({"dst_scale", {dst_scale}});
#endif  // !ENABLE_ONEDNN_V3
        } break;
        case PostOpKind::kLinear: {
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
        default:
          OP_REQUIRES(ctx, false,
                      absl::UnimplementedError("Unsupported post-op-kind."));
      }
    }
  }
};

// Direct calls for MklMatMulOp to BatchMatMulMkl for aarch64,
// because the Arm Compute Library does not provide a BLAS SGEMM
// interface, which is what MklMatMulOp calls by default.
#ifdef DNNL_AARCH64_USE_ACL
template <typename Device, typename T, bool USE_CUBLAS>
class MklMatMulOp : public BatchMatMulMkl<Device, T, T, T, USE_CUBLAS> {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx)
      : BatchMatMulMkl<Device, T, T, T, false>(ctx) {}

  virtual ~MklMatMulOp() {}
};

#define REGISTER_MATMUL_MKL(TYPE)                         \
  REGISTER_KERNEL_BUILDER(                                \
      Name("_MklMatMul")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<TYPE>("T")                      \
          .Label(mkl_op_registry::kMklNameChangeOpLabel), \
      MklMatMulOp<CPUDevice, TYPE, false /* cublas, ignored for CPU */>);

#endif  // DNNL_AARCH64_USE_ACL

#define REGISTER_BATCH_MATMUL_MKL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMul")                             \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, false>)

#define REGISTER_BATCH_MATMUL_MKL_V2(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMulV2")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, true>)

#define REGISTER_FUSED_BATCH_MATMUL_MKL(TYPE) \
  REGISTER_KERNEL_BUILDER(                    \
      Name("_MklFusedBatchMatMulV2")          \
          .Device(DEVICE_CPU)                 \
          .TypeConstraint<TYPE>("T"),         \
      FusedBatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, TYPE, true>)

TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_float(REGISTER_FUSED_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_bfloat16(REGISTER_FUSED_BATCH_MATMUL_MKL);
TF_CALL_half(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_half(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_half(REGISTER_FUSED_BATCH_MATMUL_MKL);

#ifdef DNNL_AARCH64_USE_ACL
TF_CALL_float(REGISTER_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_MATMUL_MKL);
#endif  // DNNL_AARCH64_USE_ACL

#define REGISTER_QUANTIZED_KERNEL(U, T) \
  REGISTER_KERNEL_BUILDER(              \
      Name("_QuantizedBatchMatMul")     \
          .Device(DEVICE_CPU)           \
          .TypeConstraint<qint8>("T1")  \
          .TypeConstraint<qint8>("T2")  \
          .TypeConstraint<U>("U")       \
          .TypeConstraint<T>("Tout"),   \
      QuantizedBatchMatMulOp<CPUDevice, qint8, qint8, T, U>);

REGISTER_QUANTIZED_KERNEL(float, float);
REGISTER_QUANTIZED_KERNEL(float, qint8);
REGISTER_QUANTIZED_KERNEL(float, quint8);
REGISTER_QUANTIZED_KERNEL(bfloat16, bfloat16);
REGISTER_QUANTIZED_KERNEL(bfloat16, qint8);
REGISTER_QUANTIZED_KERNEL(bfloat16, quint8);

}  // end namespace tensorflow
#endif  // INTEL_MKL
