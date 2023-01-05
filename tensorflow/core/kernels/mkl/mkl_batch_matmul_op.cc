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

#define EIGEN_USE_THREADS

#if defined(INTEL_MKL)

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/mkl/mkl_batch_matmul_helper.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
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
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMulMkl() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(ctx, lhs.dims() == rhs.dims(),
                  errors::InvalidArgument("lhs and rhs has different ndims: ",
                                          lhs.shape().DebugString(), " vs. ",
                                          rhs.shape().DebugString()));
      const int ndims = lhs.dims();
      OP_REQUIRES(
          ctx, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                    errors::InvalidArgument(
                        "lhs.dim(", i, ") and rhs.dim(", i,
                        ") must be the same: ", lhs.shape().DebugString(),
                        " vs ", rhs.shape().DebugString()));
      }
    } else {
      OP_REQUIRES(
          ctx, lhs.dims() >= 2,
          errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
      OP_REQUIRES(
          ctx, rhs.dims() >= 2,
          errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
    }

    // lhs and rhs can have different dimensions
    const auto ndims_lhs = lhs.dims();
    const auto ndims_rhs = rhs.dims();

    // Get broadcast info
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);
    OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                errors::InvalidArgument(
                    "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                    ": ", lhs.shape().DebugString(), " ",
                    rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);
    // The maximum number of DNNL tensor dimensions is DNNL_MAX_NDIMS = 12.
    OP_REQUIRES(
        ctx, out_shape.dims() <= DNNL_MAX_NDIMS,
        errors::InvalidArgument(
            "Rank of output tensor must be <= 12, but is ", out_shape.dims(),
            ". Current implementation supports upto rank 12 tensors."));

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
    MklDnnThreadPool eigen_tp(ctx);
    // Create or retrieve matmul primitive from cache.
    MklMatMulPrimitive<Tlhs, Trhs, Toutput>* matmul_prim =
        MklMatMulPrimitiveFactory<float, Tlhs, Trhs, Toutput>::Get(
            *params, false /* value for do_not_cache */);

    Trhs* weight_data = const_cast<Trhs*>(rhs.flat<Trhs>().data());

// TODO(Arm, Intel): Reach agreement on whether this block should be deleted.
// https://github.com/tensorflow/tensorflow/pull/57987#discussion_r993731524
#ifdef DNNL_AARCH64_USE_ACL
    memory::format_tag weight_format;
    switch (params->b_dims.size()) {
      case 2:
        weight_format =
            adj_y_ ? memory::format_tag::ba : memory::format_tag::ab;
        break;
      case 3:
        weight_format =
            adj_y_ ? memory::format_tag::acb : memory::format_tag::abc;
        break;
      case 4:
        weight_format =
            adj_y_ ? memory::format_tag::abdc : memory::format_tag::abcd;
        break;
      case 5:
        weight_format =
            adj_y_ ? memory::format_tag::abced : memory::format_tag::abcde;
        break;
      default:
        weight_format = memory::format_tag::undef;
    }
    MklDnnData<Trhs> weights_mkl(&(this->cpu_engine_));
    if (weight_format != memory::format_tag::undef) {
      auto weight_md =
          memory::desc(params->b_dims, MklDnnType<Trhs>(), weight_format);
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
    }
#endif  // DNNL_AARCH64_USE_ACL

    UserScratchPad<unsigned char> scratch_pad;
    scratch_pad.AllocateSPTensor(matmul_prim, ctx);
    // Execute matmul primitive.
    std::shared_ptr<stream> cpu_stream;
    cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
    if (fused_ops_.size() > 0) {
      void* mul_data = nullptr;
      void* add_data = nullptr;
      if (fused_ops_.at(0) == "Mul") {
        const Tensor& mul_tensor = ctx->input(2);
        mul_data = static_cast<void*>(
            const_cast<Toutput*>(mul_tensor.flat<Toutput>().data()));
      }
      if (fused_ops_.size() > 1 && fused_ops_.at(1) == "Add") {
        const Tensor& add_tensor = ctx->input(3);
        add_data = static_cast<void*>(
            const_cast<Toutput*>(add_tensor.flat<Toutput>().data()));
      }
      matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                           out->flat<Toutput>().data(), scratch_pad.Get(),
                           mul_data, add_data);
    } else {
      matmul_prim->Execute(cpu_stream, lhs.flat<Tlhs>().data(), weight_data,
                           out->flat<Toutput>().data(), scratch_pad.Get());
    }
  }

  engine cpu_engine_ = engine(engine::kind::cpu, 0);

 protected:
  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {}
  std::vector<string> fused_ops_;

 private:
  bool adj_x_;
  bool adj_y_;
};

template <typename Device, typename Tlhs, typename Trhs, typename Toutput,
          bool v2_bcast>
class FusedBatchMatMulMkl
    : public BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast> {
 public:
  explicit FusedBatchMatMulMkl(OpKernelConstruction* context)
      : BatchMatMulMkl<Device, Tlhs, Trhs, Toutput, v2_bcast>(context) {
    OP_REQUIRES_OK(context, context->GetAttr("fused_ops", &this->fused_ops_));
    OP_REQUIRES(context, !this->fused_ops_.empty(),
                errors::InvalidArgument(
                    "Fused BatchMatMul must have at least one fused op."));

    int num_args;
    OP_REQUIRES_OK(context, context->GetAttr("num_args", &num_args));

    if (this->fused_ops_ == std::vector<string>{"Mul"} ||
        this->fused_ops_ == std::vector<string>{"Mul", "Add"}) {
      OP_REQUIRES(context, num_args == this->fused_ops_.size(),
                  errors::InvalidArgument(
                      "Fused BatchMatmul should have same number of additional "
                      "inputs as the number of fusions"));
    } else {
      OP_REQUIRES(
          context, false,
          errors::Unimplemented("Fusion is not implemented: [",
                                absl::StrJoin(this->fused_ops_, ","), "]"));
    }
  }

  virtual ~FusedBatchMatMulMkl() {}

 protected:
  virtual void ExtendMklMatMulParams(OpKernelContext* ctx,
                                     MklMatMulParams& params) {
    if (this->fused_ops_.size() > 0) {
      const Tensor& scale_tensor = ctx->input(2);
      OP_REQUIRES(ctx, scale_tensor.NumElements() == 1,
                  errors::InvalidArgument("Scale tensor must be a scalar"));

      memory::data_type data_type = MklDnnType<Toutput>();
      memory::format_tag format_tag;
      switch (params.c_dims.size()) {
        case 3:
          format_tag = memory::format_tag::abc;
          break;
        case 4:
          format_tag = memory::format_tag::abcd;
          break;
        default:
          OP_REQUIRES(ctx, false, errors::Unimplemented("Unimplemented"));
      }
      if (this->fused_ops_.at(0) == "Mul") {
        memory::dims mul_dims(params.c_dims.size(), 1);
        params.post_op_params.push_back(
            {"mul", {}, mul_dims, data_type, format_tag});
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument(
                        "Currently first fusion is supported only for Mul",
                        ", but it is ", this->fused_ops_.at(0), " op."));
      }
      if (this->fused_ops_.size() > 1 && this->fused_ops_.at(1) == "Add") {
        auto add_shape = ctx->input(3).shape();
        memory::dims add_dims = {add_shape.dim_size(0), add_shape.dim_size(1),
                                 add_shape.dim_size(2), add_shape.dim_size(3)};
        params.post_op_params.push_back(
            {"add", {}, add_dims, data_type, format_tag});
      } else {
        OP_REQUIRES(ctx, false,
                    errors::InvalidArgument(
                        "Currently second fusion is supported only for Add",
                        ", but it is ", this->fused_ops_.at(1), " op."));
      }
    }
  }
};

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
      FusedBatchMatMulMkl<CPUDevice, TYPE, TYPE, TYPE, true>)

#ifdef INTEL_MKL
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_float(REGISTER_FUSED_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_bfloat16(REGISTER_FUSED_BATCH_MATMUL_MKL);
#endif  // INTEL_MKL

}  // end namespace tensorflow
#endif
