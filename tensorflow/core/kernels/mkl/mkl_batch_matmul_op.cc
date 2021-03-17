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
// dimensions (rank) for output tensor is 12 in oneDNN. If output tensor rank
// exceeds 12, we fall back to Eigen library based kernel.

#define EIGEN_USE_THREADS

#if defined(INTEL_MKL)

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
template <typename Device, typename Scalar, bool v2_bcast>
class BatchMatMulMkl : public OpKernel {
 public:
  explicit BatchMatMulMkl(OpKernelConstruction* context)
      : OpKernel(context), eigen_batch_mm_v2_(context) {
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
    // The maximum number of dimensions for a tensor in DNNL is 12.
    OP_REQUIRES(
        ctx, out_shape.dims() <= 12,
        errors::InvalidArgument(
            "Rank of output tensor must be <= 12, but is ", out_shape.dims(),
            ". Current implementation supports upto rank 12 tensors."));

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    // Compute parameters for DNNL matmul primitive.
    auto params = CreateMatMulParams(lhs.shape(), rhs.shape(), out_shape);
    // Create or retrieve matmul primitive from cache.
    MklMatMulPrimitive<Scalar>* matmul_prim =
        MklMatMulPrimitiveFactory<Scalar>::Get(
            *params, false /* value for do_not_cache */);
    // Execute matmul primitive.
    std::shared_ptr<stream> cpu_stream;
    MklDnnThreadPool eigen_tp(ctx);
    cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));
    matmul_prim->Execute(lhs.flat<Scalar>().data(), rhs.flat<Scalar>().data(),
                         out->flat<Scalar>().data(), cpu_stream);
  }

 private:
  bool adj_x_;
  bool adj_y_;
  BatchMatMulV2Op<CPUDevice, Scalar> eigen_batch_mm_v2_;

  using dims = dnnl::memory::dims;

  // This method makes the rank (ndims) of input same as the output by adding
  // new axes to the input. For example, if input shape is [a, b, c, d] and
  // output shape is [e, f, g, h, i, j], then the reshaped input would have a
  // shape of [1, 1, a, b, c, d].
  void ExpandInputDimsToOutputShape(const TensorShape& input_shape,
                                    const TensorShape& output_shape,
                                    dims* reshaped_dims) {
    auto ndims_input = input_shape.dims();
    auto ndims_output = output_shape.dims();
    auto dim_offset = ndims_output - ndims_input;
    DCHECK(dim_offset > 0);
    reshaped_dims->clear();
    reshaped_dims->resize(ndims_output, 1);
    auto input_dims = input_shape.dim_sizes();
    for (int dim_idx = 0; dim_idx < ndims_input; ++dim_idx)
      reshaped_dims->at(dim_idx + dim_offset) = input_dims[dim_idx];
  }

  std::unique_ptr<MklMatMulParams> CreateMatMulParams(
      const TensorShape& lhs_shape, const TensorShape& rhs_shape,
      const TensorShape& out_shape) {
    const auto ndims_lhs = lhs_shape.dims();
    const auto ndims_rhs = rhs_shape.dims();
    const auto ndims_out = out_shape.dims();
    auto lhs_dims = TFShapeToMklDnnDims(lhs_shape);
    auto rhs_dims = TFShapeToMklDnnDims(rhs_shape);
    auto out_dims = TFShapeToMklDnnDims(out_shape);

    // DNNL matmul_primitive requires ranks of inputs and output to be same.
    // Create dnnl::memory::dims for inputs and output of same rank.
    // It is assumed here that MatMulBCast object creates output_batch_shape as
    // a conforming superset of input batch shapes, i.e., ndims_out >=
    // ndims_lhs and ndims_out >= ndims_rhs.
    if (ndims_lhs < ndims_out) {
      ExpandInputDimsToOutputShape(lhs_shape, out_shape, &lhs_dims);
    }
    if (ndims_rhs < ndims_out) {
      ExpandInputDimsToOutputShape(rhs_shape, out_shape, &rhs_dims);
    }

    using dim = dnnl::memory::dim;
    dim m;  // number of rows in x
    dim k;  // number of columns in x
    dim n;  // number of columns in y
    auto lhs_strides = CalculateTFStrides(lhs_dims);
    auto rhs_strides = CalculateTFStrides(rhs_dims);
    auto out_strides = CalculateTFStrides(out_dims);

    if (adj_x_) {
      int m_idx = ndims_out - 1;
      int k_idx = ndims_out - 2;
      m = lhs_dims[m_idx];
      k = lhs_dims[k_idx];
      std::swap(lhs_dims[m_idx], lhs_dims[k_idx]);
      lhs_strides[m_idx] = m;
      lhs_strides[k_idx] = 1;
    }

    if (adj_y_) {
      int k_idx = ndims_out - 1;
      int n_idx = ndims_out - 2;
      k = rhs_dims[k_idx];
      n = rhs_dims[n_idx];
      std::swap(rhs_dims[k_idx], rhs_dims[n_idx]);
      rhs_strides[k_idx] = k;
      rhs_strides[n_idx] = 1;
    }
    return std::make_unique<MklMatMulParams>(
        lhs_dims, rhs_dims, out_dims, lhs_strides, rhs_strides, out_strides);
  }
};

#define REGISTER_BATCH_MATMUL_MKL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMul")                             \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, false>)

#define REGISTER_BATCH_MATMUL_MKL_V2(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMulV2")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, true>)
#ifdef ENABLE_MKL
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL_V2);
#endif  // ENABLE_MKL

}  // end namespace tensorflow
#endif
