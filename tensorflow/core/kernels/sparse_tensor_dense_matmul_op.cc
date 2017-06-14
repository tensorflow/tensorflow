/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_tensor_dense_matmul_op.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/kernels/fill_functor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T, typename Tindices>
class SparseTensorDenseMatMulOp : public OpKernel {
 public:
  explicit SparseTensorDenseMatMulOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_a", &adjoint_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("adjoint_b", &adjoint_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* a_indices;
    const Tensor* a_values;
    const Tensor* a_shape;
    const Tensor* b;
    OP_REQUIRES_OK(ctx, ctx->input("a_indices", &a_indices));
    OP_REQUIRES_OK(ctx, ctx->input("a_values", &a_values));
    OP_REQUIRES_OK(ctx, ctx->input("a_shape", &a_shape));
    OP_REQUIRES_OK(ctx, ctx->input("b", &b));

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrixOrHigher(b->shape()),
                errors::InvalidArgument("Tensor 'b' is not a matrix or higher"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_shape->shape()),
                errors::InvalidArgument("Tensor 'a_shape' is not a vector"));

    OP_REQUIRES(
        ctx, a_shape->NumElements() >= 2,
        errors::InvalidArgument("Tensor 'a_shape' must have at least 2 elements"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsVector(a_values->shape()),
                errors::InvalidArgument("Tensor 'a_values' is not a vector"));

    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a_indices->shape()),
                errors::InvalidArgument("Tensor 'a_indices' is not a matrix"));

    const int64 nnz = a_indices->shape().dim_size(0);
    OP_REQUIRES(ctx, nnz == a_values->NumElements(),
                errors::InvalidArgument("Number of rows of a_indices does not "
                                        "match number of entries in a_values"));

    OP_REQUIRES(
        ctx, a_indices->shape().dim_size(1) == a_shape->NumElements(),
        errors::InvalidArgument("Number of columns of a_indices does not match "
                                "number of entries in a_shape"));
                                
    OP_REQUIRES(ctx, b->dims() == a_shape->NumElements(),
                errors::InvalidArgument("Rank of 'b' does not match rank of 'a'"));
                                
    const int64 ndim = a_shape->NumElements();

    auto a_shape_t = a_shape->vec<int64>();
    const int64 outer_left = (adjoint_a_) ? a_shape_t(ndim-1) : a_shape_t(ndim-2);
    const int64 outer_right =
        (adjoint_b_) ? b->shape().dim_size(ndim-2) : b->shape().dim_size(ndim-1);
    const int64 inner_left = (adjoint_a_) ? a_shape_t(ndim-2) : a_shape_t(ndim-1);
    const int64 inner_right =
        (adjoint_b_) ? b->shape().dim_size(ndim-1) : b->shape().dim_size(ndim-2);
        
    for (int64 i = 0; i < ndim - 2; ++i) {
      OP_REQUIRES(
          ctx, b->dim_size(i) == a_shape_t(i),
          errors::InvalidArgument(
              "Cannot multiply A and B because (batch) dimension ", i,
              " does not match: ", b->dim_size(i), " vs. ", a_shape_t(i)));
    }

    OP_REQUIRES(
        ctx, inner_right == inner_left,
        errors::InvalidArgument(
            "Cannot multiply A and B because inner dimension does not match: ",
            inner_left, " vs. ", inner_right,
            ".  Did you forget a transpose?  "
            "Dimensions of A: [",
            a_shape_t(0), ", ", a_shape_t(1),
            ").  Dimensions of B: ", b->shape().DebugString()));

    if (std::is_same<Device, GPUDevice>::value) {
      // The GPU implementation is optimized to use 32 bit indexing, so
      // give a friendly error to the programmer early on if they
      // exceed.
      const int int32max = std::numeric_limits<int>::max();
      OP_REQUIRES(
          ctx,
          (FastBoundsCheck(inner_left, int32max) &&
           FastBoundsCheck(inner_right, int32max) &&
           FastBoundsCheck(outer_left, int32max) &&
           FastBoundsCheck(outer_right, int32max) &&
           FastBoundsCheck(b->NumElements(), int32max) &&
           FastBoundsCheck(outer_left * outer_right, int32max) &&
           FastBoundsCheck(a_values->NumElements(), int32max)),
          errors::InvalidArgument("Cannot use GPU for > 2^31 entry inputs"));
      OP_REQUIRES(ctx, FastBoundsCheck(nnz * outer_right, int32max),
                  errors::InvalidArgument(
                      "Cannot use GPU when output.shape[1] * nnz(a) > 2^31"));
    }

    TensorShape out_shape(b->shape());
    out_shape.set_dim(ndim - 2, outer_left);
    out_shape.set_dim(ndim - 1, outer_right);
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a_values->NumElements() == 0 || b->NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    switch (ndim) {
#define MAYBE_ADJOINT(ADJ_A, ADJ_B, N)                               \
  if (adjoint_a_ == ADJ_A && adjoint_b_ == ADJ_B) {                  \
    Status functor_status = functor::SparseTensorDenseMatMulFunctor< \
        Device, T, Tindices, ADJ_A, ADJ_B, N>::Compute(              \
            ctx->eigen_device<Device>(),                             \
            out->tensor<T, N>(),                                     \
            a_indices->matrix<Tindices>(),                           \
            a_values->vec<T>(), b->tensor<T, N>());                  \
    OP_REQUIRES_OK(ctx, functor_status);                             \
  }
  
#define NDIMS_CASE(N)                 \
  case N: {                           \
    MAYBE_ADJOINT(false, false, N);   \
    MAYBE_ADJOINT(false, true, N);    \
    MAYBE_ADJOINT(true, false, N);    \
    MAYBE_ADJOINT(true, true, N);     \
  } break;
    
      NDIMS_CASE(2);
      NDIMS_CASE(3);
      NDIMS_CASE(4);
      NDIMS_CASE(5);
      default:
        OP_REQUIRES(ctx, false, errors::InvalidArgument(
                                    "Only tensors with ranks between 2 and 5 "
                                    "are currently supported.  Tensor rank: ",
                                    ndim));

#undef NDIMS_CASE
#undef MAYBE_ADJOINT
    }
  }

 private:
  bool adjoint_a_;
  bool adjoint_b_;
};

#define REGISTER_CPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_CPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<CPUDevice, TypeT, TypeIndex>);

#define REGISTER_KERNELS_CPU(T) \
  REGISTER_CPU(T, int64);       \
  REGISTER_CPU(T, int32)

REGISTER_KERNELS_CPU(float);
REGISTER_KERNELS_CPU(double);
REGISTER_KERNELS_CPU(int32);
REGISTER_KERNELS_CPU(complex64);
REGISTER_KERNELS_CPU(complex128);

#if GOOGLE_CUDA

namespace functor {
#define DECLARE_GPU_SPEC(T, Tindices, ADJ_A, ADJ_B, NDIM)                      \
  template <>                                                                  \
  Status SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices, ADJ_A,         \
                                        ADJ_B, NDIM>::Compute(                 \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor out,                \
      typename TTypes<Tindices>::ConstMatrix a_indices,                        \
      typename TTypes<T>::ConstVec a_values,                                   \
      typename TTypes<T, NDIM>::ConstTensor b);                                \
  extern template struct SparseTensorDenseMatMulFunctor<GPUDevice, T, Tindices,\
                                                        ADJ_A, ADJ_B, NDIM>;

#define REGISTER_GPU_SPEC(T, ADJ_A, ADJ_B, NDIM)  \
  DECLARE_GPU_SPEC(T, int32, ADJ_A, ADJ_B, NDIM); \
  DECLARE_GPU_SPEC(T, int64, ADJ_A, ADJ_B, NDIM)

#define DECLARE_ADJOINT_GPU_SPEC(T, NDIM)  \
  REGISTER_GPU_SPEC(T, false, false, NDIM) \
  REGISTER_GPU_SPEC(T, false, true, NDIM)  \
  REGISTER_GPU_SPEC(T, true, false, NDIM)  \
  REGISTER_GPU_SPEC(T, true, true, NDIM)

DECLARE_ADJOINT_GPU_SPEC(float, 2);
DECLARE_ADJOINT_GPU_SPEC(float, 3);
DECLARE_ADJOINT_GPU_SPEC(float, 4);
DECLARE_ADJOINT_GPU_SPEC(float, 5);
  
#undef DECLARE_ADJOINT_GPU_SPEC
#undef DECLARE_GPU_SPEC
#undef REGISTER_GPU_SPEC

}  // namespace functor

#define REGISTER_GPU(TypeT, TypeIndex)           \
  REGISTER_KERNEL_BUILDER(                       \
      Name("SparseTensorDenseMatMul")            \
          .Device(DEVICE_GPU)                    \
          .TypeConstraint<TypeT>("T")            \
          .TypeConstraint<TypeIndex>("Tindices") \
          .HostMemory("a_shape"),                \
      SparseTensorDenseMatMulOp<GPUDevice, TypeT, TypeIndex>);

#define REGISTER_KERNELS_GPU(T) \
  REGISTER_GPU(T, int64);       \
  REGISTER_GPU(T, int32)

REGISTER_KERNELS_GPU(float);
#undef REGISTER_GPU
#undef REGISTER_KERNELS_GPU
#endif  // GOOGLE_CUDA

namespace functor {

namespace {
Status BatchIndexOutOfBoundsError(int64 ind, std::size_t i, int j,
                                  std::size_t a_dim_size) {
  return errors::InvalidArgument("a_index (", ind, ") from index[", i, ",", j,
                                 "] out of bounds (>=", a_dim_size, ")");
}

Status KOutOfBoundsError(int64 k, std::size_t i, int rhs_index_a,
                         std::size_t lhs_right) {
  return errors::InvalidArgument("k (", k, ") from index[", i, ",", rhs_index_a,
                                 "] out of bounds (>=", lhs_right, ")");
}

Status MOutOfBoundsError(int64 m, std::size_t i, int lhs_index_a,
                         int64 out_dim0) {
  return errors::InvalidArgument("m (", m, ") from index[", i, ",", lhs_index_a,
                                 "] out of bounds (>=", out_dim0, ")");
}
}  // namespace

template <typename T, typename Tindices, bool ADJ_A, bool ADJ_B, int NDIM>
struct SparseTensorDenseMatMulFunctor<CPUDevice, T, Tindices, ADJ_A, ADJ_B, NDIM> {
  // Vectorize certain operations above this size.
  static const std::size_t kNumVectorize = 32;

  static Status Compute(const CPUDevice& d, typename TTypes<T, NDIM>::Tensor out,
                        typename TTypes<Tindices>::ConstMatrix a_indices,
                        typename TTypes<T>::ConstVec a_values,
                        typename TTypes<T, NDIM>::ConstTensor b) {
    const std::size_t nnz = a_values.size();
    const std::size_t rhs_right = (ADJ_B ? b.dimension(NDIM-2) : b.dimension(NDIM-1));
    const std::size_t lhs_right = (ADJ_B ? b.dimension(NDIM-1) : b.dimension(NDIM-2));
    const int lhs_index_a = ADJ_A ? NDIM-1 : NDIM-2;
    const int rhs_index_a = ADJ_A ? NDIM-2 : NDIM-1;

    out.setZero();

    // TODO(ebrevdo): After many failed experiments, can't find a multi-threaded
    // approach that achieves the performance of the single threaded
    // one.  Perhaps Eigen threadpool implementation is just too slow?

    if (rhs_right < kNumVectorize) {
      // Disable vectorization if the RHS of output is too small
      auto maybe_adjoint_b = MaybeAdjoint<decltype(b), ADJ_B, NDIM>(b);

      for (std::size_t i = 0; i < nnz; ++i) {
        Eigen::array<Eigen::Index, NDIM> indices;
        for (int j = 0; j < NDIM; ++j) {
          indices[j] = internal::SubtleMustCopy(a_indices(i, j));
        }
        const Tindices m = indices[lhs_index_a];
        const Tindices k = indices[rhs_index_a];
        
        // bounds checks
        for (int j = 0; j < NDIM - 2; ++j) {
          if(!FastBoundsCheck(indices[j], b.dimension(j))) {
            return BatchIndexOutOfBoundsError(indices[j], i, j, b.dimension(j));
          }
        }
        if (!FastBoundsCheck(k, lhs_right)) {
          return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);
        }
        if (!FastBoundsCheck(m, out.dimension(NDIM - 2))) {
          return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(NDIM - 2));
        }
        
        const T a_value = ADJ_A ? MaybeConj(a_values(i)) : a_values(i);
        for (std::size_t n = 0; n < rhs_right; ++n) {
          indices[NDIM - 2] = k;
          indices[NDIM - 1] = n;
          const T b_value = maybe_adjoint_b(indices);
          indices[NDIM - 2] = m;
          out(indices) += a_value * b_value;
        }
      }
    } else {
      // Vectorization via Eigen.
      for (std::size_t i = 0; i < nnz; ++i) {
        Eigen::array<Eigen::Index, NDIM> out_offsets;
        Eigen::array<Eigen::Index, NDIM> out_extends;
        Eigen::array<Eigen::Index, NDIM> b_offsets;
        Eigen::array<Eigen::Index, NDIM> b_extends;
        for (int j = 0; j < NDIM; ++j) {
          out_offsets[j] = internal::SubtleMustCopy(a_indices(i, j));
          out_extends[j] = 1;
          b_offsets[j] = internal::SubtleMustCopy(a_indices(i, j));
          b_extends[j] = 1;
        }
        const Tindices m = out_offsets[lhs_index_a];
        const Tindices k = out_offsets[rhs_index_a];
        const T a_value = (ADJ_A) ? MaybeConj(a_values(i)) : a_values(i);
        
        // bounds checks
        for (int j = 0; j < NDIM - 2; ++j) {
          if (!FastBoundsCheck(out_offsets[j], b.dimension(j))) {
            return BatchIndexOutOfBoundsError(out_offsets[j], i, j, b.dimension(j));
          }
        }
        if (!FastBoundsCheck(k, lhs_right)) {
          return KOutOfBoundsError(k, i, rhs_index_a, lhs_right);
        }
        if (!FastBoundsCheck(m, out.dimension(NDIM - 2))) {
          return MOutOfBoundsError(m, i, lhs_index_a, out.dimension(NDIM - 2));
        }
        
        out_offsets[NDIM - 2] = m;
        out_offsets[NDIM - 1] = 0;
        out_extends[NDIM - 2] = 1;
        out_extends[NDIM - 1] = rhs_right;
        b_offsets[NDIM - 2] = ADJ_B ? 0 : k;
        b_offsets[NDIM - 1] = ADJ_B ? k : 0;
        b_extends[NDIM - 2] = ADJ_B ? rhs_right : 1;
        b_extends[NDIM - 1] = ADJ_B ? 1 : rhs_right;
        Eigen::array<Eigen::Index, 1> vector_size;
        vector_size[0] = rhs_right;
        out.slice(out_offsets, out_extends).reshape(vector_size) +=
            (ADJ_B ?
                b.slice(b_offsets, b_extends).conjugate() :
                b.slice(b_offsets, b_extends)).reshape(vector_size)
            * a_value;
      }
    }
    return Status::OK();
  }
};

}  // namespace functor

}  // namespace tensorflow
