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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// inv = InvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

template <typename T>
class InvertPermutationOp : public OpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* context)
      : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64_t>("T"),
    InvertPermutationOp<int64_t>);

REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64_t>);

namespace {
template <typename Tperm>
absl::Status PermutationHelper(const Tensor& perm, const int dims,
                               std::vector<int32>* permutation) {
  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return absl::OkStatus();
}
}  // namespace

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

void TransposeOp::Compute(OpKernelContext* ctx) {
  const Tensor& input = ctx->input(0);
  const Tensor& perm = ctx->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be rank 1, got shape ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(ctx, PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(ctx, PermutationHelper<int64_t>(perm, dims, &permutation));
  }
  TensorShape shape;

  // Check whether permutation is a permutation of integers of [0 .. dims).
  absl::InlinedVector<bool, 8UL> bits(dims);
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    int32_t d = permutation[i];
    if (d < 0) {
      d += dims;
      permutation[i] = d;
    }
    OP_REQUIRES(
        ctx, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    OP_REQUIRES_OK(ctx, shape.AddDimWithStatus(dim_size));
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(ctx, bits[i],
                errors::InvalidArgument(i, " is missing from {",
                                        absl::StrJoin(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!IsConjugate() && (dims <= 1 || is_identity)) {
    ctx->set_output(0, input);
    return;
  } else if (!IsConjugate() && internal::NonSingletonDimensionsAlign(
                                   input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                errors::Unknown("Error reshaping Tensor."));
    ctx->set_output(0, output);
    return;
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
  }
}

absl::Status TransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                         absl::Span<const int32> perm,
                                         Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

absl::Status ConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                                  const Tensor& in,
                                                  absl::Span<const int32> perm,
                                                  Tensor* out) {
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeCpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
TF_CALL_float8_e5m2(REGISTER) TF_CALL_float8_e4m3fn(REGISTER)
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    Status TransposeGpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                       gtl::ArraySlice<int32> perm,
                                       Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<GPUDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeGpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<GPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeGpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeGpuOp);
TF_CALL_POD_TYPES(REGISTER);
TF_CALL_float8_e5m2(REGISTER);
TF_CALL_float8_e4m3fn(REGISTER);
#undef REGISTER
#endif

}  // namespace tensorflow
