/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_KERNELS_TRANSPOSE_OP_H_
#define TENSORFLOW_KERNELS_TRANSPOSE_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

class TransposeOp : public OpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override;

 protected:
  virtual Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                             gtl::ArraySlice<int32> perm, Tensor* out) = 0;
  virtual bool IsConjugate() const { return false; }
};

class TransposeCpuOp : public TransposeOp {
 public:
  explicit TransposeCpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
};

#ifdef INTEL_MKL
class MklTransposeCpuOp : public TransposeOp {
 public:
  explicit MklTransposeCpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
};
#endif  // INTEL_MKL

class TransposeGpuOp : public TransposeOp {
 public:
  explicit TransposeGpuOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
};

#ifdef TENSORFLOW_USE_SYCL
class TransposeSyclOp : public TransposeOp {
 public:
  explicit TransposeSyclOp(OpKernelConstruction* ctx) : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
};
#endif  // TENSORFLOW_USE_SYCL

// Conjugating transpose ops.
class ConjugateTransposeCpuOp : public TransposeOp {
 public:
  explicit ConjugateTransposeCpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};

#ifdef INTEL_MKL
class MklConjugateTransposeCpuOp : public TransposeOp {
 public:
  explicit MklConjugateTransposeCpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};
#endif  // INTEL_MKL

class ConjugateTransposeGpuOp : public TransposeOp {
 public:
  explicit ConjugateTransposeGpuOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};

#ifdef TENSORFLOW_USE_SYCL
class ConjugateTransposeSyclOp : public TransposeOp {
 public:
  explicit ConjugateTransposeSyclOp(OpKernelConstruction* ctx)
      : TransposeOp(ctx) {}

 protected:
  Status DoTranspose(OpKernelContext* ctx, const Tensor& in,
                     gtl::ArraySlice<int32> perm, Tensor* out) override;
  bool IsConjugate() const override { return true; }
};
#endif  // TENSORFLOW_USE_SYCL

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TRANSPOSE_OP_H_
