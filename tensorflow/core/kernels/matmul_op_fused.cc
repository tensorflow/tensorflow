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

// Implements matmul operations with other kernels baked into the
// processing, to optimize latency and memory usage:
//  - MatMul + BiasAdd + <Activation>
//  - MatMul + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...
//
// Currently supported only on CPU device.

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/util/tensor_format.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
struct LaunchFusedMatMulOp {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output);
};

template <typename T>
struct LaunchFusedMatMulOp<CPUDevice, T> {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output) {
    auto lhs = a.matrix<T>();
    auto rhs = b.matrix<T>();
    auto out = output->matrix<T>();

    auto& d = context->eigen_device<CPUDevice>();

    // Executes Eigen contraction with output kernel wrapped into type erased
    // wrapper to reduce the number of unique template instantiations.
    auto executeWithOutputKernel = [&](auto output_kernel) {
      OutputKernelWrapper output_kernel_wrapper(
          [&output_kernel](
              const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
              const Eigen::TensorContractionParams& params, Eigen::Index i,
              Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) {
            output_kernel(output_mapper, params, i, j, num_rows, num_cols);
          });

      out.device(d) = lhs.contract(rhs, dim_pair, output_kernel_wrapper);
    };

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      if (fusion == FusedComputationType::kBiasAddWithLeakyRelu) {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args,
                                                &fusion_args.leakyrelu_alpha));
      } else {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
      }
    }

    switch (fusion) {
      case FusedComputationType::kBiasAdd:
        executeWithOutputKernel(WithBiasAdd<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu:
        executeWithOutputKernel(WithBiasAddAndRelu<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        executeWithOutputKernel(WithBiasAddAndRelu6<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithElu:
        executeWithOutputKernel(WithBiasAddAndElu<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithLeakyRelu:
        executeWithOutputKernel(WithBiasAddAndLeakyRelu<T>(bias_add_args));
        break;
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      default:
        OP_REQUIRES_OK(context,
                       errors::Internal("Fusion type is not supported"));
    }
  }

 private:
  // Wrap output_kernel into type erased struct to reduce the number of unique
  // template instantiations for Eigen Tensor contraction expressions.
  //
  // We do not pass std::function directly as an output kernel because it blows
  // up the binary size in debug mode with super long symbol names.
  struct OutputKernelWrapper {
    using OutputKernelFn =
        std::function<void(const ContractionOutputMapper<T, Eigen::Index>&,
                           const Eigen::TensorContractionParams&, Eigen::Index,
                           Eigen::Index, Eigen::Index, Eigen::Index)>;

    explicit OutputKernelWrapper(OutputKernelFn fn)
        : output_kernel_fn(std::move(fn)) {}

    void operator()(
        const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
        const Eigen::TensorContractionParams& params, Eigen::Index i,
        Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) const {
      output_kernel_fn(output_mapper, params, i, j, num_rows, num_cols);
    }

    OutputKernelFn output_kernel_fn;
  };
};

template <typename Device, typename T>
class FusedMatMulOp : public OpKernel {
 public:
  explicit FusedMatMulOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));

    std::vector<FusedComputationPattern> patterns;

    using FCT = FusedComputationType;
    if (std::is_same<Device, CPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
      };
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "MatMul", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a.shape()),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b.shape()),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b.shape().DebugString()));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    auto launch = LaunchFusedMatMulOp<Device, T>();
    launch(ctx, a, b, dim_pair, fused_computation_, fused_computation_args_,
           out);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedMatMulOp);
};

// Registration of the CPU implementations.
#define REGISTER_FUSED_CPU_MATMUL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedMatMulOp<CPUDevice, T>);

TF_CALL_float(REGISTER_FUSED_CPU_MATMUL);

#undef REGISTER_FUSED_CPU_MATMUL

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_
