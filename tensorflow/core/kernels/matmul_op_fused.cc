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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/core/util/tensor_format.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/kernels/matmul_op_impl.h"
#include "tensorflow/core/kernels/matmul_util.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
struct LaunchFusedMatMulOp {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output, bool use_autotune);
};

template <typename T>
struct LaunchFusedMatMulOp<CPUDevice, T> {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output, bool use_autotune) {
    OP_REQUIRES(context, DataTypeToEnum<T>::value != DT_HALF,
                errors::InvalidArgument("_FusedMatMul doesn't support DT_HALF "
                                        "data type on CPU devices."));
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

#if GOOGLE_CUDA
namespace {

StatusOr<se::cuda::BlasLt::Epilogue> GetBlasLtEpilogOp(
    FusedComputationType fusion) {
  if (fusion == FusedComputationType::kBiasAdd) {
    return se::cuda::BlasLt::Epilogue::kBias;
  } else if (fusion == FusedComputationType::kBiasAddWithRelu) {
    return se::cuda::BlasLt::Epilogue::kBiasThenReLU;
  } else if (fusion == FusedComputationType::kBiasAddWithGeluApproximate) {
    return se::cuda::BlasLt::Epilogue::kBiasThenGeLUApproximate;
  } else {
    return errors::Internal("Unsupported fusion for BlasLt Matmul");
  }
}

template <typename LaunchFunc>
se::blas::AlgorithmConfig AutotuneMatmul(
    const std::vector<se::cuda::BlasLt::MatmulAlgorithm>& algorithms,
    const se::cuda::BlasLt::MatmulPlanParams& matmul_params,
    OpKernelContext* context, const LaunchFunc& launch_func) {
  // Note that algorithm_config.algorithm() here is used to refer
  // to the index within the algorithms vector, not the algorithm
  // itself.
  se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);
  if (!AutoTuneBatchMatmul::GetInstance()->Find(matmul_params,
                                                &algorithm_config)) {
    VLOG(4) << "Autotuning BlasLtMatmul over " << algorithms.size()
            << " algorithms.";
    se::blas::ProfileResult best_result;
    se::blas::ProfileResult profile_result;

    for (size_t i = 0; i != algorithms.size(); ++i) {
      const auto& profile_algorithm = algorithms[i];

      // Create a new scratch allocator with every autotuning run so that
      // scratch space is deallocated between runs.
      BlasScratchAllocator scratch_allocator(context);

      bool cublaslt_launch_ok =
          launch_func(&scratch_allocator, profile_algorithm, &profile_result);

      VLOG(4) << "  Autotune algorithm " << i
              << " result: " << profile_result.elapsed_time_in_ms()
              << " ms, valid=" << profile_result.is_valid()
              << ", workspace_size=" << profile_algorithm.workspace_size();

      if (cublaslt_launch_ok && profile_result.is_valid() &&
          profile_result.elapsed_time_in_ms() <
              best_result.elapsed_time_in_ms()) {
        best_result = profile_result;
      }
    }

    if (best_result.is_valid()) {
      algorithm_config.set_algorithm(best_result.algorithm());
    }
    // We make sure that each matmul parameter set only gets one pass of
    // autotune. If no algorithms works, we add kNoAlgorithm to the autotune
    // map.
    AutoTuneBatchMatmul::GetInstance()->Insert(matmul_params, algorithm_config);
  }
  return algorithm_config;
}

}  // namespace

template <typename T>
struct LaunchFusedMatMulOp<GPUDevice, T> {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output, bool use_autotune) {
    OP_REQUIRES(
        context, DataTypeToEnum<T>::value != DT_BFLOAT16,
        errors::InvalidArgument("_FusedMatMul doesn't support "
                                "DT_BFLOAT16 data type on CPU devices."));
    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    // All fusion patterns supported by GPU are in the form of MatMul + BiasAdd
    // + <other pointwise operations>. Therefore, the bias tensor is required.
    const Tensor& bias = context->input(2);

    if (bias.dims() != 1) {
      OP_REQUIRES_OK(context,
                     errors::InvalidArgument("bias must be 1-dimensional",
                                             bias.shape().DebugString()));
    }

    auto a_ptr = AsDeviceMemory(a.template flat<T>().data(),
                                a.template flat<T>().size());
    auto b_ptr = AsDeviceMemory(b.template flat<T>().data(),
                                b.template flat<T>().size());
    auto bias_ptr = AsDeviceMemory(bias.template flat<T>().data(),
                                   bias.template flat<T>().size());
    auto c_ptr = AsDeviceMemory(output->template flat<T>().data(),
                                output->template flat<T>().size());

    auto epilog_op_or = GetBlasLtEpilogOp(fusion);
    OP_REQUIRES_OK(context, epilog_op_or.status());
    se::cuda::BlasLt::Epilogue epilog_op = epilog_op_or.ValueOrDie();

    bool trans_a = dim_pair[0].first == 0 ? true : false;
    bool trans_b = dim_pair[0].second == 1 ? true : false;

    const int64_t m = a.dim_size(trans_a ? 1 : 0);
    const int64_t k = a.dim_size(trans_a ? 0 : 1);
    const int64_t n = b.dim_size(trans_b ? 0 : 1);

    se::blas::DataType blas_dtype = se::blas::ToDataType<T>::value;

    DataType dtype = DataTypeToEnum<T>::value;
    auto status_or_computation_type = GetBlasComputationType(dtype);
    OP_REQUIRES(context, status_or_computation_type.ok(),
                errors::Internal("Unsupported dtype for batched matmul."));
    se::blas::ComputationType computation_type =
        status_or_computation_type.ConsumeValueOrDie();

    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   se::blas::Transpose::kTranspose};

    se::cuda::BlasLt::MatmulPlanParams matmul_params{
        /*ab_type=*/blas_dtype,
        /*c_type=*/blas_dtype,
        computation_type,
        se::cuda::BlasLt::PointerMode::kHost,
        epilog_op,
        trans[trans_b ? 1 : 0],
        trans[trans_a ? 1 : 0],
        static_cast<uint64_t>(n),
        static_cast<uint64_t>(m),
        static_cast<uint64_t>(k),
        trans_b ? k : n,
        trans_a ? m : k,
        static_cast<int64_t>(n),
        /*batch_size=*/1,
        static_cast<int64_t>(k * n),
        static_cast<int64_t>(m * k),
        static_cast<int64_t>(m * n)};

    auto plan_and_algorithms_or = GetPlanAndAlgorithms(stream, matmul_params);
    OP_REQUIRES_OK(context, plan_and_algorithms_or.status());
    const auto* plan_and_algorithms = std::move(plan_and_algorithms_or).value();

    const auto& plan = plan_and_algorithms->plan;
    const auto& algorithms = plan_and_algorithms->algorithms;
    OP_REQUIRES(context, algorithms.size() > 0,
                errors::InvalidArgument("No matmul algorithm returned!"));

    T alpha(1.0);
    T beta(0.0);

    se::cuda::BlasLt* blas_lt = se::cuda::GetBlasLt(stream);
    OP_REQUIRES(context, blas_lt != nullptr,
                errors::Internal("blaslt not supported"));

    auto launch_func = [&](BlasScratchAllocator* scratch_allocator,
                           const se::cuda::BlasLt::MatmulAlgorithm& algorithm,
                           se::blas::ProfileResult* profile_result) -> bool {
      return blas_lt->DoMatmul(stream, plan, alpha, b_ptr, a_ptr, beta, c_ptr,
                               scratch_allocator, algorithm, bias_ptr,
                               profile_result);
    };

    se::cuda::BlasLt::MatmulAlgorithm algorithm = algorithms[0];
    if (use_autotune) {
      se::blas::AlgorithmConfig algorithm_config =
          AutotuneMatmul(algorithms, matmul_params, context, launch_func);

      se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
      algorithm = algorithms[algorithm_idx];
    }

    BlasScratchAllocator scratch_allocator(context);

    bool cublaslt_launch_ok =
        launch_func(&scratch_allocator, algorithm, nullptr);
    if (!cublaslt_launch_ok) {
      OP_REQUIRES_OK(context,
                     errors::Internal("BlasLt Matmul launch failed : a.shape=",
                                      a.shape().DebugString(),
                                      ", b.shape=", b.shape().DebugString(),
                                      ", m=", m, ", n=", n, ", k=", k));
    }
  }
};

#endif  // GOOGLE_CUDA

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
    } else if (std::is_same<Device, GPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithGeluApproximate, {"BiasAdd", "GeluApproximate"}}};
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "MatMul", patterns,
                                &fused_computation_, &fused_computation_args_));
    use_autotune_ = MatmulAutotuneEnable();
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, a.dims() == b.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        a.shape().DebugString(), " vs. ",
                                        b.shape().DebugString()));
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
           out, use_autotune_);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  bool use_autotune_;

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

#if GOOGLE_CUDA

// Registration of the GPU implementations.
#define REGISTER_FUSED_GPU_MATMUL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedMatMul").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      FusedMatMulOp<GPUDevice, T>);

TF_CALL_float(REGISTER_FUSED_GPU_MATMUL);
TF_CALL_half(REGISTER_FUSED_GPU_MATMUL);

#undef REGISTER_FUSED_GPU_MATMUL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_
